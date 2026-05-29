from __future__ import annotations
from dataclasses import dataclass, replace
from enum import Enum
from combo_bot.types import (
    AccountState, EMAState, Order, OrderSource, Side,
    SymbolState, ExchangeParams,
)
from combo_bot.grid_engine import quantize_qty
from combo_bot.hsl import HslConfig, HslSupervisor, HslTier


class RiskTier(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"


@dataclass
class RiskConfig:
    max_drawdown_pct: float = 0.25
    yellow_threshold: float = 0.10
    orange_threshold: float = 0.18
    red_threshold: float = 0.25
    max_total_wallet_exposure: float = 3.0
    max_single_exposure: float = 0.5
    max_realized_loss_pct: float = 0.05
    liquidation_threshold: float = 0.05
    cooldown_after_red_minutes: int = 60
    # Stage 4 per-source circuit breakers. When a bucket's drawdown
    # (measured from its own peak, normalized to wallet balance) exceeds
    # the threshold, new entries from that source are dropped; reduce-only
    # orders always pass. Set to 1.0+ to disable the breaker for that
    # source. Trend defaults are tighter than grid because overlay
    # positions can size up fast and reverse hard in choppy regimes.
    pause_trend_dd_pct: float = 0.15
    pause_grid_dd_pct: float = 0.30
    # Stage 5: passivbot-style controlled exposure shedding ("unstuck").
    # When a bucket's wallet exposure exceeds threshold * its WEL, emit
    # small reduce-only limit orders priced outside the EMA band — but
    # only fire when current price has reverted that far (so we sell
    # into a bounce, not into the bleed). Throttled by a 24h rolling
    # loss budget so a sustained drawdown can't realize unlimited losses.
    unstuck_threshold: float = 0.90
    unstuck_close_pct: float = 0.01
    unstuck_ema_dist: float = 0.02
    daily_loss_allowance_pct: float = 0.01
    # WEL ceiling for the trend bucket — separate from the grid bucket's
    # GridConfig.wallet_exposure_limit. Used by the unstuck mechanism to
    # decide when the trend bucket is "stuck". Mirrors MergerConfig
    # .trend_position_max_pct but lives here so the risk layer can stay
    # self-contained.
    trend_wallet_exposure_limit: float = 0.15
    # Stage 6: passivbot-style hard-stop drawdown smoothing.
    # ``dd_ema_span_minutes <= 0`` disables smoothing (use raw dd, legacy
    # behavior). When enabled, ``score = min(raw, ema)`` — the minimum
    # protects against two failure modes simultaneously:
    #   * a single-tick flash spike (raw briefly > red) doesn't latch RED
    #     because ema is still low;
    #   * a stale EMA after recovery doesn't keep us in RED because raw
    #     has dropped below the threshold again.
    # When ``red_latch_enabled`` is True, entering RED *latches* the tier
    # — RED stays RED until ``reset_red_latch()`` is called. This mirrors
    # passivbot's ``red_latched`` and is the difference between a bot
    # that survives a deep drawdown and one that re-enters during the
    # bleed because a cooldown timer ticked over.
    dd_ema_span_minutes: float = 30.0
    red_latch_enabled: bool = True


class RiskManager:
    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()
        # HSL tier classification is delegated to a standalone supervisor
        # so the drawdown decision surface can be tested, persisted, and
        # reasoned about independently of enforcement actions.
        self._hsl = HslSupervisor(
            HslConfig(
                red_threshold=self.config.red_threshold,
                yellow_threshold=self.config.yellow_threshold,
                orange_threshold=self.config.orange_threshold,
                dd_ema_span_minutes=self.config.dd_ema_span_minutes,
                red_latch_enabled=self.config.red_latch_enabled,
                cooldown_after_red_minutes=self.config.cooldown_after_red_minutes,
            )
        )

    # ------------------------------------------------------------------
    # Backward-compatible properties (delegated to _hsl)
    # ------------------------------------------------------------------
    @property
    def tier(self) -> RiskTier:
        return RiskTier(self._hsl.tier.value)

    @tier.setter
    def tier(self, value: RiskTier) -> None:
        # Allow external state restore (live._load_state, tests).
        self._hsl.tier = HslTier(value.value)

    @property
    def red_latched(self) -> bool:
        return self._hsl.red_latched

    @red_latched.setter
    def red_latched(self, value: bool) -> None:
        self._hsl.red_latched = value

    @property
    def red_cooldown_until(self) -> int:
        return self._hsl.red_cooldown_until

    @red_cooldown_until.setter
    def red_cooldown_until(self, value: int) -> None:
        self._hsl.red_cooldown_until = value

    @property
    def red_latched_at_ms(self) -> int:
        return self._hsl.red_latched_at_ms

    @red_latched_at_ms.setter
    def red_latched_at_ms(self, value: int) -> None:
        self._hsl.red_latched_at_ms = value

    @property
    def dd_ema(self) -> float:
        return self._hsl.dd_ema

    @dd_ema.setter
    def dd_ema(self, value: float) -> None:
        self._hsl.dd_ema = value

    @property
    def last_assess_minute(self) -> int:
        return self._hsl._last_assess_minute

    @last_assess_minute.setter
    def last_assess_minute(self, value: int) -> None:
        self._hsl._last_assess_minute = value

    # _dd_initialized is accessed by live._save_state / _load_state.
    @property
    def _dd_initialized(self) -> bool:
        return self._hsl._dd_initialized

    @_dd_initialized.setter
    def _dd_initialized(self, value: bool) -> None:
        self._hsl._dd_initialized = value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_red_latch(self) -> None:
        """Clear the RED latch and any active cooldown.

        Intended for explicit operator action (or for tests). Re-entry
        into RED requires the drawdown threshold to be crossed again.
        Cooldown is reset too — an explicit reset means "fully clear",
        otherwise the lingering cooldown would still block entries.
        """
        self._hsl.reset_red_latch()

    def assess(
        self, account: AccountState, timestamp_ms: int = 0
    ) -> RiskTier:
        """Classify the current account drawdown.

        Delegates to :class:`HslSupervisor.assess` and converts the
        result to the legacy :class:`RiskTier` enum for backward
        compatibility.
        """
        hsl_tier = self._hsl.assess(account, timestamp_ms)
        return RiskTier(hsl_tier.value)

    def filter_orders(
        self, orders: list[Order], account: AccountState, timestamp: int = 0
    ) -> list[Order]:
        tier = self.assess(account, timestamp)

        if tier == RiskTier.RED:
            self.red_cooldown_until = timestamp + self.config.cooldown_after_red_minutes * 60_000
            return self._panic_close_all(account)

        if timestamp < self.red_cooldown_until:
            return [o for o in orders if o.reduce_only]

        if tier == RiskTier.ORANGE:
            return [o for o in orders if o.reduce_only]

        # Per-source circuit breakers, with thresholds RELAXED when the
        # global tier is GREEN. Old behaviour treated source pause as
        # tier-independent, so a 15% trend-bucket drawdown paused trend
        # entries even with the account at all-time high — over-defensive
        # for a high-conviction profile. With tier coupling: GREEN
        # multiplies the pause threshold by 1.5 (let it drawdown more
        # before pausing); YELLOW uses the configured value; ORANGE+
        # already short-circuited above (no entries at all).
        orders = self._apply_source_pause(orders, account, tier)

        if tier == RiskTier.YELLOW:
            # YELLOW: scale entries by 0.5, then STILL enforce the hard
            # exposure caps. Without the second pass, 0.5 × N orders
            # could collectively still breach max_total_wallet_exposure
            # (each individual order halved, but cumulative projection
            # over N orders is unbounded). The exposure-limit pass also
            # accumulates same-tick projection (see _enforce_exposure_limits).
            orders = self._limit_new_entries(orders, account, scale=0.5)

        return self._enforce_exposure_limits(orders, account)

    def _apply_source_pause(
        self, orders: list[Order], account: AccountState,
        tier: RiskTier = RiskTier.GREEN,
    ) -> list[Order]:
        """Drop new entries for any source whose bucket is in deep drawdown.

        Reduce-only orders always pass — the breaker is about preventing
        *new* exposure, not stranding existing positions. Trips when the
        bucket's drawdown (from its own peak, normalized to balance)
        exceeds the configured threshold, scaled by tier-aware relax.
        """
        # Tier-aware relax: when the account isn't showing distress
        # (GREEN), let a bucket draw down further before pausing — the
        # cross-bucket loss isolation still kicks in via Kelly / vol-
        # target / correlation sizing, so the source pause is the
        # last line of defense, not the first.
        relax = 1.5 if tier == RiskTier.GREEN else 1.0
        trend_dd = account.source_drawdown_pct(OrderSource.TREND)
        grid_dd = account.source_drawdown_pct(OrderSource.GRID)
        trend_paused = trend_dd >= self.config.pause_trend_dd_pct * relax
        grid_paused = grid_dd >= self.config.pause_grid_dd_pct * relax
        if not (trend_paused or grid_paused):
            return orders

        result: list[Order] = []
        for o in orders:
            if o.reduce_only:
                result.append(o)
                continue
            if o.source == OrderSource.TREND and trend_paused:
                continue
            if o.source != OrderSource.TREND and grid_paused:
                # GRID and RISK entries both target the grid bucket, so
                # the grid breaker drops both.
                continue
            result.append(o)
        return result

    def _panic_close_all(self, account: AccountState) -> list[Order]:
        """Emit one reduce-only market order per open bucket.

        The fill simulator routes by ``Order.source``, so emitting GRID for
        the grid bucket and TREND for the trend bucket keeps panic closes
        source-isolated. We lose the "RISK" tag in fills, but P&L
        attribution stays correct (each bucket settles into its own
        running total).
        """
        orders: list[Order] = []
        for symbol, ss in account.symbols.items():
            for side, source, pos in (
                (Side.LONG, OrderSource.GRID, ss.position_long),
                (Side.LONG, OrderSource.TREND, ss.trend_long),
                (Side.SHORT, OrderSource.GRID, ss.position_short),
                (Side.SHORT, OrderSource.TREND, ss.trend_short),
            ):
                if pos.is_open:
                    orders.append(Order(
                        symbol=symbol,
                        side=side,
                        price=ss.last_price,
                        qty=abs(pos.size),
                        source=source,
                        reduce_only=True,
                        is_market=True,
                    ))
        return orders

    def compute_unstuck_orders(
        self,
        account: AccountState,
        grid_wallet_exposure_limit: float,
        now_ms: int,
        exchange_params: dict[str, ExchangeParams] | None = None,
    ) -> list[Order]:
        """Controlled exposure shedding.

        Modeled on passivbot's ``calc_unstucking_action``: emit small
        reduce-only limit orders when a bucket's wallet exposure crosses
        ``unstuck_threshold * WEL`` and current price has reverted through
        the appropriate EMA band. Like passivbot Rust, all candidates
        compete and only the most stuck position emits in a tick.

        Parameters
        ----------
        grid_wallet_exposure_limit
            The GridConfig.wallet_exposure_limit — passed in because the
            risk layer doesn't own GridConfig. The trend bucket uses
            ``self.config.trend_wallet_exposure_limit`` instead.
        exchange_params
            Per-symbol exchange constraints. When provided, the
            allowance-scaled close qty is quantized to qty_step and
            validated against min_qty / min_cost — without this the
            unstuck path can emit sub-step / sub-min orders that the
            live executor would silently reject, creating
            backtest/live divergence.
        """
        if self.config.unstuck_threshold < 0 or self.config.unstuck_close_pct <= 0:
            return []
        balance = account.balance
        if balance <= 0:
            return []

        allowance_budget = self.config.daily_loss_allowance_pct * balance
        # add_realized_pnl records losses as negative numbers, so
        # loss_24h returns ≤ 0. Compare |loss_24h| against the budget.
        grid_loss_spent = -account.loss_24h(OrderSource.GRID, now_ms)
        trend_loss_spent = -account.loss_24h(OrderSource.TREND, now_ms)

        candidates: list[tuple[float, float, Order]] = []
        for symbol, ss in account.symbols.items():
            ep = exchange_params.get(symbol) if exchange_params else None
            candidate = self._unstuck_for_bucket(
                symbol, ss, Side.LONG, OrderSource.GRID,
                ss.position_long, grid_wallet_exposure_limit,
                balance, allowance_budget - grid_loss_spent, ep,
            )
            if candidate is not None:
                candidates.append(candidate)
            candidate = self._unstuck_for_bucket(
                symbol, ss, Side.SHORT, OrderSource.GRID,
                ss.position_short, grid_wallet_exposure_limit,
                balance, allowance_budget - grid_loss_spent, ep,
            )
            if candidate is not None:
                candidates.append(candidate)
            candidate = self._unstuck_for_bucket(
                symbol, ss, Side.LONG, OrderSource.TREND,
                ss.trend_long, self.config.trend_wallet_exposure_limit,
                balance, allowance_budget - trend_loss_spent, ep,
            )
            if candidate is not None:
                candidates.append(candidate)
            candidate = self._unstuck_for_bucket(
                symbol, ss, Side.SHORT, OrderSource.TREND,
                ss.trend_short, self.config.trend_wallet_exposure_limit,
                balance, allowance_budget - trend_loss_spent, ep,
            )
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            return []
        # Lower profit_pct means more underwater; higher exposure ratio
        # breaks ties. This mirrors passivbot's single "most stuck" pick
        # without reimplementing its integer pprice-diff helpers.
        candidates.sort(key=lambda item: (item[0], item[1]))
        return [candidates[0][2]]

    def _unstuck_for_bucket(
        self,
        symbol: str,
        ss: SymbolState,
        side: Side,
        source: OrderSource,
        pos,  # combo_bot.types.Position — avoid the import cycle
        wallet_exposure_limit: float,
        balance: float,
        allowance_remaining: float,
        ep: ExchangeParams | None = None,
    ) -> tuple[float, float, Order] | None:
        if not pos.is_open or wallet_exposure_limit <= 0:
            return None
        if allowance_remaining <= 0:
            # Out of loss budget — don't emit any new unstuck orders;
            # the existing close ladder still runs.
            return None

        # we = notional / balance, with c_mult for contract-agnostic exposure.
        cm = ep.c_mult if ep else ss.c_mult
        notional = abs(pos.size) * pos.entry_price * cm
        we = notional / max(balance, 1e-12)
        if we / wallet_exposure_limit <= self.config.unstuck_threshold:
            return None

        ema = ss.ema
        if not ema.initialized or ema.upper <= 0 or ema.lower <= 0:
            return None

        # passivbot Rust trigger bands:
        #   long  → ema_band_upper * (1 + dist)
        #   short → ema_band_lower * (1 - dist)
        # Once triggered, passivbot returns the close at current price.
        last_price = ss.last_price
        if last_price <= 0:
            return None
        if side == Side.LONG:
            trigger_price = ema.upper * (1.0 + self.config.unstuck_ema_dist)
            if trigger_price <= 0 or last_price < trigger_price:
                return None
            order_price = last_price
            profit_pct = (last_price - pos.entry_price) / pos.entry_price
        else:
            trigger_price = ema.lower * (1.0 - self.config.unstuck_ema_dist)
            if trigger_price <= 0 or last_price > trigger_price:
                return None
            order_price = last_price
            profit_pct = (pos.entry_price - last_price) / pos.entry_price

        close_qty = (
            balance
            * wallet_exposure_limit
            * self.config.unstuck_close_pct
            / max(last_price * cm, 1e-12)
        )
        close_qty = min(abs(pos.size), close_qty)
        if close_qty <= 0:
            return None

        # Project the realized P&L this unstuck would book if it filled
        # at order_price. If that's a LOSS, cap close_qty so the loss
        # doesn't exceed ``allowance_remaining`` (the 24h budget left
        # over for this bucket). The pre-fix gate only checked whether
        # allowance > 0 — once you crossed the threshold, qty was a
        # fixed % regardless of how big the projected loss was vs the
        # remaining budget. That made the 24h budget a step function
        # instead of a continuous brake.
        if side == Side.LONG:
            projected_pnl_per_unit = order_price - pos.entry_price
        else:
            projected_pnl_per_unit = pos.entry_price - order_price
        if projected_pnl_per_unit < 0:
            loss_per_unit = -projected_pnl_per_unit * cm
            max_qty_by_allowance = allowance_remaining / max(loss_per_unit, 1e-12)
            if max_qty_by_allowance < close_qty:
                if max_qty_by_allowance <= 0:
                    return None
                close_qty = max_qty_by_allowance

        # Quantize to qty_step and validate the exchange constraints
        # AFTER allowance scaling — otherwise an allowance-shrunk close
        # qty can land below qty_step / min_qty / min_cost. Without
        # ep we have to fall through (used by some tests); the
        # downstream _simulate_fills / _quantize_order_for_send will
        # then act as the backstop.
        if ep is not None:
            if ep.qty_step > 0:
                close_qty = quantize_qty(close_qty, ep.qty_step)
            if close_qty < ep.min_qty:
                return None
            cost = close_qty * order_price * ep.c_mult
            if cost < ep.min_cost:
                return None

        if close_qty <= 0:
            return None

        order = Order(
            symbol=symbol,
            side=side,
            price=order_price,
            qty=close_qty,
            source=source,
            reduce_only=True,
            # Limit, not market — the whole point is to let the market
            # come to us at a controlled price above the EMA band.
            is_market=False,
        )
        return (profit_pct, -(we / wallet_exposure_limit), order)

    def _limit_new_entries(
        self, orders: list[Order], account: AccountState, scale: float
    ) -> list[Order]:
        """Scale new entries by ``scale``; preserve every other field.

        Previously this rebuilt Order with an explicit field list, which
        silently dropped ``is_market`` (and any future field added to
        Order) — most importantly, trend-overlay entries (is_market=True)
        passing through YELLOW risk became is_market=False mid-pipeline
        and the live executor then sent them as limits instead of
        crossing the book.
        """
        filtered = []
        for o in orders:
            if o.reduce_only:
                filtered.append(o)
            else:
                filtered.append(replace(o, qty=o.qty * scale))
        return filtered

    def _enforce_exposure_limits(
        self, orders: list[Order], account: AccountState
    ) -> list[Order]:
        """Drop entries that would breach the WE caps.

        IMPORTANT: cumulative projection. Each accepted order's cost is
        added to a running ``twe_added`` / per-(symbol, side)
        ``we_added`` so that the second, third, ... entry through the
        loop sees the exposure it would have IF the previously-accepted
        orders fill. Without this, every order is checked against the
        current snapshot only and N entries each below the cap can
        collectively breach it N-fold.
        """
        filtered = []
        denom = max(account.balance, 1e-12)
        base_twe = (
            account.total_wallet_exposure(Side.LONG)
            + account.total_wallet_exposure(Side.SHORT)
        )
        # Per-(symbol, side) base WE cache so we don't pay the sum
        # over open buckets every iteration.
        base_we_per_key: dict[tuple[str, Side], float] = {}
        twe_added = 0.0
        we_added: dict[tuple[str, Side], float] = {}
        for o in orders:
            if o.reduce_only:
                filtered.append(o)
                continue

            ss = account.symbols.get(o.symbol)
            cm = ss.c_mult if ss else 1.0
            cost = o.qty * o.price * cm
            cost_we = cost / denom

            if base_twe + twe_added + cost_we > self.config.max_total_wallet_exposure:
                continue

            if ss:
                # Single-symbol exposure must sum grid + trend buckets —
                # otherwise the trend overlay sneaks past the per-symbol
                # cap by living in a separate bucket.
                key = (o.symbol, o.side)
                if key not in base_we_per_key:
                    if o.side == Side.LONG:
                        buckets = (ss.position_long, ss.trend_long)
                    else:
                        buckets = (ss.position_short, ss.trend_short)
                    base_we_per_key[key] = sum(
                        abs(p.size) * p.entry_price * cm / denom
                        for p in buckets if p.is_open
                    )
                base_we = base_we_per_key[key]
                already_added = we_added.get(key, 0.0)
                if base_we + already_added + cost_we > self.config.max_single_exposure:
                    continue
                we_added[key] = already_added + cost_we

            twe_added += cost_we
            filtered.append(o)
        return filtered

    def check_liquidation(self, account: AccountState) -> bool:
        if account.equity_peak <= 0:
            return False
        return account.equity <= account.equity_peak * self.config.liquidation_threshold
