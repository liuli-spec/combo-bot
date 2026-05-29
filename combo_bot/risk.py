from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from combo_bot.types import (
    AccountState, EMAState, Order, OrderSource, Side,
    SymbolState, ExchangeParams,
)


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
        self.tier = RiskTier.GREEN
        self.red_cooldown_until: int = 0
        # Stage 6 HardStop state. ``dd_ema`` holds the time-decayed
        # drawdown EMA; ``_dd_initialized`` lets the very first
        # ``assess`` call short-circuit to raw drawdown so single-shot
        # unit tests still produce a deterministic tier without needing
        # to advance through multiple ticks. ``red_latched`` mirrors
        # passivbot's latch — once RED, the tier returns RED until
        # ``reset_red_latch`` is called.
        self.dd_ema: float = 0.0
        self.last_assess_minute: int = 0
        self._dd_initialized: bool = False
        self.red_latched: bool = False

    def reset_red_latch(self) -> None:
        """Clear the RED latch and any active cooldown.

        Intended for explicit operator action (or for tests). Re-entry
        into RED requires the drawdown threshold to be crossed again.
        Cooldown is reset too — an explicit reset means "fully clear",
        otherwise the lingering cooldown would still block entries.
        """
        self.red_latched = False
        self.red_cooldown_until = 0
        self.tier = RiskTier.GREEN

    def assess(
        self, account: AccountState, timestamp_ms: int = 0
    ) -> RiskTier:
        raw = account.drawdown
        score = self._update_dd_score(raw, timestamp_ms)

        if score >= self.config.red_threshold:
            base_tier = RiskTier.RED
        elif score >= self.config.orange_threshold:
            base_tier = RiskTier.ORANGE
        elif score >= self.config.yellow_threshold:
            base_tier = RiskTier.YELLOW
        else:
            base_tier = RiskTier.GREEN

        if base_tier == RiskTier.RED and self.config.red_latch_enabled:
            self.red_latched = True

        # Once latched, the tier is pinned at RED until reset, regardless
        # of how far drawdown recovers.
        self.tier = RiskTier.RED if self.red_latched else base_tier
        return self.tier

    def _update_dd_score(self, raw: float, timestamp_ms: int) -> float:
        """Return the drawdown score used for tier classification.

        ``score = min(raw, ema)`` per passivbot's hard-stop design.
        Disabled (returns raw) when ``dd_ema_span_minutes <= 0`` or on
        the very first call (so single-shot tests still classify
        deterministically without needing a tick history).
        """
        if self.config.dd_ema_span_minutes <= 0:
            return raw

        current_minute = timestamp_ms // 60_000
        if not self._dd_initialized:
            # Seed the EMA at the current raw drawdown so the next
            # call's smoothing starts from a sane anchor, and use raw
            # for this call.
            self.dd_ema = raw
            self.last_assess_minute = current_minute
            self._dd_initialized = True
            return raw

        elapsed_minutes = max(0, current_minute - self.last_assess_minute)
        if elapsed_minutes > 0:
            alpha = 2.0 / (self.config.dd_ema_span_minutes + 1.0)
            decay = (1.0 - alpha) ** elapsed_minutes
            # raw + (ema - raw) * decay  ==  alpha-blended EMA, accounting
            # for variable elapsed time between ticks.
            self.dd_ema = raw + (self.dd_ema - raw) * decay
            self.last_assess_minute = current_minute

        # min(raw, ema) — see RiskConfig docstring for the two failure
        # modes this guards against.
        return min(raw, self.dd_ema)

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

        # Per-source circuit breakers run regardless of tier so the trend
        # bucket can be paused for trend-specific drawdown even while the
        # overall account is still GREEN.
        orders = self._apply_source_pause(orders, account)

        if tier == RiskTier.YELLOW:
            return self._limit_new_entries(orders, account, scale=0.5)

        return self._enforce_exposure_limits(orders, account)

    def _apply_source_pause(
        self, orders: list[Order], account: AccountState
    ) -> list[Order]:
        """Drop new entries for any source whose bucket is in deep drawdown.

        Reduce-only orders always pass — the breaker is about preventing
        *new* exposure, not stranding existing positions. Trips when the
        bucket's drawdown (from its own peak, normalized to balance)
        exceeds the configured threshold.
        """
        trend_dd = account.source_drawdown_pct(OrderSource.TREND)
        grid_dd = account.source_drawdown_pct(OrderSource.GRID)
        trend_paused = trend_dd >= self.config.pause_trend_dd_pct
        grid_paused = grid_dd >= self.config.pause_grid_dd_pct
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
    ) -> list[Order]:
        """Per-bucket controlled exposure shedding.

        Modeled on passivbot's ``calc_unstucking_action``: emit small
        reduce-only limit orders priced outside the EMA band when a
        bucket's wallet exposure crosses ``unstuck_threshold * WEL``,
        and only when current price has rebounded across that band
        (sell the bounce, not the bleed). Throttled by a 24h rolling
        loss budget so a sustained drawdown can't realize unlimited
        losses.

        Parameters
        ----------
        grid_wallet_exposure_limit
            The GridConfig.wallet_exposure_limit — passed in because the
            risk layer doesn't own GridConfig. The trend bucket uses
            ``self.config.trend_wallet_exposure_limit`` instead.
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

        orders: list[Order] = []
        for symbol, ss in account.symbols.items():
            orders.extend(self._unstuck_for_bucket(
                symbol, ss, Side.LONG, OrderSource.GRID,
                ss.position_long, grid_wallet_exposure_limit,
                balance, allowance_budget - grid_loss_spent,
            ))
            orders.extend(self._unstuck_for_bucket(
                symbol, ss, Side.SHORT, OrderSource.GRID,
                ss.position_short, grid_wallet_exposure_limit,
                balance, allowance_budget - grid_loss_spent,
            ))
            orders.extend(self._unstuck_for_bucket(
                symbol, ss, Side.LONG, OrderSource.TREND,
                ss.trend_long, self.config.trend_wallet_exposure_limit,
                balance, allowance_budget - trend_loss_spent,
            ))
            orders.extend(self._unstuck_for_bucket(
                symbol, ss, Side.SHORT, OrderSource.TREND,
                ss.trend_short, self.config.trend_wallet_exposure_limit,
                balance, allowance_budget - trend_loss_spent,
            ))
        return orders

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
    ) -> list[Order]:
        if not pos.is_open or wallet_exposure_limit <= 0:
            return []
        if allowance_remaining <= 0:
            # Out of loss budget — don't emit any new unstuck orders;
            # the existing close ladder still runs.
            return []

        # we = notional / balance. Same formula as grid_engine.calc_wallet_exposure.
        notional = abs(pos.size) * pos.entry_price
        we = notional / max(balance, 1e-12)
        if we / wallet_exposure_limit <= self.config.unstuck_threshold:
            return []

        ema = ss.ema
        if not ema.initialized or ema.upper <= 0 or ema.lower <= 0:
            return []

        # Order price sits outside the EMA band by unstuck_ema_dist.
        # passivbot only triggers when current price has reverted across
        # that band — i.e. we sell into a bounce. Without this guard the
        # bot just bleeds into adverse moves.
        last_price = ss.last_price
        if last_price <= 0:
            return []
        if side == Side.LONG:
            order_price = ema.upper * (1.0 + self.config.unstuck_ema_dist)
            if last_price < order_price:
                return []
        else:
            order_price = ema.lower * (1.0 - self.config.unstuck_ema_dist)
            if last_price > order_price:
                return []

        close_qty = abs(pos.size) * self.config.unstuck_close_pct
        if close_qty <= 0:
            return []

        return [Order(
            symbol=symbol,
            side=side,
            price=order_price,
            qty=close_qty,
            source=source,
            reduce_only=True,
            # Limit, not market — the whole point is to let the market
            # come to us at a controlled price above the EMA band.
            is_market=False,
        )]

    def _limit_new_entries(
        self, orders: list[Order], account: AccountState, scale: float
    ) -> list[Order]:
        filtered = []
        for o in orders:
            if o.reduce_only:
                filtered.append(o)
            else:
                scaled = Order(
                    symbol=o.symbol,
                    side=o.side,
                    price=o.price,
                    qty=o.qty * scale,
                    source=o.source,
                    reduce_only=False,
                )
                filtered.append(scaled)
        return filtered

    def _enforce_exposure_limits(
        self, orders: list[Order], account: AccountState
    ) -> list[Order]:
        filtered = []
        for o in orders:
            if o.reduce_only:
                filtered.append(o)
                continue

            cost = o.qty * o.price
            current_twe = (
                account.total_wallet_exposure(Side.LONG)
                + account.total_wallet_exposure(Side.SHORT)
            )
            if current_twe + cost / max(account.balance, 1e-12) > self.config.max_total_wallet_exposure:
                continue

            ss = account.symbols.get(o.symbol)
            if ss:
                # Single-symbol exposure must sum grid + trend buckets —
                # otherwise the trend overlay sneaks past the per-symbol
                # cap by living in a separate bucket.
                if o.side == Side.LONG:
                    buckets = (ss.position_long, ss.trend_long)
                else:
                    buckets = (ss.position_short, ss.trend_short)
                denom = max(account.balance, 1e-12)
                current_we = sum(
                    abs(p.size) * p.entry_price / denom
                    for p in buckets if p.is_open
                )
                if current_we + cost / denom > self.config.max_single_exposure:
                    continue

            filtered.append(o)
        return filtered

    def check_liquidation(self, account: AccountState) -> bool:
        if account.equity_peak <= 0:
            return False
        return account.equity <= account.equity_peak * self.config.liquidation_threshold
