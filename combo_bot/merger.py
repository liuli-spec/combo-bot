from __future__ import annotations
from dataclasses import dataclass, replace
from combo_bot.types import (
    AccountState,
    Order,
    OrderSource,
    Position,
    Side,
    TradingMode,
    TrendRegime,
    TrendSignal,
    ExchangeParams,
)


@dataclass
class MergerConfig:
    trend_override_strong: bool = True
    grid_depth_limit_in_downtrend: int = 2
    # Trend overlay sizing — re-tuned for high-conviction directional
    # trades. Old defaults (0.15 × 0.03 = 0.45% per entry) were so
    # conservative the overlay barely moved the needle even in
    # STRONG_BULL/BEAR. New defaults (0.35 × 0.10 = 3.5% per entry)
    # let the overlay actually express conviction; the upstream
    # KellySizer / VolTargetSizer / RegimeArbiter conviction scale
    # still throttle the final qty.
    #
    # NOTE: ``trend_position_max_pct`` is the LEGACY field and is only
    # consulted when DecisionMerger is constructed without
    # ``trend_wallet_exposure_limit``. Production wiring (Backtester,
    # LiveTrader) passes ``risk.trend_wallet_exposure_limit`` directly
    # so overlay sizing and risk unstuck stay in lockstep.
    trend_position_max_pct: float = 0.35
    trend_entry_qty_pct: float = 0.10
    mode_switch_strong_threshold: float = 0.6
    mode_switch_weak_threshold: float = 0.3
    trend_stop_loss_pct: float = 0.03
    trend_take_profit_pct: float = 0.06
    funding_rate_pause_threshold: float = 0.001


class DecisionMerger:
    def __init__(
        self,
        config: MergerConfig | None = None,
        *,
        trend_wallet_exposure_limit: float | None = None,
    ):
        self.config = config or MergerConfig()
        # Round-22 unification: when the caller passes the canonical
        # ``risk.trend_wallet_exposure_limit`` it becomes the single
        # source of truth for overlay sizing. Without it we fall back
        # to the legacy ``MergerConfig.trend_position_max_pct`` for
        # back-compat with older test fixtures.
        self._trend_wel_override: float | None = trend_wallet_exposure_limit

    @property
    def effective_trend_wel(self) -> float:
        if self._trend_wel_override is not None:
            return self._trend_wel_override
        return self.config.trend_position_max_pct

    def compute_mode(
        self, signal: TrendSignal, side: Side, position: Position
    ) -> TradingMode:
        """DEPRECATED: superseded by :class:`RegimeArbiter.compute`.
        Retained for test compatibility only; production paths use the
        arbiter directly."""
        import warnings

        warnings.warn(
            "DecisionMerger.compute_mode is deprecated; use RegimeArbiter.compute",
            DeprecationWarning,
            stacklevel=2,
        )
        if side == Side.LONG:
            if signal.regime == TrendRegime.STRONG_BEAR:
                return (
                    TradingMode.GRACEFUL_STOP
                    if position.is_open
                    else TradingMode.TP_ONLY
                )
            if signal.regime == TrendRegime.BEAR:
                return TradingMode.TP_ONLY
            return TradingMode.NORMAL

        if signal.regime == TrendRegime.STRONG_BULL:
            return (
                TradingMode.GRACEFUL_STOP if position.is_open else TradingMode.TP_ONLY
            )
        if signal.regime == TrendRegime.BULL:
            return TradingMode.TP_ONLY
        return TradingMode.NORMAL

    def filter_grid_orders(
        self, orders: list[Order], signal: TrendSignal, side: Side
    ) -> list[Order]:
        is_adverse = (
            side == Side.LONG
            and signal.direction < -self.config.mode_switch_weak_threshold
        ) or (
            side == Side.SHORT
            and signal.direction > self.config.mode_switch_weak_threshold
        )
        if not is_adverse:
            return orders

        filtered = []
        entry_count = 0
        for o in orders:
            if o.reduce_only:
                filtered.append(o)
                continue
            entry_count += 1
            if entry_count <= self.config.grid_depth_limit_in_downtrend:
                filtered.append(o)
        return filtered

    def generate_trend_orders(
        self,
        symbol: str,
        signal: TrendSignal,
        price: float,
        account: AccountState,
        exchange: ExchangeParams,
    ) -> list[Order]:
        if signal.strength < self.config.mode_switch_strong_threshold:
            return []

        max_cost = account.balance * self.effective_trend_wel
        qty = (
            max_cost
            * self.config.trend_entry_qty_pct
            / max(price * exchange.c_mult, 1e-12)
        )
        qty = max(qty, exchange.min_qty)
        cost = qty * price * exchange.c_mult
        if cost > max_cost or cost < exchange.min_cost:
            return []

        orders = []

        if signal.regime == TrendRegime.STRONG_BULL:
            ss = account.symbols.get(symbol)
            # After Stage 3 source isolation, the overlay co-exists with
            # the grid bucket — we only skip when the *trend* bucket is
            # already populated to avoid pyramiding the overlay.
            if ss and ss.trend_long.is_open:
                return []
            orders.append(
                Order(
                    symbol=symbol,
                    side=Side.LONG,
                    price=price,
                    qty=qty,
                    source=OrderSource.TREND,
                )
            )

        elif signal.regime == TrendRegime.STRONG_BEAR:
            ss = account.symbols.get(symbol)
            if ss and ss.trend_short.is_open:
                return []
            orders.append(
                Order(
                    symbol=symbol,
                    side=Side.SHORT,
                    price=price,
                    qty=qty,
                    source=OrderSource.TREND,
                )
            )

        return orders

    def merge_orders(
        self,
        grid_orders: list[Order],
        trend_orders: list[Order],
        signal: TrendSignal,
    ) -> list[Order]:
        """DEPRECATED: production paths call filter_grid_orders and
        generate_trend_orders separately.  Kept as a convenience wrapper
        for interactive use."""
        import warnings

        warnings.warn(
            "DecisionMerger.merge_orders is unused in production; use filter_grid_orders + generate_trend_orders separately",
            DeprecationWarning,
            stacklevel=2,
        )
        closes = [o for o in grid_orders if o.reduce_only]
        entries = [o for o in grid_orders if not o.reduce_only]
        entries = self.filter_grid_orders(
            entries, signal, entries[0].side if entries else Side.LONG
        )

        # In a strong adverse trend, accelerate closes by scaling qty up by 50%.
        # Return a new list with replaced orders rather than mutating in place.
        if signal.strength > self.config.mode_switch_strong_threshold:
            accelerated: list[Order] = []
            for o in closes:
                adverse_for_long = o.side == Side.LONG and signal.regime in (
                    TrendRegime.STRONG_BEAR,
                    TrendRegime.BEAR,
                )
                adverse_for_short = o.side == Side.SHORT and signal.regime in (
                    TrendRegime.STRONG_BULL,
                    TrendRegime.BULL,
                )
                if adverse_for_long or adverse_for_short:
                    accelerated.append(replace(o, qty=o.qty * 1.5))
                else:
                    accelerated.append(o)
            closes = accelerated

        return closes + entries + trend_orders

    def generate_trend_exit_orders(
        self,
        symbol: str,
        position: Position,
        side: Side,
        price: float,
        exchange: ExchangeParams,
    ) -> list[Order]:
        if not position.is_open:
            return []

        orders = []
        if side == Side.LONG:
            sl_price = position.entry_price * (1.0 - self.config.trend_stop_loss_pct)
            tp_price = position.entry_price * (1.0 + self.config.trend_take_profit_pct)
            if price <= sl_price or price >= tp_price:
                orders.append(
                    Order(
                        symbol=symbol,
                        side=Side.LONG,
                        price=price,
                        qty=abs(position.size),
                        source=OrderSource.TREND,
                        reduce_only=True,
                        is_market=True,  # SL/TP exits cross the book immediately
                    )
                )
        else:
            sl_price = position.entry_price * (1.0 + self.config.trend_stop_loss_pct)
            tp_price = position.entry_price * (1.0 - self.config.trend_take_profit_pct)
            if price >= sl_price or price <= tp_price:
                orders.append(
                    Order(
                        symbol=symbol,
                        side=Side.SHORT,
                        price=price,
                        qty=abs(position.size),
                        source=OrderSource.TREND,
                        reduce_only=True,
                        is_market=True,
                    )
                )

        return orders
