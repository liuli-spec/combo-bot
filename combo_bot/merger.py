from __future__ import annotations
from dataclasses import dataclass
from combo_bot.types import (
    AccountState, Order, OrderSource, Position, Side,
    TradingMode, TrendRegime, TrendSignal, ExchangeParams,
)


@dataclass
class MergerConfig:
    trend_override_strong: bool = True
    grid_depth_limit_in_downtrend: int = 2
    trend_position_max_pct: float = 0.15
    trend_entry_qty_pct: float = 0.03
    mode_switch_strong_threshold: float = 0.6
    mode_switch_weak_threshold: float = 0.3
    trend_stop_loss_pct: float = 0.03
    trend_take_profit_pct: float = 0.06
    funding_rate_pause_threshold: float = 0.001


class DecisionMerger:
    def __init__(self, config: MergerConfig | None = None):
        self.config = config or MergerConfig()

    def compute_mode(
        self, signal: TrendSignal, side: Side, position: Position
    ) -> TradingMode:
        if side == Side.LONG:
            if signal.regime == TrendRegime.STRONG_BEAR:
                return TradingMode.GRACEFUL_STOP if position.is_open else TradingMode.TP_ONLY
            if signal.regime == TrendRegime.BEAR:
                return TradingMode.TP_ONLY
            return TradingMode.NORMAL

        if signal.regime == TrendRegime.STRONG_BULL:
            return TradingMode.GRACEFUL_STOP if position.is_open else TradingMode.TP_ONLY
        if signal.regime == TrendRegime.BULL:
            return TradingMode.TP_ONLY
        return TradingMode.NORMAL

    def filter_grid_orders(
        self, orders: list[Order], signal: TrendSignal, side: Side
    ) -> list[Order]:
        is_adverse = (
            (side == Side.LONG and signal.direction < -self.config.mode_switch_weak_threshold)
            or (side == Side.SHORT and signal.direction > self.config.mode_switch_weak_threshold)
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

        max_cost = account.balance * self.config.trend_position_max_pct
        qty = max_cost * self.config.trend_entry_qty_pct / max(price * exchange.c_mult, 1e-12)
        qty = max(qty, exchange.min_qty)
        cost = qty * price * exchange.c_mult
        if cost > max_cost or cost < exchange.min_cost:
            return []

        orders = []

        if signal.regime == TrendRegime.STRONG_BULL:
            ss = account.symbols.get(symbol)
            if ss and ss.position_long.is_open:
                return []
            orders.append(Order(
                symbol=symbol,
                side=Side.LONG,
                price=price,
                qty=qty,
                source=OrderSource.TREND,
            ))

        elif signal.regime == TrendRegime.STRONG_BEAR:
            ss = account.symbols.get(symbol)
            if ss and ss.position_short.is_open:
                return []
            orders.append(Order(
                symbol=symbol,
                side=Side.SHORT,
                price=price,
                qty=qty,
                source=OrderSource.TREND,
            ))

        return orders

    def merge_orders(
        self,
        grid_orders: list[Order],
        trend_orders: list[Order],
        signal: TrendSignal,
    ) -> list[Order]:
        closes = [o for o in grid_orders if o.reduce_only]
        entries = [o for o in grid_orders if not o.reduce_only]
        entries = self.filter_grid_orders(entries, signal, entries[0].side if entries else Side.LONG)

        if signal.strength > self.config.mode_switch_strong_threshold:
            for o in closes:
                if (
                    (o.side == Side.LONG and signal.regime in (TrendRegime.STRONG_BEAR, TrendRegime.BEAR))
                    or (o.side == Side.SHORT and signal.regime in (TrendRegime.STRONG_BULL, TrendRegime.BULL))
                ):
                    o.qty = min(o.qty * 1.5, abs(o.qty) * 2)

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
                orders.append(Order(
                    symbol=symbol,
                    side=Side.LONG,
                    price=price,
                    qty=abs(position.size),
                    source=OrderSource.TREND,
                    reduce_only=True,
                ))
        else:
            sl_price = position.entry_price * (1.0 + self.config.trend_stop_loss_pct)
            tp_price = position.entry_price * (1.0 - self.config.trend_take_profit_pct)
            if price >= sl_price or price <= tp_price:
                orders.append(Order(
                    symbol=symbol,
                    side=Side.SHORT,
                    price=price,
                    qty=abs(position.size),
                    source=OrderSource.TREND,
                    reduce_only=True,
                ))

        return orders
