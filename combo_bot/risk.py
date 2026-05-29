from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from combo_bot.types import AccountState, Order, OrderSource, Side, ExchangeParams


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


class RiskManager:
    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()
        self.tier = RiskTier.GREEN
        self.red_cooldown_until: int = 0

    def assess(self, account: AccountState) -> RiskTier:
        dd = account.drawdown

        if dd >= self.config.red_threshold:
            self.tier = RiskTier.RED
        elif dd >= self.config.orange_threshold:
            self.tier = RiskTier.ORANGE
        elif dd >= self.config.yellow_threshold:
            self.tier = RiskTier.YELLOW
        else:
            self.tier = RiskTier.GREEN

        return self.tier

    def filter_orders(
        self, orders: list[Order], account: AccountState, timestamp: int = 0
    ) -> list[Order]:
        tier = self.assess(account)

        if tier == RiskTier.RED:
            self.red_cooldown_until = timestamp + self.config.cooldown_after_red_minutes * 60_000
            return self._panic_close_all(account)

        if timestamp < self.red_cooldown_until:
            return [o for o in orders if o.reduce_only]

        if tier == RiskTier.ORANGE:
            return [o for o in orders if o.reduce_only]

        if tier == RiskTier.YELLOW:
            return self._limit_new_entries(orders, account, scale=0.5)

        return self._enforce_exposure_limits(orders, account)

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
