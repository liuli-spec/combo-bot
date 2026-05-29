from __future__ import annotations

from combo_bot.backtest import BacktestConfig, Backtester
from combo_bot.grid_engine import GridConfig
from combo_bot.protections import (
    CooldownPeriod,
    CooldownPeriodConfig,
    IProtection,
    ProtectionLock,
    ProtectionManager,
    StoplossGuard,
    StoplossGuardConfig,
)
from combo_bot.types import AccountState, Fill, Order, OrderSource, Side
from tests.conftest import make_candles


def _fill(
    *,
    timestamp: int,
    pnl: float,
    symbol: str = "BTC",
    side: Side = Side.LONG,
    source: OrderSource = OrderSource.GRID,
) -> Fill:
    return Fill(
        timestamp=timestamp,
        symbol=symbol,
        side=side,
        price=50_000.0,
        qty=0.01,
        fee=0.0,
        realized_pnl=pnl,
        source=source,
    )


class TestProtectionManager:
    def test_filter_orders_drops_locked_entries_but_keeps_exits(self):
        manager = ProtectionManager()
        manager.locks.append(ProtectionLock(
            until_ms=10_000,
            reason="test",
            symbol="BTC",
            side=Side.LONG,
            source=OrderSource.GRID,
        ))
        entry = Order("BTC", Side.LONG, 49_000, 0.01, OrderSource.GRID)
        exit_order = Order(
            "BTC", Side.LONG, 51_000, 0.01, OrderSource.GRID,
            reduce_only=True,
        )
        other_source = Order("BTC", Side.LONG, 49_000, 0.01, OrderSource.TREND)

        filtered = manager.filter_orders([entry, exit_order, other_source], 1_000)

        assert entry not in filtered
        assert exit_order in filtered
        assert other_source in filtered

    def test_expired_locks_do_not_match(self):
        manager = ProtectionManager()
        manager.locks.append(ProtectionLock(until_ms=1_000, reason="old"))
        assert manager.is_locked("BTC", Side.LONG, OrderSource.GRID, 2_000) is False


class TestStoplossGuard:
    def test_global_lock_after_loss_cluster(self):
        guard = StoplossGuard(StoplossGuardConfig(
            lookback_period_ms=60_000,
            trade_limit=2,
            stop_duration_ms=30_000,
        ))
        locks = guard.evaluate(
            [_fill(timestamp=1_000, pnl=-10), _fill(timestamp=2_000, pnl=-5)],
            AccountState(balance=10_000),
            now_ms=2_000,
        )

        assert len(locks) == 1
        assert locks[0].symbol is None
        assert locks[0].side is None
        assert locks[0].source is None

    def test_scoped_lock_by_pair_side_and_source(self):
        guard = StoplossGuard(StoplossGuardConfig(
            lookback_period_ms=60_000,
            trade_limit=1,
            only_per_pair=True,
            only_per_side=True,
            only_per_source=True,
        ))
        locks = guard.evaluate(
            [_fill(timestamp=1_000, pnl=-10, source=OrderSource.TREND)],
            AccountState(balance=10_000),
            now_ms=1_000,
        )

        assert locks[0].symbol == "BTC"
        assert locks[0].side == Side.LONG
        assert locks[0].source == OrderSource.TREND

    def test_old_losses_are_pruned(self):
        guard = StoplossGuard(StoplossGuardConfig(
            lookback_period_ms=1_000,
            trade_limit=2,
        ))
        guard.evaluate([_fill(timestamp=0, pnl=-10)], AccountState(), now_ms=0)
        locks = guard.evaluate(
            [_fill(timestamp=2_000, pnl=-10)],
            AccountState(),
            now_ms=2_000,
        )

        assert locks == []


class TestCooldownPeriod:
    def test_loss_creates_scoped_lock(self):
        cooldown = CooldownPeriod(CooldownPeriodConfig(stop_duration_ms=5_000))
        locks = cooldown.evaluate(
            [_fill(timestamp=1_000, pnl=-1, source=OrderSource.TREND)],
            AccountState(balance=10_000),
            now_ms=1_000,
        )

        assert len(locks) == 1
        assert locks[0].until_ms == 6_000
        assert locks[0].source == OrderSource.TREND

    def test_open_fill_does_not_create_lock(self):
        cooldown = CooldownPeriod()
        locks = cooldown.evaluate(
            [_fill(timestamp=1_000, pnl=0)],
            AccountState(balance=10_000),
            now_ms=1_000,
        )
        assert locks == []


class _LockImmediately(IProtection):
    def evaluate(
        self,
        fills: list[Fill],
        account: AccountState,
        now_ms: int,
    ) -> list[ProtectionLock]:
        return []


class TestBacktesterProtectionIntegration:
    def test_active_lock_suppresses_backtest_entries(self):
        candles = make_candles([50_000 - i * 10 for i in range(160)])
        bt = Backtester(
            BacktestConfig(
                starting_balance=10_000,
                symbols=["BTC"],
                grid=GridConfig(max_grid_levels=2, entry_initial_ema_dist=0.001),
            ),
            protections=[_LockImmediately()],
        )
        bt.protections.locks.append(ProtectionLock(
            until_ms=10**18,
            reason="integration_test",
            symbol="BTC",
            side=Side.LONG,
            source=OrderSource.GRID,
        ))

        result = bt.run({"BTC": candles})

        grid_long_entries = [
            f for f in result.fills
            if f.source == OrderSource.GRID
            and f.side == Side.LONG
            and f.realized_pnl == 0
        ]
        assert grid_long_entries == []

