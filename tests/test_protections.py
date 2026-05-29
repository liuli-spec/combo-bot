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


# ---------------------------------------------------------------------------
# Supplementary coverage: edge cases not exercised above.
# ---------------------------------------------------------------------------


class TestProtectionLockMatching:
    def test_global_lock_matches_anything(self):
        lock = ProtectionLock(until_ms=1_000, reason="x")
        assert lock.matches("BTC", Side.LONG, OrderSource.GRID)
        assert lock.matches("ETH", Side.SHORT, OrderSource.TREND)

    def test_partial_scope_filters_only_specified_dims(self):
        # side scoped, symbol/source open
        lock = ProtectionLock(until_ms=1_000, reason="x", side=Side.SHORT)
        assert lock.matches("BTC", Side.SHORT, OrderSource.GRID)
        assert not lock.matches("BTC", Side.LONG, OrderSource.GRID)


class TestStoplossGuardExtra:
    def test_winning_fills_do_not_count(self):
        guard = StoplossGuard(StoplossGuardConfig(trade_limit=2))
        fills = [_fill(timestamp=0, pnl=-5)] + [
            _fill(timestamp=i, pnl=+100) for i in range(1, 11)
        ]
        locks = guard.evaluate(fills, AccountState(), now_ms=10_000)
        assert locks == []

    def test_only_per_pair_isolates_symbols(self):
        guard = StoplossGuard(StoplossGuardConfig(
            trade_limit=2, only_per_pair=True,
        ))
        fills = [
            _fill(timestamp=0, pnl=-5, symbol="BTC"),
            _fill(timestamp=0, pnl=-5, symbol="BTC"),
            _fill(timestamp=0, pnl=-5, symbol="ETH"),
        ]
        locks = guard.evaluate(fills, AccountState(), now_ms=10_000)
        assert len(locks) == 1
        assert locks[0].symbol == "BTC"

    def test_only_per_side_isolates_long_short(self):
        guard = StoplossGuard(StoplossGuardConfig(
            trade_limit=2, only_per_side=True,
        ))
        fills = [
            _fill(timestamp=0, pnl=-5, side=Side.LONG),
            _fill(timestamp=0, pnl=-5, side=Side.LONG),
            _fill(timestamp=0, pnl=-5, side=Side.SHORT),
        ]
        locks = guard.evaluate(fills, AccountState(), now_ms=10_000)
        assert len(locks) == 1
        assert locks[0].side == Side.LONG

    def test_only_per_source_isolates_buckets(self):
        guard = StoplossGuard(StoplossGuardConfig(
            trade_limit=2, only_per_source=True,
        ))
        fills = [
            _fill(timestamp=0, pnl=-5, source=OrderSource.TREND),
            _fill(timestamp=0, pnl=-5, source=OrderSource.TREND),
            _fill(timestamp=0, pnl=-5, source=OrderSource.GRID),
        ]
        locks = guard.evaluate(fills, AccountState(), now_ms=10_000)
        assert len(locks) == 1
        assert locks[0].source == OrderSource.TREND


class TestManagerLockLifecycle:
    def test_protection_emits_lock_via_update(self):
        """Driving the manager through fills should produce a lock from
        the StoplossGuard, not just from manually-injected ones."""
        mgr = ProtectionManager([
            StoplossGuard(StoplossGuardConfig(
                trade_limit=2, stop_duration_ms=30_000,
            )),
        ])
        mgr.update(
            [_fill(timestamp=1_000, pnl=-5), _fill(timestamp=1_000, pnl=-5)],
            AccountState(), now_ms=1_000,
        )
        assert len(mgr.locks) == 1
        # That global lock should block a new entry but pass a reduce_only.
        entry = Order("BTC", Side.LONG, 49_000, 0.01, OrderSource.GRID)
        close = Order("BTC", Side.LONG, 51_000, 0.01, OrderSource.GRID, reduce_only=True)
        out = mgr.filter_orders([entry, close], now_ms=5_000)
        assert close in out
        assert entry not in out

    def test_locks_expire_after_their_duration(self):
        mgr = ProtectionManager([])
        mgr.locks.append(ProtectionLock(until_ms=1_000, reason="x"))
        mgr.locks.append(ProtectionLock(until_ms=10_000, reason="y"))
        mgr.update([], AccountState(), now_ms=2_000)
        assert len(mgr.locks) == 1
        assert mgr.locks[0].reason == "y"


class TestCooldownPeriodExtra:
    def test_default_scope_is_per_symbol_side_source(self):
        cd = CooldownPeriod()
        locks = cd.evaluate(
            [_fill(timestamp=0, pnl=-1, symbol="BTC",
                   side=Side.LONG, source=OrderSource.GRID)],
            AccountState(), now_ms=0,
        )
        lock = locks[0]
        assert lock.symbol == "BTC"
        assert lock.side == Side.LONG
        assert lock.source == OrderSource.GRID

    def test_disabling_all_scope_flags_creates_global_lock(self):
        cd = CooldownPeriod(CooldownPeriodConfig(
            only_per_pair=False,
            only_per_side=False,
            only_per_source=False,
        ))
        locks = cd.evaluate([_fill(timestamp=0, pnl=-1)], AccountState(), now_ms=0)
        lock = locks[0]
        assert lock.symbol is None
        assert lock.side is None
        assert lock.source is None

