from __future__ import annotations
import pytest
from combo_bot.types import AccountState, Order, OrderSource, Position, Side, SymbolState
from combo_bot.risk import RiskConfig, RiskManager, RiskTier


@pytest.fixture
def risk():
    return RiskManager(RiskConfig(
        yellow_threshold=0.10,
        orange_threshold=0.18,
        red_threshold=0.25,
        max_total_wallet_exposure=3.0,
        max_single_exposure=0.5,
    ))


class TestRiskTier:
    def test_green_on_no_drawdown(self, risk):
        acc = AccountState(balance=10000, equity=10000, equity_peak=10000)
        assert risk.assess(acc) == RiskTier.GREEN

    def test_yellow_on_moderate_drawdown(self, risk):
        acc = AccountState(balance=10000, equity=8800, equity_peak=10000)
        assert risk.assess(acc) == RiskTier.YELLOW

    def test_orange_on_high_drawdown(self, risk):
        acc = AccountState(balance=10000, equity=8000, equity_peak=10000)
        assert risk.assess(acc) == RiskTier.ORANGE

    def test_red_on_extreme_drawdown(self, risk):
        acc = AccountState(balance=10000, equity=7000, equity_peak=10000)
        assert risk.assess(acc) == RiskTier.RED


class TestOrderFiltering:
    def test_green_passes_all_orders(self, risk):
        acc = AccountState(balance=10000, equity=10000, equity_peak=10000)
        acc.symbols["BTC"] = SymbolState("BTC", last_price=50000)
        orders = [
            Order("BTC", Side.LONG, 49000, 0.01, OrderSource.GRID),
            Order("BTC", Side.LONG, 51000, 0.01, OrderSource.GRID, reduce_only=True),
        ]
        filtered = risk.filter_orders(orders, acc)
        assert len(filtered) == 2

    def test_orange_only_closes(self, risk):
        acc = AccountState(balance=10000, equity=8000, equity_peak=10000)
        acc.symbols["BTC"] = SymbolState("BTC", last_price=50000)
        orders = [
            Order("BTC", Side.LONG, 49000, 0.01, OrderSource.GRID),
            Order("BTC", Side.LONG, 51000, 0.01, OrderSource.GRID, reduce_only=True),
        ]
        filtered = risk.filter_orders(orders, acc)
        assert all(o.reduce_only for o in filtered)

    def test_red_generates_panic_close(self, risk):
        acc = AccountState(balance=10000, equity=7000, equity_peak=10000)
        acc.symbols["BTC"] = SymbolState("BTC", last_price=50000)
        acc.symbols["BTC"].position_long = Position(0.1, 50000)
        orders = [
            Order("BTC", Side.LONG, 49000, 0.01, OrderSource.GRID),
        ]
        filtered = risk.filter_orders(orders, acc)
        assert all(o.reduce_only for o in filtered)
        assert any(o.source == OrderSource.RISK for o in filtered)

    def test_yellow_scales_entries(self, risk):
        acc = AccountState(balance=10000, equity=8800, equity_peak=10000)
        acc.symbols["BTC"] = SymbolState("BTC", last_price=50000)
        orders = [
            Order("BTC", Side.LONG, 49000, 0.02, OrderSource.GRID),
        ]
        filtered = risk.filter_orders(orders, acc)
        assert len(filtered) == 1
        assert filtered[0].qty == pytest.approx(0.01, abs=0.001)

    def test_exposure_limit_blocks_oversized_orders(self, risk):
        acc = AccountState(balance=10000, equity=10000, equity_peak=10000)
        acc.symbols["BTC"] = SymbolState("BTC", last_price=50000)
        acc.symbols["BTC"].position_long = Position(0.5, 50000)
        orders = [
            Order("BTC", Side.LONG, 49000, 0.2, OrderSource.GRID),
        ]
        filtered = risk.filter_orders(orders, acc)
        entries = [o for o in filtered if not o.reduce_only]
        assert len(entries) == 0


class TestLiquidation:
    def test_not_liquidated_normally(self, risk):
        acc = AccountState(balance=10000, equity=10000, equity_peak=10000)
        assert not risk.check_liquidation(acc)

    def test_liquidated_at_threshold(self, risk):
        acc = AccountState(balance=500, equity=400, equity_peak=10000)
        assert risk.check_liquidation(acc)
