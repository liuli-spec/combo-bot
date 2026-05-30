"""Tests for Stage 3 SourcedPosition source isolation.

Verifies that grid-engine fills, trend-overlay fills, and risk-driven
panic closes route to the correct bucket on :class:`SymbolState`, and
that ``AccountState`` correctly sums exposure / equity across all
buckets.
"""

from __future__ import annotations

import pytest

from combo_bot.backtest import Backtester, BacktestConfig
from combo_bot.risk import RiskConfig, RiskManager
from combo_bot.types import (
    AccountState,
    Candle,
    ExchangeParams,
    Order,
    OrderSource,
    Position,
    Side,
    SymbolState,
)

# ---------------------------------------------------------------------------
# AccountState aggregation across buckets
# ---------------------------------------------------------------------------


class TestAccountAggregation:
    def test_total_wallet_exposure_sums_grid_and_trend_buckets(self):
        acc = AccountState(balance=10000, equity=10000, equity_peak=10000)
        ss = SymbolState("BTC", last_price=50000)
        ss.position_long = Position(0.1, 50000)  # 50% WE
        ss.trend_long = Position(0.05, 50000)  # 25% WE
        acc.symbols["BTC"] = ss

        we = acc.total_wallet_exposure(Side.LONG)
        # 5000 + 2500 = 7500 notional / 10000 balance = 0.75
        assert we == pytest.approx(0.75)

    def test_update_equity_includes_trend_unrealized_pnl(self):
        acc = AccountState(balance=10000, equity=10000, equity_peak=10000)
        ss = SymbolState("BTC", last_price=55000)  # +10% from entry
        ss.position_long = Position(0.1, 50000)
        ss.trend_long = Position(0.05, 50000)
        acc.symbols["BTC"] = ss

        acc.update_equity()
        # Grid upnl: 0.1 * (55000-50000) = 500
        # Trend upnl: 0.05 * (55000-50000) = 250
        assert acc.equity == pytest.approx(10000 + 500 + 250)


# ---------------------------------------------------------------------------
# Fill routing by Order.source
# ---------------------------------------------------------------------------


class TestFillRouting:
    @pytest.fixture
    def setup(self):
        backtester = Backtester(BacktestConfig(starting_balance=10000))
        account = AccountState(balance=10000, equity=10000, equity_peak=10000)
        account.symbols["BTC"] = SymbolState("BTC", last_price=50000)
        ep = {"BTC": ExchangeParams()}
        candle = Candle(
            timestamp=1,
            open=50000,
            high=50500,
            low=49500,
            close=50000,
            volume=1.0,
        )
        return backtester, account, ep, candle

    def test_grid_entry_grows_grid_bucket(self, setup):
        bt, acc, ep, c = setup
        order = Order("BTC", Side.LONG, 49500, 0.1, OrderSource.GRID)
        bt._simulate_fills([order], {"BTC": c}, acc, ep, 1)

        ss = acc.symbols["BTC"]
        assert ss.position_long.is_open
        assert ss.position_long.size == pytest.approx(0.1)
        assert not ss.trend_long.is_open

    def test_trend_entry_grows_trend_bucket(self, setup):
        bt, acc, ep, c = setup
        order = Order("BTC", Side.LONG, 50000, 0.05, OrderSource.TREND)
        bt._simulate_fills([order], {"BTC": c}, acc, ep, 1)

        ss = acc.symbols["BTC"]
        assert ss.trend_long.is_open
        assert ss.trend_long.size == pytest.approx(0.05)
        assert not ss.position_long.is_open

    def test_grid_close_reduces_grid_bucket_only(self, setup):
        bt, acc, ep, c = setup
        ss = acc.symbols["BTC"]
        ss.position_long = Position(0.1, 50000)
        ss.trend_long = Position(0.05, 50000)

        close = Order("BTC", Side.LONG, 50500, 0.1, OrderSource.GRID, reduce_only=True)
        bt._simulate_fills([close], {"BTC": c}, acc, ep, 1)

        assert not ss.position_long.is_open
        assert ss.trend_long.is_open
        assert ss.trend_long.size == pytest.approx(0.05)

    def test_trend_close_reduces_trend_bucket_only(self, setup):
        bt, acc, ep, c = setup
        ss = acc.symbols["BTC"]
        ss.position_long = Position(0.1, 50000)
        ss.trend_long = Position(0.05, 50000)

        close = Order(
            "BTC", Side.LONG, 50500, 0.05, OrderSource.TREND, reduce_only=True
        )
        bt._simulate_fills([close], {"BTC": c}, acc, ep, 1)

        assert ss.position_long.is_open
        assert ss.position_long.size == pytest.approx(0.1)
        assert not ss.trend_long.is_open

    def test_risk_close_targets_grid_bucket(self, setup):
        """OrderSource.RISK (strategy custom_exit) routes to grid bucket."""
        bt, acc, ep, c = setup
        ss = acc.symbols["BTC"]
        ss.position_long = Position(0.1, 50000)
        ss.trend_long = Position(0.05, 50000)

        close = Order(
            "BTC",
            Side.LONG,
            50500,
            0.1,
            OrderSource.RISK,
            reduce_only=True,
            is_market=True,
        )
        bt._simulate_fills([close], {"BTC": c}, acc, ep, 1)

        # Grid bucket emptied, trend bucket untouched.
        assert not ss.position_long.is_open
        assert ss.trend_long.is_open
        assert ss.trend_long.size == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# PnL attribution
# ---------------------------------------------------------------------------


class TestPnLAttribution:
    @pytest.fixture
    def setup(self):
        backtester = Backtester(BacktestConfig(starting_balance=10000))
        account = AccountState(balance=10000, equity=10000, equity_peak=10000)
        account.symbols["BTC"] = SymbolState("BTC", last_price=50000)
        ss = account.symbols["BTC"]
        ss.position_long = Position(0.1, 50000)
        ss.trend_long = Position(0.05, 50000)
        ep = {"BTC": ExchangeParams()}
        candle = Candle(1, 50000, 51000, 49000, 50000, 1.0)
        return backtester, account, ep, candle

    def test_grid_fill_attributed_to_grid_pnl(self, setup):
        """Grid TP fill should produce a non-zero realized PnL with source=GRID."""
        bt, acc, ep, c = setup
        close = Order(
            "BTC",
            Side.LONG,
            51000,
            0.1,
            OrderSource.GRID,
            reduce_only=True,
        )
        fills = bt._simulate_fills([close], {"BTC": c}, acc, ep, 1)
        assert len(fills) == 1
        assert fills[0].source == OrderSource.GRID
        # 0.1 * (51000 - 50000) = 100
        assert fills[0].realized_pnl == pytest.approx(100.0)

    def test_trend_fill_attributed_to_trend_pnl(self, setup):
        bt, acc, ep, c = setup
        close = Order(
            "BTC",
            Side.LONG,
            51000,
            0.05,
            OrderSource.TREND,
            reduce_only=True,
        )
        fills = bt._simulate_fills([close], {"BTC": c}, acc, ep, 1)
        assert len(fills) == 1
        assert fills[0].source == OrderSource.TREND
        # 0.05 * (51000 - 50000) = 50
        assert fills[0].realized_pnl == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Risk panic close emits per-bucket orders
# ---------------------------------------------------------------------------


class TestPanicCloseIsolation:
    def test_panic_close_emits_one_order_per_open_bucket(self):
        risk = RiskManager(RiskConfig())
        acc = AccountState(balance=10000, equity=7000, equity_peak=10000)
        ss = SymbolState("BTC", last_price=50000)
        ss.position_long = Position(0.1, 50000)
        ss.position_short = Position(0.05, 50000)
        ss.trend_long = Position(0.02, 50000)
        ss.trend_short = Position(0.03, 50000)
        acc.symbols["BTC"] = ss

        orders = risk.filter_orders([], acc)

        # Four open buckets => four reduce-only market orders.
        assert len(orders) == 4
        assert all(o.reduce_only for o in orders)
        assert all(o.is_market for o in orders)

        by_source_side = {(o.source, o.side, o.qty) for o in orders}
        assert (OrderSource.GRID, Side.LONG, 0.1) in by_source_side
        assert (OrderSource.GRID, Side.SHORT, 0.05) in by_source_side
        assert (OrderSource.TREND, Side.LONG, 0.02) in by_source_side
        assert (OrderSource.TREND, Side.SHORT, 0.03) in by_source_side

    def test_panic_close_skips_empty_buckets(self):
        risk = RiskManager(RiskConfig())
        acc = AccountState(balance=10000, equity=7000, equity_peak=10000)
        ss = SymbolState("BTC", last_price=50000)
        # Only one bucket open — the trend long.
        ss.trend_long = Position(0.02, 50000)
        acc.symbols["BTC"] = ss

        orders = risk.filter_orders([], acc)
        assert len(orders) == 1
        assert orders[0].source == OrderSource.TREND
        assert orders[0].side == Side.LONG


# ---------------------------------------------------------------------------
# Exposure limit applies to combined buckets
# ---------------------------------------------------------------------------


class TestExposureLimits:
    def test_single_symbol_exposure_sums_grid_plus_trend(self):
        """A trend entry that fits its own bucket but exceeds the
        combined per-symbol exposure cap is now TRIMMED to the headroom
        (Round-23 Passivbot partial-fit semantic) instead of being
        dropped outright. The crucial invariant is unchanged: combined
        exposure across grid + trend buckets MUST NOT exceed the cap.
        """
        risk = RiskManager(
            RiskConfig(
                max_single_exposure=0.5,
                max_drawdown_pct=1.0,
                yellow_threshold=1.0,
                orange_threshold=1.0,
                red_threshold=1.0,
            )
        )
        acc = AccountState(balance=10000, equity=10000, equity_peak=10000)
        ss = SymbolState("BTC", last_price=50000)
        # Grid bucket already at 40% exposure.
        ss.position_long = Position(0.08, 50000)
        acc.symbols["BTC"] = ss

        # Trend entry would add another 15% and push combined to 55%.
        # Partial-fit trims it to 10% (fits the remaining headroom).
        orders = [Order("BTC", Side.LONG, 50000, 0.03, OrderSource.TREND)]
        filtered = risk.filter_orders(orders, acc)
        assert len(filtered) == 1, (
            f"partial-fit must trim the trend entry rather than drop it; "
            f"got {len(filtered)} orders"
        )
        # Trimmed qty ≈ 0.02 (cost 1000 → WE 0.1). Floating-point math
        # produces a value just below 0.02 — assert with tolerance.
        trimmed = filtered[0]
        assert trimmed.qty <= 0.02 + 1e-9, (
            f"trimmed qty must not exceed the 0.1-WE headroom (qty 0.02); "
            f"got {trimmed.qty}"
        )
        # Combined exposure stays AT or below the cap, never above.
        combined_we = (
            ss.position_long.size * 50000 / 10000 + trimmed.qty * trimmed.price / 10000
        )
        assert (
            combined_we <= 0.5 + 1e-9
        ), f"combined grid+trend WE must not exceed 0.5 cap; got {combined_we}"

    def test_small_trend_entry_passes_under_combined_cap(self):
        risk = RiskManager(
            RiskConfig(
                max_single_exposure=0.5,
                max_drawdown_pct=1.0,
                yellow_threshold=1.0,
                orange_threshold=1.0,
                red_threshold=1.0,
            )
        )
        acc = AccountState(balance=10000, equity=10000, equity_peak=10000)
        ss = SymbolState("BTC", last_price=50000)
        ss.position_long = Position(0.08, 50000)  # 40% WE
        acc.symbols["BTC"] = ss

        # Trend adds 5% — combined 45%, under cap.
        orders = [Order("BTC", Side.LONG, 50000, 0.01, OrderSource.TREND)]
        filtered = risk.filter_orders(orders, acc)
        assert len(filtered) == 1


# ---------------------------------------------------------------------------
# SymbolState.bucket() dispatch
# ---------------------------------------------------------------------------


class TestBucketDispatch:
    def test_grid_source_returns_grid_bucket(self):
        ss = SymbolState("BTC")
        assert ss.bucket(OrderSource.GRID, Side.LONG) is ss.position_long
        assert ss.bucket(OrderSource.GRID, Side.SHORT) is ss.position_short

    def test_trend_source_returns_trend_bucket(self):
        ss = SymbolState("BTC")
        assert ss.bucket(OrderSource.TREND, Side.LONG) is ss.trend_long
        assert ss.bucket(OrderSource.TREND, Side.SHORT) is ss.trend_short

    def test_risk_source_falls_back_to_grid(self):
        """RISK source historically targets the grid (formerly combined)
        bucket so strategy ``custom_exit`` keeps closing what it always
        closed."""
        ss = SymbolState("BTC")
        assert ss.bucket(OrderSource.RISK, Side.LONG) is ss.position_long
        assert ss.bucket(OrderSource.RISK, Side.SHORT) is ss.position_short
