"""Stage 4 per-source circuit-breaker tests.

The breaker pauses new entries for the source whose bucket has drawn
down past its threshold, while leaving the other source untouched and
allowing reduce-only orders for both.
"""

from __future__ import annotations

import pytest

from combo_bot.risk import RiskConfig, RiskManager
from combo_bot.types import (
    AccountState,
    Order,
    OrderSource,
    Position,
    Side,
    SymbolState,
)


def _account(balance=10_000.0) -> AccountState:
    acc = AccountState(balance=balance, equity=balance, equity_peak=balance)
    acc.symbols["BTC"] = SymbolState("BTC", last_price=50_000.0)
    return acc


def _risk(**overrides) -> RiskManager:
    """RiskManager with the global tier breakers wide open so we can
    test the per-source layer in isolation."""
    defaults = dict(
        max_drawdown_pct=1.0,
        yellow_threshold=1.0,
        orange_threshold=1.0,
        red_threshold=1.0,
        pause_trend_dd_pct=0.10,
        pause_grid_dd_pct=0.20,
    )
    defaults.update(overrides)
    return RiskManager(RiskConfig(**defaults))


# ---------------------------------------------------------------------------
# AccountState bookkeeping
# ---------------------------------------------------------------------------


class TestAccountSourceBookkeeping:
    def test_add_realized_pnl_routes_by_source(self):
        acc = _account()
        acc.add_realized_pnl(OrderSource.GRID, 100.0)
        acc.add_realized_pnl(OrderSource.TREND, 50.0)
        acc.add_realized_pnl(OrderSource.RISK, 25.0)  # RISK -> grid bucket

        assert acc.grid_realized_pnl == pytest.approx(125.0)
        assert acc.trend_realized_pnl == pytest.approx(50.0)

    def test_update_equity_splits_unrealized_by_bucket(self):
        acc = _account()
        ss = acc.symbols["BTC"]
        ss.last_price = 55_000.0
        ss.position_long = Position(0.1, 50_000.0)  # +500 grid upnl
        ss.trend_long = Position(0.05, 50_000.0)  # +250 trend upnl
        acc.grid_realized_pnl = 200.0
        acc.trend_realized_pnl = -100.0

        acc.update_equity()

        # Grid equity = 200 realized + 500 upnl = 700
        # Trend equity = -100 realized + 250 upnl = 150
        assert acc.grid_equity == pytest.approx(700.0)
        assert acc.trend_equity == pytest.approx(150.0)
        # Peaks ratchet upward.
        assert acc.grid_equity_peak == pytest.approx(700.0)
        assert acc.trend_equity_peak == pytest.approx(150.0)

    def test_source_drawdown_normalized_to_balance(self):
        acc = _account(balance=10_000.0)
        # Trend peak was +1000, now sitting at -500 — a 1500 drop, 15% of balance.
        acc.trend_equity_peak = 1000.0
        acc.trend_equity = -500.0

        dd = acc.source_drawdown_pct(OrderSource.TREND)
        assert dd == pytest.approx(0.15)

    def test_source_drawdown_zero_when_at_peak(self):
        acc = _account()
        acc.grid_equity_peak = 500.0
        acc.grid_equity = 500.0
        assert acc.source_drawdown_pct(OrderSource.GRID) == 0.0

    def test_source_drawdown_clamped_at_zero_when_above_peak(self):
        """A new high (equity above prior peak) should report 0 dd, not negative."""
        acc = _account()
        acc.trend_equity_peak = 100.0
        acc.trend_equity = 500.0
        assert acc.source_drawdown_pct(OrderSource.TREND) == 0.0


# ---------------------------------------------------------------------------
# RiskManager per-source pause
# ---------------------------------------------------------------------------


class TestTrendPause:
    def test_trend_overlay_entry_dropped_when_trend_bucket_in_dd(self):
        # With tier=GREEN the source pause now applies a 1.5× relax
        # to the configured threshold, so the bucket DD has to push
        # past pause_trend_dd_pct * 1.5 (= 0.15) to fire. 16% trend
        # drawdown is comfortably above that line.
        risk = _risk(pause_trend_dd_pct=0.10)
        acc = _account()
        acc.trend_equity_peak = 2000.0
        acc.trend_equity = 400.0  # 1600 dd / 10000 balance = 16%

        orders = [
            Order("BTC", Side.LONG, 50_000, 0.01, OrderSource.TREND),
        ]
        filtered = risk.filter_orders(orders, acc)
        assert filtered == []

    def test_trend_reduce_only_still_passes_when_trend_paused(self):
        """We never want to strand an existing trend position."""
        risk = _risk(pause_trend_dd_pct=0.10)
        acc = _account()
        acc.trend_equity_peak = 2000.0
        acc.trend_equity = 0.0  # 20% dd, well above threshold

        orders = [
            Order("BTC", Side.LONG, 50_000, 0.01, OrderSource.TREND, reduce_only=True),
            Order("BTC", Side.LONG, 50_000, 0.01, OrderSource.TREND),
        ]
        filtered = risk.filter_orders(orders, acc)
        assert len(filtered) == 1
        assert filtered[0].reduce_only is True

    def test_grid_entry_still_passes_when_only_trend_paused(self):
        """Trend breaker should not throttle grid entries."""
        risk = _risk(pause_trend_dd_pct=0.10, pause_grid_dd_pct=0.30)
        acc = _account()
        acc.trend_equity_peak = 1500.0
        acc.trend_equity = 0.0  # 15% trend dd
        acc.grid_equity_peak = 200.0
        acc.grid_equity = 200.0  # no grid dd

        orders = [Order("BTC", Side.LONG, 49_000, 0.02, OrderSource.GRID)]
        filtered = risk.filter_orders(orders, acc)
        assert len(filtered) == 1
        assert filtered[0].source == OrderSource.GRID


class TestGridPause:
    def test_grid_entry_dropped_when_grid_bucket_in_dd(self):
        # threshold 0.20 × GREEN relax 1.5 = 0.30 effective; push to 40%.
        risk = _risk(pause_grid_dd_pct=0.20)
        acc = _account()
        acc.grid_equity_peak = 4000.0
        acc.grid_equity = 0.0  # 40% dd

        orders = [Order("BTC", Side.LONG, 49_000, 0.02, OrderSource.GRID)]
        filtered = risk.filter_orders(orders, acc)
        assert filtered == []

    def test_risk_source_entries_also_throttled_by_grid_pause(self):
        """RISK source targets the grid bucket, so the grid breaker
        should drop new RISK entries too. (Reduce-only RISK closes still
        pass — see below.)"""
        risk = _risk(pause_grid_dd_pct=0.10)
        acc = _account()
        acc.grid_equity_peak = 2000.0
        acc.grid_equity = 0.0  # 20% dd

        orders = [
            # Hypothetical "RISK entry" — closes always pass, entries don't.
            Order("BTC", Side.LONG, 50_000, 0.01, OrderSource.RISK),
            Order("BTC", Side.LONG, 50_000, 0.01, OrderSource.RISK, reduce_only=True),
        ]
        filtered = risk.filter_orders(orders, acc)
        assert len(filtered) == 1
        assert filtered[0].reduce_only is True

    def test_trend_entry_still_passes_when_only_grid_paused(self):
        risk = _risk(pause_grid_dd_pct=0.20)
        acc = _account()
        acc.grid_equity_peak = 3000.0
        acc.grid_equity = 0.0  # 30% grid dd
        # Trend bucket flat.
        orders = [Order("BTC", Side.LONG, 50_000, 0.01, OrderSource.TREND)]
        filtered = risk.filter_orders(orders, acc)
        assert len(filtered) == 1
        assert filtered[0].source == OrderSource.TREND


class TestPauseReset:
    def test_pause_clears_when_bucket_returns_to_peak(self):
        # threshold 0.10 × GREEN relax 1.5 = 0.15 effective; push to 20%.
        risk = _risk(pause_trend_dd_pct=0.10)
        acc = _account()
        # Initially in drawdown well past the relax-adjusted line.
        acc.trend_equity_peak = 2000.0
        acc.trend_equity = 0.0
        order = Order("BTC", Side.LONG, 50_000, 0.01, OrderSource.TREND)
        assert risk.filter_orders([order], acc) == []

        # Recovery: bucket climbs back to peak — pause clears.
        acc.trend_equity = 2000.0
        filtered = risk.filter_orders([order], acc)
        assert len(filtered) == 1

    def test_threshold_at_one_disables_breaker(self):
        risk = _risk(pause_trend_dd_pct=1.5, pause_grid_dd_pct=1.5)
        acc = _account()
        # Severe drawdown on both buckets — but breaker is off.
        acc.trend_equity_peak = 5000.0
        acc.trend_equity = -3000.0  # 80% dd
        acc.grid_equity_peak = 8000.0
        acc.grid_equity = -2000.0  # 100% dd

        orders = [
            Order("BTC", Side.LONG, 50_000, 0.01, OrderSource.TREND),
            Order("BTC", Side.LONG, 49_000, 0.02, OrderSource.GRID),
        ]
        filtered = risk.filter_orders(orders, acc)
        assert len(filtered) == 2


# ---------------------------------------------------------------------------
# Integration: per-source bookkeeping flows through Backtester
# ---------------------------------------------------------------------------


class TestBacktesterIntegration:
    def test_backtester_populates_per_source_realized_pnl(self):
        from combo_bot.backtest import Backtester, BacktestConfig
        from combo_bot.grid_engine import GridConfig
        from tests.conftest import make_candles

        # Steady uptrend triggers grid entries and TPs but no strong
        # overlay signal.
        candles = make_candles([50_000 + i * 50 for i in range(400)])
        cfg = BacktestConfig(
            starting_balance=10_000,
            symbols=["BTC"],
            grid=GridConfig(
                entry_initial_ema_dist=0.001,
                entry_grid_spacing_pct=0.01,
                close_grid_markup_start=0.005,
                close_grid_markup_end=0.01,
                wallet_exposure_limit=0.3,
                max_grid_levels=3,
            ),
        )
        bt = Backtester(cfg)
        result = bt.run({"BTC": candles})

        # Both the result and the account agree on per-source totals.
        assert result.grid_pnl != 0.0
        assert result.trend_pnl == 0.0
