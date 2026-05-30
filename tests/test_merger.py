from __future__ import annotations
import pytest
from combo_bot.types import (
    Order,
    OrderSource,
    Position,
    Side,
    TradingMode,
    TrendRegime,
    TrendSignal,
)
from combo_bot.merger import MergerConfig, DecisionMerger


@pytest.fixture
def merger():
    return DecisionMerger(
        MergerConfig(
            grid_depth_limit_in_downtrend=2,
            trend_position_max_pct=0.15,
            mode_switch_strong_threshold=0.6,
            mode_switch_weak_threshold=0.3,
        )
    )


class TestModeComputation:
    def test_long_normal_in_neutral(self, merger):
        signal = TrendSignal(direction=0.0, strength=0.0, regime=TrendRegime.NEUTRAL)
        mode = merger.compute_mode(signal, Side.LONG, Position())
        assert mode == TradingMode.NORMAL

    def test_long_tp_only_in_bear(self, merger):
        signal = TrendSignal(direction=-0.4, strength=0.4, regime=TrendRegime.BEAR)
        mode = merger.compute_mode(signal, Side.LONG, Position())
        assert mode == TradingMode.TP_ONLY

    def test_long_graceful_stop_in_strong_bear_with_position(self, merger):
        signal = TrendSignal(
            direction=-0.8, strength=0.8, regime=TrendRegime.STRONG_BEAR
        )
        pos = Position(size=0.1, entry_price=50000.0)
        mode = merger.compute_mode(signal, Side.LONG, pos)
        assert mode == TradingMode.GRACEFUL_STOP

    def test_short_tp_only_in_bull(self, merger):
        signal = TrendSignal(direction=0.4, strength=0.4, regime=TrendRegime.BULL)
        mode = merger.compute_mode(signal, Side.SHORT, Position())
        assert mode == TradingMode.TP_ONLY

    def test_short_normal_in_bear(self, merger):
        signal = TrendSignal(direction=-0.5, strength=0.5, regime=TrendRegime.BEAR)
        mode = merger.compute_mode(signal, Side.SHORT, Position())
        assert mode == TradingMode.NORMAL


class TestGridFiltering:
    def test_no_filtering_in_neutral(self, merger):
        signal = TrendSignal(direction=0.0, strength=0.0, regime=TrendRegime.NEUTRAL)
        orders = [
            Order("BTC", Side.LONG, 49000.0, 0.01, OrderSource.GRID),
            Order("BTC", Side.LONG, 48500.0, 0.015, OrderSource.GRID),
            Order("BTC", Side.LONG, 48000.0, 0.02, OrderSource.GRID),
        ]
        filtered = merger.filter_grid_orders(orders, signal, Side.LONG)
        assert len(filtered) == 3

    def test_limits_depth_in_downtrend_for_long(self, merger):
        signal = TrendSignal(direction=-0.5, strength=0.5, regime=TrendRegime.BEAR)
        orders = [
            Order("BTC", Side.LONG, 49000.0, 0.01, OrderSource.GRID),
            Order("BTC", Side.LONG, 48500.0, 0.015, OrderSource.GRID),
            Order("BTC", Side.LONG, 48000.0, 0.02, OrderSource.GRID),
            Order("BTC", Side.LONG, 47500.0, 0.025, OrderSource.GRID),
        ]
        filtered = merger.filter_grid_orders(orders, signal, Side.LONG)
        entries = [o for o in filtered if not o.reduce_only]
        assert len(entries) == 2

    def test_preserves_close_orders_during_filtering(self, merger):
        signal = TrendSignal(direction=-0.5, strength=0.5, regime=TrendRegime.BEAR)
        orders = [
            Order("BTC", Side.LONG, 51000.0, 0.05, OrderSource.GRID, reduce_only=True),
            Order("BTC", Side.LONG, 49000.0, 0.01, OrderSource.GRID),
            Order("BTC", Side.LONG, 48000.0, 0.02, OrderSource.GRID),
            Order("BTC", Side.LONG, 47000.0, 0.03, OrderSource.GRID),
        ]
        filtered = merger.filter_grid_orders(orders, signal, Side.LONG)
        closes = [o for o in filtered if o.reduce_only]
        assert len(closes) == 1


class TestTrendOrders:
    def test_generates_long_on_strong_bull(
        self, merger, account_state, exchange_params
    ):
        signal = TrendSignal(
            direction=0.8, strength=0.8, regime=TrendRegime.STRONG_BULL
        )
        orders = merger.generate_trend_orders(
            "BTC/USDT:USDT",
            signal,
            50000.0,
            account_state,
            exchange_params,
        )
        assert len(orders) == 1
        assert orders[0].side == Side.LONG
        assert orders[0].source == OrderSource.TREND

    def test_generates_short_on_strong_bear(
        self, merger, account_state, exchange_params
    ):
        signal = TrendSignal(
            direction=-0.8, strength=0.8, regime=TrendRegime.STRONG_BEAR
        )
        orders = merger.generate_trend_orders(
            "BTC/USDT:USDT",
            signal,
            50000.0,
            account_state,
            exchange_params,
        )
        assert len(orders) == 1
        assert orders[0].side == Side.SHORT

    def test_no_trend_orders_on_neutral(self, merger, account_state, exchange_params):
        signal = TrendSignal(direction=0.1, strength=0.1, regime=TrendRegime.NEUTRAL)
        orders = merger.generate_trend_orders(
            "BTC/USDT:USDT",
            signal,
            50000.0,
            account_state,
            exchange_params,
        )
        assert len(orders) == 0

    def test_no_duplicate_long_if_trend_bucket_already_open(
        self, merger, account_state, exchange_params
    ):
        # Stage 3: overlay only skips when its own trend bucket is populated.
        # A grid bucket on the same side is fine — overlay can co-exist.
        account_state.symbols["BTC/USDT:USDT"].trend_long = Position(0.01, 50000.0)
        signal = TrendSignal(
            direction=0.8, strength=0.8, regime=TrendRegime.STRONG_BULL
        )
        orders = merger.generate_trend_orders(
            "BTC/USDT:USDT",
            signal,
            50000.0,
            account_state,
            exchange_params,
        )
        assert len(orders) == 0

    def test_overlay_emitted_even_with_grid_position_open(
        self, merger, account_state, exchange_params
    ):
        """Stage 3: a grid long doesn't block the trend overlay long."""
        account_state.symbols["BTC/USDT:USDT"].position_long = Position(0.01, 50000.0)
        signal = TrendSignal(
            direction=0.8, strength=0.8, regime=TrendRegime.STRONG_BULL
        )
        orders = merger.generate_trend_orders(
            "BTC/USDT:USDT",
            signal,
            50000.0,
            account_state,
            exchange_params,
        )
        assert len(orders) == 1
        assert orders[0].source == OrderSource.TREND
