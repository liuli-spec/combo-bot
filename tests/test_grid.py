from __future__ import annotations
import pytest
from combo_bot.types import (
    OrderSource,
    Position,
    Side,
    TradingMode,
)
from combo_bot.grid_engine import (
    GridConfig,
    GridEngine,
    quantize_qty,
    quantize_price,
    calc_wallet_exposure,
)


class TestQuantize:
    def test_quantize_qty_rounds_down(self):
        assert quantize_qty(1.2345, 0.01) == 1.23

    def test_quantize_qty_exact(self):
        assert quantize_qty(1.0, 0.001) == 1.0

    def test_quantize_qty_zero_step(self):
        assert quantize_qty(1.5, 0) == 1.5

    def test_quantize_price_rounds_down(self):
        assert quantize_price(50123.456, 0.01) == pytest.approx(50123.45, abs=1e-8)

    def test_quantize_price_rounds_up(self):
        assert quantize_price(50123.451, 0.01, round_up=True) == 50123.46

    def test_calc_wallet_exposure(self):
        we = calc_wallet_exposure(10000.0, 0.01, 50000.0, 1.0)
        assert abs(we - 0.05) < 1e-10

    def test_calc_wallet_exposure_zero_balance(self):
        assert calc_wallet_exposure(0.0, 1.0, 50000.0, 1.0) == 0.0


class TestGridEntries:
    @pytest.fixture
    def engine(self):
        return GridEngine(
            GridConfig(
                entry_initial_ema_dist=0.005,
                entry_initial_qty_pct=0.02,
                entry_grid_spacing_pct=0.02,
                entry_grid_double_down_factor=1.5,
                wallet_exposure_limit=1.0,
                max_grid_levels=5,
            )
        )

    def test_generates_entries_when_no_position(
        self, engine, ema_state, volatility_state, exchange_params
    ):
        orders = engine.compute_orders(
            "BTC/USDT:USDT",
            Side.LONG,
            Position(),
            ema_state,
            volatility_state,
            10000.0,
            0.0,
            exchange_params,
            TradingMode.NORMAL,
        )
        entries = [o for o in orders if not o.reduce_only]
        assert len(entries) > 0
        assert all(o.source == OrderSource.GRID for o in entries)

    def test_entry_prices_decrease_for_long(
        self, engine, ema_state, volatility_state, exchange_params
    ):
        orders = engine.compute_orders(
            "BTC/USDT:USDT",
            Side.LONG,
            Position(),
            ema_state,
            volatility_state,
            10000.0,
            0.0,
            exchange_params,
            TradingMode.NORMAL,
        )
        entries = [o for o in orders if not o.reduce_only]
        prices = [o.price for o in entries]
        for i in range(1, len(prices)):
            assert prices[i] < prices[i - 1], "Long entry prices must decrease"

    def test_entry_prices_increase_for_short(
        self, engine, ema_state, volatility_state, exchange_params
    ):
        orders = engine.compute_orders(
            "BTC/USDT:USDT",
            Side.SHORT,
            Position(),
            ema_state,
            volatility_state,
            10000.0,
            0.0,
            exchange_params,
            TradingMode.NORMAL,
        )
        entries = [o for o in orders if not o.reduce_only]
        prices = [o.price for o in entries]
        for i in range(1, len(prices)):
            assert prices[i] > prices[i - 1], "Short entry prices must increase"

    def test_qty_increases_with_double_down(
        self, engine, ema_state, volatility_state, exchange_params
    ):
        orders = engine.compute_orders(
            "BTC/USDT:USDT",
            Side.LONG,
            Position(),
            ema_state,
            volatility_state,
            10000.0,
            0.0,
            exchange_params,
            TradingMode.NORMAL,
        )
        entries = [o for o in orders if not o.reduce_only]
        if len(entries) >= 2:
            assert entries[1].qty >= entries[0].qty

    def test_respects_wallet_exposure_limit(
        self, engine, ema_state, volatility_state, exchange_params
    ):
        orders = engine.compute_orders(
            "BTC/USDT:USDT",
            Side.LONG,
            Position(),
            ema_state,
            volatility_state,
            10000.0,
            0.0,
            exchange_params,
            TradingMode.NORMAL,
        )
        entries = [o for o in orders if not o.reduce_only]
        total_cost = sum(o.qty * o.price for o in entries)
        assert total_cost / 10000.0 <= engine._cfg.wallet_exposure_limit * 1.05

    def test_no_entries_in_tp_only_mode(
        self, engine, ema_state, volatility_state, exchange_params
    ):
        orders = engine.compute_orders(
            "BTC/USDT:USDT",
            Side.LONG,
            Position(),
            ema_state,
            volatility_state,
            10000.0,
            0.0,
            exchange_params,
            TradingMode.TP_ONLY,
        )
        entries = [o for o in orders if not o.reduce_only]
        assert len(entries) == 0


class TestGridCloses:
    @pytest.fixture
    def engine(self):
        return GridEngine(
            GridConfig(
                close_grid_markup_start=0.005,
                close_grid_markup_end=0.015,
                close_grid_qty_pct=0.5,
            )
        )

    def test_generates_closes_for_open_long(
        self, engine, ema_state, volatility_state, exchange_params
    ):
        pos = Position(size=0.1, entry_price=50000.0)
        orders = engine.compute_orders(
            "BTC/USDT:USDT",
            Side.LONG,
            pos,
            ema_state,
            volatility_state,
            10000.0,
            0.5,
            exchange_params,
            TradingMode.NORMAL,
        )
        closes = [o for o in orders if o.reduce_only]
        assert len(closes) > 0
        assert all(o.side == Side.LONG for o in closes)

    def test_generates_closes_with_position_side_for_open_short(
        self, engine, ema_state, volatility_state, exchange_params
    ):
        pos = Position(size=0.1, entry_price=50000.0)
        orders = engine.compute_orders(
            "BTC/USDT:USDT",
            Side.SHORT,
            pos,
            ema_state,
            volatility_state,
            10000.0,
            0.5,
            exchange_params,
            TradingMode.NORMAL,
        )
        closes = [o for o in orders if o.reduce_only]
        assert len(closes) > 0
        assert all(o.side == Side.SHORT for o in closes)

    def test_close_prices_above_entry_for_long(
        self, engine, ema_state, volatility_state, exchange_params
    ):
        pos = Position(size=0.1, entry_price=50000.0)
        orders = engine.compute_orders(
            "BTC/USDT:USDT",
            Side.LONG,
            pos,
            ema_state,
            volatility_state,
            10000.0,
            0.5,
            exchange_params,
            TradingMode.NORMAL,
        )
        closes = [o for o in orders if o.reduce_only]
        for o in closes:
            assert o.price > pos.entry_price, "Long close price must be above entry"

    def test_close_prices_below_entry_for_short(
        self, engine, ema_state, volatility_state, exchange_params
    ):
        pos = Position(size=0.1, entry_price=50000.0)
        orders = engine.compute_orders(
            "BTC/USDT:USDT",
            Side.SHORT,
            pos,
            ema_state,
            volatility_state,
            10000.0,
            0.5,
            exchange_params,
            TradingMode.NORMAL,
        )
        closes = [o for o in orders if o.reduce_only]
        for o in closes:
            assert o.price < pos.entry_price, "Short close price must be below entry"

    def test_close_qty_sums_to_position_size(
        self, engine, ema_state, volatility_state, exchange_params
    ):
        pos = Position(size=0.1, entry_price=50000.0)
        orders = engine.compute_orders(
            "BTC/USDT:USDT",
            Side.LONG,
            pos,
            ema_state,
            volatility_state,
            10000.0,
            0.5,
            exchange_params,
            TradingMode.NORMAL,
        )
        closes = [o for o in orders if o.reduce_only]
        total_close = sum(o.qty for o in closes)
        assert abs(total_close - pos.size) < exchange_params.qty_step * 2

    def test_panic_mode_closes_full_position(
        self, engine, ema_state, volatility_state, exchange_params
    ):
        pos = Position(size=0.1, entry_price=50000.0)
        orders = engine.compute_orders(
            "BTC/USDT:USDT",
            Side.LONG,
            pos,
            ema_state,
            volatility_state,
            10000.0,
            0.5,
            exchange_params,
            TradingMode.PANIC,
        )
        assert len(orders) == 1
        assert orders[0].reduce_only
        assert orders[0].qty == pos.size
        assert orders[0].side == Side.LONG
