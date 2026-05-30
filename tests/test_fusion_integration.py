"""Integration tests for Stage 1 fusion changes.

These verify the new behaviors wired in during the first fusion stage:
  - is_market orders bypass limit-cross checks and fill at candle close
  - Position.best_price ratchets in the favorable direction each tick
  - StrategyRunner is invoked by Backtester and its callbacks affect the run
  - GridEngine PANIC mode produces a fillable is_market close

Each test focuses on a single behavior and uses the smallest viable scenario.
"""

from __future__ import annotations


import pytest

from combo_bot.backtest import BacktestConfig, Backtester
from combo_bot.grid_engine import GridConfig, GridEngine
from combo_bot.strategy import DefaultStrategy, ExampleTrendStrategy, TradeContext
from combo_bot.types import (
    AccountState,
    Candle,
    ExchangeParams,
    Order,
    OrderSource,
    Position,
    Side,
    SymbolState,
    TradingMode,
)
from tests.conftest import make_candles


class TestIsMarketFill:
    def test_market_close_fills_when_limit_would_not(self):
        # Long position; candle low/close are below the order's stated price.
        # A limit reduce-only at the stated price normally fills only when
        # candle.high crosses it. Here we put the price above the high so the
        # limit would NOT fill — but is_market=True should force the fill.
        account = AccountState(balance=10_000, equity=10_000, equity_peak=10_000)
        account.symbols["BTC"] = SymbolState("BTC", last_price=49_000)
        account.symbols["BTC"].position_long = Position(size=0.1, entry_price=50_000)
        candle = Candle(
            timestamp=1_700_000_000_000,
            open=49_500,
            high=49_600,
            low=49_000,
            close=49_200,
            volume=100,
        )
        order = Order(
            symbol="BTC",
            side=Side.LONG,
            price=60_000,
            qty=0.1,
            source=OrderSource.RISK,
            reduce_only=True,
            is_market=True,
        )

        fills = Backtester(BacktestConfig())._simulate_fills(
            [order],
            {"BTC": candle},
            account,
            {"BTC": ExchangeParams()},
            candle.timestamp,
        )

        assert len(fills) == 1
        # Position fully closed
        assert account.symbols["BTC"].position_long.size == 0.0
        # Loss realized (entry 50k, fill near 49.2k close)
        assert fills[0].realized_pnl < 0


class TestGridPanicClose:
    def test_panic_mode_emits_is_market_close(
        self, ema_state, volatility_state, exchange_params
    ):
        engine = GridEngine(GridConfig())
        pos = Position(size=0.1, entry_price=50_000)
        orders = engine.compute_orders(
            "BTC",
            Side.LONG,
            pos,
            ema_state,
            volatility_state,
            10_000,
            0.5,
            exchange_params,
            TradingMode.PANIC,
            mark_price=49_500,
        )
        assert len(orders) == 1
        assert orders[0].is_market is True
        assert orders[0].price == 49_500
        assert orders[0].reduce_only is True


class TestBestPriceTracking:
    def test_update_best_price_long_takes_max(self):
        pos = Position(size=0.1, entry_price=50_000, best_price=50_000)
        pos.update_best_price(50_500, Side.LONG)
        pos.update_best_price(50_300, Side.LONG)
        pos.update_best_price(50_700, Side.LONG)
        assert pos.best_price == 50_700

    def test_update_best_price_short_takes_min(self):
        pos = Position(size=0.1, entry_price=50_000, best_price=50_000)
        pos.update_best_price(49_800, Side.SHORT)
        pos.update_best_price(49_900, Side.SHORT)
        pos.update_best_price(49_500, Side.SHORT)
        assert pos.best_price == 49_500

    def test_update_best_price_ignores_closed_position(self):
        pos = Position()
        pos.update_best_price(50_500, Side.LONG)
        assert pos.best_price == 0.0


class TestTrailingStop:
    def test_example_strategy_trailing_stop_uses_best_price(self):
        strat = ExampleTrendStrategy()
        account = AccountState(balance=10_000, equity=10_000, equity_peak=10_000)
        account.symbols["BTC"] = SymbolState("BTC", last_price=51_500)
        pos = Position(size=0.1, entry_price=50_000, best_price=52_000)
        candle = Candle(1_700_000_000_000, 51_500, 51_600, 51_400, 51_500, 100)
        ctx = TradeContext(
            symbol="BTC",
            side=Side.LONG,
            position=pos,
            account=account,
            candle=candle,
            signal=None,
            current_time_ms=1_700_000_000_000,
            exchange_params=ExchangeParams(),
        )
        # Profit pct = (51500 - 50000) / 50000 = 0.03 > 0, so trailing arms.
        sl = strat.custom_stoploss(ctx, 0.03)
        assert sl is not None
        # 5% below the best_price (52000), not below the current price (51500).
        assert sl == pytest.approx(52_000 * 0.95)

    def test_trailing_stop_not_armed_below_breakeven(self):
        strat = ExampleTrendStrategy()
        account = AccountState(balance=10_000, equity=10_000, equity_peak=10_000)
        account.symbols["BTC"] = SymbolState("BTC", last_price=49_000)
        pos = Position(size=0.1, entry_price=50_000, best_price=50_000)
        candle = Candle(1_700_000_000_000, 49_000, 49_100, 48_900, 49_000, 100)
        ctx = TradeContext(
            symbol="BTC",
            side=Side.LONG,
            position=pos,
            account=account,
            candle=candle,
            signal=None,
            current_time_ms=1_700_000_000_000,
            exchange_params=ExchangeParams(),
        )
        assert strat.custom_stoploss(ctx, -0.02) is None


class _VetoEverything(DefaultStrategy):
    def confirm_trade_entry(self, ctx, qty, price):
        return False


class _CountingStrategy(DefaultStrategy):
    def __init__(self):
        self.entry_calls = 0
        self.exit_calls = 0

    def confirm_trade_entry(self, ctx, qty, price):
        self.entry_calls += 1
        return True

    def confirm_trade_exit(self, ctx, qty, price, exit_reason):
        self.exit_calls += 1
        return True


class TestBacktestStrategyWiring:
    def test_default_strategy_does_not_change_behavior(self):
        candles = make_candles([50_000 + i * 10 for i in range(300)])
        baseline_cfg = BacktestConfig(
            starting_balance=10_000,
            symbols=["BTC"],
            grid=GridConfig(max_grid_levels=3),
        )
        baseline = Backtester(baseline_cfg).run({"BTC": candles})
        with_default = Backtester(baseline_cfg, strategy=DefaultStrategy()).run(
            {"BTC": candles}
        )
        assert baseline.n_trades == with_default.n_trades
        assert baseline.final_balance == pytest.approx(
            with_default.final_balance, rel=1e-9
        )

    def test_veto_strategy_suppresses_all_entries(self):
        candles = make_candles([50_000 - i * 5 for i in range(500)])
        cfg = BacktestConfig(
            starting_balance=10_000,
            symbols=["BTC"],
            grid=GridConfig(max_grid_levels=3, entry_initial_ema_dist=0.002),
        )
        result = Backtester(cfg, strategy=_VetoEverything()).run({"BTC": candles})
        # No grid entries should fill since strategy vetoes them all.
        entry_fills = [f for f in result.fills if not _is_close_fill(f)]
        assert len(entry_fills) == 0

    def test_counting_strategy_is_actually_invoked(self):
        candles = make_candles([50_000 - i * 10 for i in range(200)])
        strat = _CountingStrategy()
        cfg = BacktestConfig(
            starting_balance=10_000,
            symbols=["BTC"],
            grid=GridConfig(max_grid_levels=3, entry_initial_ema_dist=0.002),
        )
        Backtester(cfg, strategy=strat).run({"BTC": candles})
        assert strat.entry_calls > 0


def _is_close_fill(fill) -> bool:
    return fill.realized_pnl != 0.0
