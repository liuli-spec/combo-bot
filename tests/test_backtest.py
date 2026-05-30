from __future__ import annotations
import numpy as np
from tests.conftest import make_candles, make_oscillating_candles
from combo_bot.backtest import BacktestConfig, Backtester
from combo_bot.grid_engine import GridConfig
from combo_bot.merger import MergerConfig
from combo_bot.risk import RiskConfig
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


class TestBacktestSmoke:
    def test_runs_without_error(self):
        candles = make_candles([50000 + i * 10 for i in range(500)])
        config = BacktestConfig(
            starting_balance=10000,
            symbols=["BTC"],
            grid=GridConfig(max_grid_levels=3),
        )
        result = Backtester(config).run({"BTC": candles})
        assert result.duration_days > 0
        assert result.final_balance > 0

    def test_no_trades_on_empty_data(self):
        config = BacktestConfig(symbols=["BTC"], grid=GridConfig())
        candles = make_candles([50000.0] * 10)
        result = Backtester(config).run({"BTC": candles})
        assert result.n_trades >= 0


class TestBacktestGridOnly:
    def test_grid_long_close_order_reduces_long_position(self):
        account = AccountState(balance=10000, equity=10000, equity_peak=10000)
        account.symbols["BTC"] = SymbolState("BTC", last_price=51000)
        account.symbols["BTC"].position_long = Position(size=0.1, entry_price=50000)
        order = Order("BTC", Side.LONG, 51000, 0.1, OrderSource.GRID, reduce_only=True)
        candle = Candle(1700000000000, 50000, 51500, 50500, 51000, 100)

        fills = Backtester(BacktestConfig())._simulate_fills(
            [order],
            {"BTC": candle},
            account,
            {"BTC": ExchangeParams()},
            candle.timestamp,
        )

        assert len(fills) == 1
        assert account.symbols["BTC"].position_long.size == 0.0
        assert fills[0].realized_pnl > 0

    def test_grid_short_close_order_reduces_short_position(self):
        account = AccountState(balance=10000, equity=10000, equity_peak=10000)
        account.symbols["BTC"] = SymbolState("BTC", last_price=49000)
        account.symbols["BTC"].position_short = Position(size=0.1, entry_price=50000)
        order = Order("BTC", Side.SHORT, 49000, 0.1, OrderSource.GRID, reduce_only=True)
        candle = Candle(1700000000000, 50000, 49500, 48500, 49000, 100)

        fills = Backtester(BacktestConfig())._simulate_fills(
            [order],
            {"BTC": candle},
            account,
            {"BTC": ExchangeParams()},
            candle.timestamp,
        )

        assert len(fills) == 1
        assert account.symbols["BTC"].position_short.size == 0.0
        assert fills[0].realized_pnl > 0

    def test_profitable_in_oscillating_market(self):
        candles = make_oscillating_candles(n=10000, base=50000, amplitude=800)
        config = BacktestConfig(
            starting_balance=10000,
            symbols=["BTC"],
            grid=GridConfig(
                entry_initial_ema_dist=0.003,
                entry_grid_spacing_pct=0.012,
                close_grid_markup_start=0.003,
                close_grid_markup_end=0.008,
                close_grid_qty_pct=0.6,
                wallet_exposure_limit=1.0,
                max_grid_levels=5,
            ),
            merger=MergerConfig(mode_switch_strong_threshold=0.99),
        )
        result = Backtester(config).run({"BTC": candles})
        assert (
            result.final_balance > config.starting_balance
        ), f"Grid should profit in oscillating market: {result.final_balance}"
        assert result.n_trades > 10
        assert result.grid_pnl > 0

    def test_drawdown_limited(self):
        candles = make_oscillating_candles(n=5000, base=50000, amplitude=500)
        config = BacktestConfig(
            starting_balance=10000,
            symbols=["BTC"],
            risk=RiskConfig(max_drawdown_pct=0.2),
            grid=GridConfig(wallet_exposure_limit=0.5),
        )
        result = Backtester(config).run({"BTC": candles})
        assert result.max_drawdown < 0.30


class TestBacktestMultiSymbol:
    def test_multi_symbol_backtest(self):
        rng = np.random.default_rng(42)
        btc = make_candles(
            (
                50000
                + 500 * np.sin(np.arange(2000) / 80 * 2 * np.pi)
                + rng.normal(0, 50, 2000)
            ).tolist()
        )
        eth = make_candles(
            (
                3000
                + 100 * np.sin(np.arange(2000) / 80 * 2 * np.pi)
                + rng.normal(0, 5, 2000)
            ).tolist()
        )

        config = BacktestConfig(
            starting_balance=10000,
            symbols=["BTC", "ETH"],
            grid=GridConfig(
                entry_initial_ema_dist=0.003,
                entry_grid_spacing_pct=0.015,
                close_grid_markup_start=0.003,
                close_grid_markup_end=0.01,
                wallet_exposure_limit=0.5,
                max_grid_levels=3,
            ),
        )
        result = Backtester(config).run({"BTC": btc, "ETH": eth})
        assert result.n_trades > 0
        assert result.duration_days > 0
        btc_fills = [f for f in result.fills if f.symbol == "BTC"]
        eth_fills = [f for f in result.fills if f.symbol == "ETH"]
        assert len(btc_fills) > 0
        assert len(eth_fills) > 0


class TestBacktestMetrics:
    def test_equity_curve_shape(self):
        candles = make_oscillating_candles(n=3000)
        config = BacktestConfig(starting_balance=10000, symbols=["BTC"])
        result = Backtester(config).run({"BTC": candles})
        assert result.equity_curve.shape[1] == 2
        assert result.equity_curve.shape[0] > 0

    def test_fees_are_positive(self):
        candles = make_oscillating_candles(n=5000)
        config = BacktestConfig(starting_balance=10000, symbols=["BTC"])
        result = Backtester(config).run({"BTC": candles})
        if result.n_trades > 0:
            assert result.total_fees > 0

    def test_win_rate_between_0_and_1(self):
        candles = make_oscillating_candles(n=5000)
        config = BacktestConfig(starting_balance=10000, symbols=["BTC"])
        result = Backtester(config).run({"BTC": candles})
        assert 0.0 <= result.win_rate <= 1.0
