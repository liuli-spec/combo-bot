from __future__ import annotations
import numpy as np
import pytest

from combo_bot.grid_engine import GridConfig
from combo_bot.types import Candle

rust_backtest = pytest.importorskip("combo_bot.rust_backtest")
if not rust_backtest.RUST_AVAILABLE:
    pytest.skip("Rust extension not built", allow_module_level=True)


def make_oscillating(n: int, base: float = 50000.0, amp: float = 500.0) -> np.ndarray:
    t = np.arange(n)
    close = base + amp * np.sin(t / 80 * 2 * np.pi)
    spread = amp * 0.05
    return np.column_stack(
        [
            close,
            close + spread,
            close - spread,
            close,
            np.full(n, 100.0),
        ]
    )


class TestRustBacktestBasic:
    def test_runs_without_error(self):
        arr = make_oscillating(2000)
        cfg = GridConfig(wallet_exposure_limit=0.5, ema_span_0=30, ema_span_1=60)
        result = rust_backtest.run_rust_backtest(arr, cfg)
        assert result.final_equity > 0
        assert result.equity_curve.size > 0

    def test_oscillating_market_profitable(self):
        arr = make_oscillating(5000, amp=800)
        cfg = GridConfig(
            entry_initial_ema_dist=0.003,
            entry_grid_spacing_pct=0.015,
            close_grid_markup_start=0.003,
            close_grid_markup_end=0.008,
            close_grid_qty_pct=0.6,
            wallet_exposure_limit=0.5,
            ema_span_0=30,
            ema_span_1=60,
        )
        result = rust_backtest.run_rust_backtest(arr, cfg)
        assert result.final_balance > 10000
        assert result.n_trades > 5

    def test_records_drawdown_in_range(self):
        arr = make_oscillating(2000)
        cfg = GridConfig(wallet_exposure_limit=0.5, ema_span_0=30, ema_span_1=60)
        result = rust_backtest.run_rust_backtest(arr, cfg)
        assert 0.0 <= result.max_drawdown <= 1.0

    def test_returns_fills(self):
        arr = make_oscillating(2000)
        cfg = GridConfig(wallet_exposure_limit=0.5, ema_span_0=30, ema_span_1=60)
        result = rust_backtest.run_rust_backtest(arr, cfg)
        assert isinstance(result.fills, list)
        if result.fills:
            f = result.fills[0]
            assert "bar_index" in f
            assert "qty" in f
            assert "price" in f


class TestRustBacktestVsPython:
    def test_rust_much_faster_than_python(self):
        """Rust should be at least 50x faster than the Python backtester."""
        import time
        from combo_bot.backtest import BacktestConfig, Backtester

        n = 5000
        arr = make_oscillating(n)
        py_candles = [
            Candle(
                timestamp=1700000000000 + i * 60000,
                open=float(arr[i, 0]),
                high=float(arr[i, 1]),
                low=float(arr[i, 2]),
                close=float(arr[i, 3]),
                volume=float(arr[i, 4]),
            )
            for i in range(n)
        ]

        cfg = GridConfig(wallet_exposure_limit=0.5, ema_span_0=30, ema_span_1=60)

        t0 = time.perf_counter()
        rust_backtest.run_rust_backtest(arr, cfg)
        rust_time = time.perf_counter() - t0

        py_cfg = BacktestConfig(starting_balance=10000, symbols=["BTC"], grid=cfg)
        t0 = time.perf_counter()
        Backtester(py_cfg).run({"BTC": py_candles})
        py_time = time.perf_counter() - t0

        speedup = py_time / rust_time
        assert speedup >= 50.0, f"Expected ≥50x speedup, got {speedup:.1f}x"


class TestRustBacktestMetrics:
    def test_sharpe_finite(self):
        arr = make_oscillating(3000)
        cfg = GridConfig(wallet_exposure_limit=0.5, ema_span_0=30, ema_span_1=60)
        result = rust_backtest.run_rust_backtest(arr, cfg)
        assert np.isfinite(result.sharpe_ratio)
        assert np.isfinite(result.sortino_ratio)

    def test_total_return_matches_equity(self):
        arr = make_oscillating(2000)
        cfg = GridConfig(wallet_exposure_limit=0.5, ema_span_0=30, ema_span_1=60)
        result = rust_backtest.run_rust_backtest(arr, cfg)
        expected = result.equity_curve[-1] / result.equity_curve[0] - 1.0
        assert abs(result.total_return - expected) < 1e-9


class TestRustBacktestEdgeCases:
    def test_empty_balance_no_trades(self):
        arr = make_oscillating(500)
        cfg = GridConfig(wallet_exposure_limit=0.0, ema_span_0=30, ema_span_1=60)
        result = rust_backtest.run_rust_backtest(arr, cfg)
        assert result.n_trades == 0

    def test_handles_short_data(self):
        arr = make_oscillating(100)
        cfg = GridConfig(wallet_exposure_limit=0.5, ema_span_0=30, ema_span_1=60)
        result = rust_backtest.run_rust_backtest(arr, cfg)
        assert result.equity_curve.size > 0
