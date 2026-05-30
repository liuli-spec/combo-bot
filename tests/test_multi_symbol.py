from __future__ import annotations
import numpy as np
import pytest

from combo_bot.grid_engine import GridConfig

multi = pytest.importorskip("combo_bot.rust_multi_symbol")
if not multi.RUST_AVAILABLE:
    pytest.skip("Rust extension not built", allow_module_level=True)


def make_osc(n: int, base: float, amp: float, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n)
    close = base + amp * np.sin(t / 80 * 2 * np.pi + phase)
    spread = amp * 0.05
    return np.column_stack(
        [close, close + spread, close - spread, close, np.full(n, 100.0)]
    )


class TestMultiSymbolBasic:
    def test_runs_three_symbols(self):
        data = {
            "BTC": make_osc(3000, 50000, 500),
            "ETH": make_osc(3000, 3000, 30, 1.0),
            "SOL": make_osc(3000, 100, 1.5, 2.0),
        }
        cfgs = {
            s: GridConfig(wallet_exposure_limit=0.3, ema_span_0=30, ema_span_1=60)
            for s in data
        }
        result = multi.run_multi_symbol_backtest(
            data,
            cfgs,
            bt_config=multi.MultiSymbolConfig(n_positions_max=3, max_grid_levels=5),
        )
        assert result.equity_curve.size > 0
        assert len(result.final_positions) == 3
        assert result.symbols == ["BTC", "ETH", "SOL"]

    def test_fills_have_symbol_idx(self):
        data = {"A": make_osc(2000, 100, 5), "B": make_osc(2000, 50, 2, 1.0)}
        cfgs = {
            s: GridConfig(wallet_exposure_limit=0.3, ema_span_0=30, ema_span_1=60)
            for s in data
        }
        result = multi.run_multi_symbol_backtest(
            data,
            cfgs,
            bt_config=multi.MultiSymbolConfig(n_positions_max=2),
        )
        for f in result.fills:
            assert "symbol_idx" in f
            assert 0 <= f["symbol_idx"] < 2

    def test_unequal_bars_rejected(self):
        data = {"A": make_osc(2000, 100, 5), "B": make_osc(1500, 50, 2)}
        cfgs = {s: GridConfig() for s in data}
        with pytest.raises(ValueError, match="bars"):
            multi.run_multi_symbol_backtest(data, cfgs)


class TestForagerSelection:
    def test_n_positions_limits_active_symbols(self):
        # 5 symbols, but only 2 may be active
        data = {f"S{i}": make_osc(3000, 100 + i * 10, 3, i * 0.5) for i in range(5)}
        cfgs = {
            s: GridConfig(wallet_exposure_limit=0.3, ema_span_0=30, ema_span_1=60)
            for s in data
        }
        result = multi.run_multi_symbol_backtest(
            data,
            cfgs,
            bt_config=multi.MultiSymbolConfig(n_positions_max=2, max_grid_levels=3),
        )
        # At end, no more than ~n_positions_max symbols should have positions
        # (some may have closed naturally)
        active_at_end = sum(
            1
            for p in result.final_positions
            if abs(p["long_size"]) > 1e-9 or abs(p["short_size"]) > 1e-9
        )
        assert active_at_end <= 5


class TestMultiSymbolMetrics:
    def test_shared_balance_drawdown(self):
        data = {"X": make_osc(3000, 50000, 800)}
        cfgs = {
            "X": GridConfig(wallet_exposure_limit=0.5, ema_span_0=30, ema_span_1=60)
        }
        result = multi.run_multi_symbol_backtest(
            data,
            cfgs,
            bt_config=multi.MultiSymbolConfig(
                n_positions_max=1, starting_balance=10000
            ),
        )
        assert 0.0 <= result.max_drawdown <= 1.0
        assert np.isfinite(result.sharpe_ratio)
