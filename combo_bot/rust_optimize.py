from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np

from combo_bot.grid_engine import GridConfig
from combo_bot.rust_backtest import (
    RUST_AVAILABLE,
    RustBacktestConfig,
    RustBacktestResult,
    candles_to_array,
    run_rust_backtest,
)
from combo_bot.rust_multi_symbol import (
    MultiSymbolConfig,
    MultiSymbolResult,
    run_multi_symbol_backtest,
)
from combo_bot.types import Candle, ExchangeParams

logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    Trial = None
    OPTUNA_AVAILABLE = False


@dataclass
class OptimizeBounds:
    """Parameter search bounds for grid optimization."""
    entry_initial_ema_dist: tuple[float, float] = (0.002, 0.02)
    entry_initial_qty_pct: tuple[float, float] = (0.005, 0.03)
    entry_grid_spacing_pct: tuple[float, float] = (0.008, 0.05)
    entry_grid_double_down_factor: tuple[float, float] = (1.0, 1.8)
    close_grid_markup_start: tuple[float, float] = (0.002, 0.015)
    close_grid_markup_end: tuple[float, float] = (0.005, 0.03)
    close_grid_qty_pct: tuple[float, float] = (0.3, 0.95)
    wallet_exposure_limit: tuple[float, float] = (0.3, 2.0)
    ema_span_0: tuple[int, int] = (50, 800)
    ema_span_1: tuple[int, int] = (100, 1500)


@dataclass
class RustOptimizeConfig:
    n_trials: int = 200
    n_jobs: int = 1
    walk_forward_splits: int = 3
    train_ratio: float = 0.7
    study_name: str = "combo_bot_rust"
    storage: str | None = None
    seed: int = 42
    sortino_weight: float = 0.35
    calmar_weight: float = 0.25
    return_weight: float = 0.25
    drawdown_weight: float = 0.15
    min_trades: int = 10
    backtest_config: RustBacktestConfig = field(default_factory=RustBacktestConfig)
    bounds: OptimizeBounds = field(default_factory=OptimizeBounds)


def suggest_grid_config(trial: Trial, bounds: OptimizeBounds) -> GridConfig:
    """Sample a GridConfig from Optuna trial within the given bounds."""
    markup_start = trial.suggest_float(
        "close_grid_markup_start", *bounds.close_grid_markup_start
    )
    markup_end_low = max(bounds.close_grid_markup_end[0], markup_start + 0.001)
    markup_end_high = max(markup_end_low + 0.001, bounds.close_grid_markup_end[1])
    markup_end = trial.suggest_float("close_grid_markup_end", markup_end_low, markup_end_high)

    ema_0 = trial.suggest_int("ema_span_0", *bounds.ema_span_0)
    ema_1_low = max(bounds.ema_span_1[0], ema_0 + 10)
    ema_1_high = max(ema_1_low + 10, bounds.ema_span_1[1])
    ema_1 = trial.suggest_int("ema_span_1", ema_1_low, ema_1_high)

    return GridConfig(
        entry_initial_ema_dist=trial.suggest_float(
            "entry_initial_ema_dist", *bounds.entry_initial_ema_dist
        ),
        entry_initial_qty_pct=trial.suggest_float(
            "entry_initial_qty_pct", *bounds.entry_initial_qty_pct
        ),
        entry_grid_spacing_pct=trial.suggest_float(
            "entry_grid_spacing_pct", *bounds.entry_grid_spacing_pct
        ),
        entry_grid_double_down_factor=trial.suggest_float(
            "entry_grid_double_down_factor", *bounds.entry_grid_double_down_factor
        ),
        close_grid_markup_start=markup_start,
        close_grid_markup_end=markup_end,
        close_grid_qty_pct=trial.suggest_float(
            "close_grid_qty_pct", *bounds.close_grid_qty_pct
        ),
        wallet_exposure_limit=trial.suggest_float(
            "wallet_exposure_limit", *bounds.wallet_exposure_limit
        ),
        ema_span_0=float(ema_0),
        ema_span_1=float(ema_1),
        max_grid_levels=8,
    )


def compute_score(
    result: RustBacktestResult | MultiSymbolResult,
    cfg: RustOptimizeConfig,
) -> float:
    """Combine backtest metrics into a single objective for Optuna."""
    if result.liquidated:
        return -1e6
    if result.n_trades < cfg.min_trades:
        return -1e6

    sortino = result.sortino_ratio if math.isfinite(result.sortino_ratio) else 0.0
    calmar = result.calmar_ratio if math.isfinite(result.calmar_ratio) else 0.0
    total_return = result.total_return if math.isfinite(result.total_return) else 0.0
    max_dd = result.max_drawdown if math.isfinite(result.max_drawdown) else 1.0

    return (
        sortino * cfg.sortino_weight
        + calmar * cfg.calmar_weight
        + total_return * 10.0 * cfg.return_weight
        - max_dd * cfg.drawdown_weight
    )


def walk_forward_score(
    candle_array: np.ndarray,
    grid_config: GridConfig,
    cfg: RustOptimizeConfig,
    exchange_params: ExchangeParams | None = None,
) -> float:
    """Score one parameter set across multiple walk-forward windows.

    Splits data chronologically into rolling train/test windows. Returns the
    mean test score; if every window failed, returns a large negative penalty.
    """
    n = candle_array.shape[0]
    splits = max(1, cfg.walk_forward_splits)
    window = n // splits
    if window < 1000:
        # Too little data — score whole series once.
        result = run_rust_backtest(candle_array, grid_config, exchange_params, cfg.backtest_config)
        return compute_score(result, cfg)

    scores = []
    for split_idx in range(splits):
        start = split_idx * (n - window) // max(splits - 1, 1)
        end = min(start + window, n)
        if end - start < 500:
            continue
        sub = candle_array[start:end]
        train_end = start + int((end - start) * cfg.train_ratio)
        test = candle_array[train_end:end]
        if test.shape[0] < 500:
            continue
        try:
            result = run_rust_backtest(test, grid_config, exchange_params, cfg.backtest_config)
            scores.append(compute_score(result, cfg))
        except Exception as e:
            logger.debug("walk-forward window failed: %s", e)
            continue

    if not scores:
        return -1e6
    return float(np.mean(scores))


class RustOptimizer:
    """Single-symbol grid optimizer using the Rust backtest engine."""

    def __init__(
        self,
        candles: list[Candle] | np.ndarray,
        config: RustOptimizeConfig | None = None,
        exchange_params: ExchangeParams | None = None,
    ):
        if not RUST_AVAILABLE:
            raise RuntimeError("Rust extension not installed")
        if not OPTUNA_AVAILABLE:
            raise ImportError("optuna is required: pip install optuna")

        self.candles = (
            candles if isinstance(candles, np.ndarray) else candles_to_array(candles)
        )
        self.config = config or RustOptimizeConfig()
        self.exchange_params = exchange_params

    def _objective(self, trial: Trial) -> float:
        grid_cfg = suggest_grid_config(trial, self.config.bounds)
        return walk_forward_score(
            self.candles, grid_cfg, self.config, self.exchange_params
        )

    def run(self) -> dict[str, Any]:
        sampler = optuna.samplers.TPESampler(seed=self.config.seed)
        study = optuna.create_study(
            study_name=self.config.study_name,
            storage=self.config.storage,
            direction="maximize",
            sampler=sampler,
            load_if_exists=True,
        )
        study.optimize(
            self._objective,
            n_trials=self.config.n_trials,
            n_jobs=self.config.n_jobs,
            show_progress_bar=False,
        )

        best = study.best_trial
        best_cfg = self._materialize_config(best.params)
        final_result = run_rust_backtest(
            self.candles, best_cfg, self.exchange_params, self.config.backtest_config
        )

        return {
            "best_score": float(best.value),
            "best_params": dict(best.params),
            "best_config": best_cfg,
            "final_balance": final_result.final_balance,
            "total_return": final_result.total_return,
            "max_drawdown": final_result.max_drawdown,
            "sharpe_ratio": final_result.sharpe_ratio,
            "sortino_ratio": final_result.sortino_ratio,
            "calmar_ratio": final_result.calmar_ratio,
            "n_trades": final_result.n_trades,
            "n_trials_completed": len([t for t in study.trials if t.value is not None]),
            "study": study,
        }

    @staticmethod
    def _materialize_config(params: dict) -> GridConfig:
        return GridConfig(
            entry_initial_ema_dist=params["entry_initial_ema_dist"],
            entry_initial_qty_pct=params["entry_initial_qty_pct"],
            entry_grid_spacing_pct=params["entry_grid_spacing_pct"],
            entry_grid_double_down_factor=params["entry_grid_double_down_factor"],
            close_grid_markup_start=params["close_grid_markup_start"],
            close_grid_markup_end=params["close_grid_markup_end"],
            close_grid_qty_pct=params["close_grid_qty_pct"],
            wallet_exposure_limit=params["wallet_exposure_limit"],
            ema_span_0=float(params["ema_span_0"]),
            ema_span_1=float(params["ema_span_1"]),
            max_grid_levels=8,
        )


class RustMultiSymbolOptimizer:
    """Multi-symbol grid optimizer. One parameter set is shared across all symbols
    (per-symbol optimization can be done by calling RustOptimizer per symbol)."""

    def __init__(
        self,
        candle_data: dict[str, list[Candle] | np.ndarray],
        config: RustOptimizeConfig | None = None,
        ms_config: MultiSymbolConfig | None = None,
        exchange_params: dict[str, ExchangeParams] | None = None,
    ):
        if not RUST_AVAILABLE:
            raise RuntimeError("Rust extension not installed")
        if not OPTUNA_AVAILABLE:
            raise ImportError("optuna is required")

        self.candle_data = candle_data
        self.config = config or RustOptimizeConfig()
        self.ms_config = ms_config or MultiSymbolConfig()
        self.exchange_params = exchange_params
        self.symbols = list(candle_data.keys())

    def _objective(self, trial: Trial) -> float:
        grid_cfg = suggest_grid_config(trial, self.config.bounds)
        grid_configs = {s: grid_cfg for s in self.symbols}
        try:
            result = run_multi_symbol_backtest(
                self.candle_data, grid_configs, self.exchange_params, self.ms_config
            )
            return compute_score(result, self.config)
        except Exception as e:
            logger.debug("trial failed: %s", e)
            return -1e6

    def run(self) -> dict[str, Any]:
        sampler = optuna.samplers.TPESampler(seed=self.config.seed)
        study = optuna.create_study(
            study_name=self.config.study_name,
            direction="maximize",
            sampler=sampler,
        )
        study.optimize(
            self._objective,
            n_trials=self.config.n_trials,
            n_jobs=self.config.n_jobs,
            show_progress_bar=False,
        )

        best = study.best_trial
        best_cfg = RustOptimizer._materialize_config(best.params)
        final = run_multi_symbol_backtest(
            self.candle_data,
            {s: best_cfg for s in self.symbols},
            self.exchange_params,
            self.ms_config,
        )

        return {
            "best_score": float(best.value),
            "best_params": dict(best.params),
            "best_config": best_cfg,
            "final_balance": final.final_balance,
            "total_return": final.total_return,
            "max_drawdown": final.max_drawdown,
            "sharpe_ratio": final.sharpe_ratio,
            "sortino_ratio": final.sortino_ratio,
            "n_trades": final.n_trades,
            "n_trials_completed": len(study.trials),
            "study": study,
        }
