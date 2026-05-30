from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass

from combo_bot.backtest import BacktestConfig, Backtester
from combo_bot.grid_engine import GridConfig
from combo_bot.merger import MergerConfig
from combo_bot.trend_signal import TrendConfig
from combo_bot.types import BacktestResult, Candle, ExchangeParams

try:
    import optuna
    from optuna import Study, Trial

    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    optuna = None  # type: ignore[assignment]
    Study = None  # type: ignore[assignment, misc]
    Trial = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------
_SCORE_SORTINO_W = 0.35
_SCORE_CALMAR_W = 0.25
_SCORE_ADG_W = 0.25
_SCORE_DRAWDOWN_W = 0.15
_ADG_ANNUALIZE = 365.0
_ADG_SCALE = 10.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class OptimizeConfig:
    """Controls the Optuna-based parameter search."""

    n_trials: int = 500
    n_jobs: int = 4
    metric: str = "combined"
    walk_forward_splits: int = 3
    train_ratio: float = 0.7
    study_name: str = "combo_bot"
    storage: str | None = None


# ---------------------------------------------------------------------------
# Reconstruct configs from a flat params dict (e.g. FrozenTrial.params)
# ---------------------------------------------------------------------------
def _grid_config_from_params(params: dict) -> GridConfig:
    return GridConfig(
        entry_initial_ema_dist=params["grid_entry_initial_ema_dist"],
        entry_grid_spacing_pct=params["grid_entry_grid_spacing_pct"],
        entry_grid_double_down_factor=params["grid_double_down_factor"],
        wallet_exposure_limit=params["grid_wallet_exposure_limit"],
        total_wallet_exposure_limit=params["grid_total_wallet_exposure_limit"],
        close_grid_markup_start=params["grid_close_grid_markup_start"],
        close_grid_markup_end=params["grid_close_grid_markup_end"],
    )


def _trend_config_from_params(params: dict) -> TrendConfig:
    return TrendConfig(
        rsi_period=params["trend_rsi_period"],
        macd_fast=params["trend_macd_fast"],
        macd_slow=params["trend_macd_slow"],
        strong_threshold=params["trend_strong_threshold"],
        weak_threshold=params["trend_weak_threshold"],
    )


def _merger_config_from_params(params: dict) -> MergerConfig:
    return MergerConfig(
        grid_depth_limit_in_downtrend=params["merger_grid_depth_limit_in_downtrend"],
        trend_position_max_pct=params["merger_trend_position_max_pct"],
        mode_switch_strong_threshold=params["merger_mode_switch_strong_threshold"],
    )


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
class Optimizer:
    """Parameter optimizer for the combined trading bot.

    Optimizes across three layers:
      1. Grid engine parameters  (GridConfig)
      2. Trend signal parameters (TrendConfig)
      3. Decision merger parameters (MergerConfig)

    Uses walk-forward validation to guard against over-fitting.
    """

    def __init__(
        self,
        config: OptimizeConfig,
        candle_data: dict[str, list[Candle]],
        exchange_params: dict[str, ExchangeParams] | None = None,
        funding_rates: dict[str, list[float]] | None = None,
    ) -> None:
        if not _OPTUNA_AVAILABLE:
            raise ImportError(
                "optuna is required for optimization.  "
                "Install it with:  pip install optuna"
            )
        self._config = config
        self._candle_data = candle_data
        self._exchange_params = exchange_params
        self._funding_rates = funding_rates
        self._symbols = list(candle_data.keys())
        self._min_length = min(len(v) for v in candle_data.values())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> dict:
        """Run the full optimization and return the best parameter set."""
        study = optuna.create_study(
            study_name=self._config.study_name,
            storage=self._config.storage,
            direction="maximize",
            load_if_exists=True,
        )
        study.optimize(
            self._objective,
            n_trials=self._config.n_trials,
            n_jobs=self._config.n_jobs,
            show_progress_bar=True,
        )

        best = study.best_trial
        logger.info(
            "Optimization complete  |  best value=%.6f  trial=%d",
            best.value,
            best.number,
        )

        print_optimization_report(study)

        return {
            "best_value": best.value,
            "best_params": best.params,
            "grid": _grid_config_from_params(best.params).__dict__,
            "trend": _trend_config_from_params(best.params).__dict__,
            "merger": _merger_config_from_params(best.params).__dict__,
            "n_trials": len(study.trials),
            "study_name": study.study_name,
        }

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    def _objective(self, trial: Trial) -> float:
        """Evaluate a single parameter combination via walk-forward."""
        grid_cfg = self._suggest_grid_params(trial)
        trend_cfg = self._suggest_trend_params(trial)
        merger_cfg = self._suggest_merger_params(trial)

        bt_config = BacktestConfig(
            grid=grid_cfg,
            trend=trend_cfg,
            merger=merger_cfg,
            symbols=list(self._symbols),
        )

        score = self._walk_forward_evaluate(bt_config)

        # Prune hopeless trials early.
        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return score

    # ------------------------------------------------------------------
    # Parameter suggestion helpers
    # ------------------------------------------------------------------
    def _suggest_grid_params(self, trial: Trial) -> GridConfig:
        return GridConfig(
            entry_initial_ema_dist=trial.suggest_float(
                "grid_entry_initial_ema_dist",
                0.002,
                0.02,
            ),
            entry_grid_spacing_pct=trial.suggest_float(
                "grid_entry_grid_spacing_pct",
                0.01,
                0.05,
            ),
            entry_grid_double_down_factor=trial.suggest_float(
                "grid_double_down_factor",
                0.8,
                2.0,
            ),
            wallet_exposure_limit=trial.suggest_float(
                "grid_wallet_exposure_limit",
                0.5,
                2.0,
            ),
            total_wallet_exposure_limit=trial.suggest_float(
                "grid_total_wallet_exposure_limit",
                1.0,
                3.0,
            ),
            close_grid_markup_start=trial.suggest_float(
                "grid_close_grid_markup_start",
                0.002,
                0.015,
            ),
            close_grid_markup_end=trial.suggest_float(
                "grid_close_grid_markup_end",
                0.005,
                0.03,
            ),
        )

    def _suggest_trend_params(self, trial: Trial) -> TrendConfig:
        return TrendConfig(
            rsi_period=trial.suggest_int("trend_rsi_period", 7, 21),
            macd_fast=trial.suggest_int("trend_macd_fast", 8, 16),
            macd_slow=trial.suggest_int("trend_macd_slow", 21, 34),
            strong_threshold=trial.suggest_float(
                "trend_strong_threshold",
                0.3,
                0.8,
            ),
            weak_threshold=trial.suggest_float(
                "trend_weak_threshold",
                0.1,
                0.4,
            ),
        )

    def _suggest_merger_params(self, trial: Trial) -> MergerConfig:
        return MergerConfig(
            grid_depth_limit_in_downtrend=trial.suggest_int(
                "merger_grid_depth_limit_in_downtrend",
                1,
                5,
            ),
            trend_position_max_pct=trial.suggest_float(
                "merger_trend_position_max_pct",
                0.0,
                0.3,
            ),
            mode_switch_strong_threshold=trial.suggest_float(
                "merger_mode_switch_strong_threshold",
                0.4,
                0.9,
            ),
        )

    # ------------------------------------------------------------------
    # Walk-forward validation
    # ------------------------------------------------------------------
    def _walk_forward_evaluate(self, config: BacktestConfig) -> float:
        """Split data chronologically, train on front, score on back.

        Repeats across ``walk_forward_splits`` rolling windows and
        returns the mean test score.
        """
        n_splits = self._config.walk_forward_splits
        total_len = self._min_length

        if total_len < 2:
            return -1e6

        window_size = total_len // n_splits
        if window_size < 2:
            return -1e6

        test_scores: list[float] = []

        for split_idx in range(n_splits):
            start = split_idx * (total_len - window_size) // max(n_splits - 1, 1)
            end = start + window_size
            if end > total_len:
                end = total_len
                start = max(0, end - window_size)

            split_len = end - start
            train_end = start + int(split_len * self._config.train_ratio)

            if train_end - start < 2 or end - train_end < 2:
                continue

            # -- train sanity check: parameter set must produce at least
            # one trade in-sample.  If it fails even on the data it
            # "knows", the trial is hopeless — short-circuit to penalty.
            train_candles = {
                s: self._candle_data[s][start:train_end] for s in self._symbols
            }
            if len(next(iter(train_candles.values()))) >= 100:
                train_result = Backtester(config).run(
                    train_candles,
                    exchange_params=self._exchange_params,
                )
                if train_result.n_trades == 0:
                    return -1e6

            # -- test partition --
            test_candles = {
                s: self._candle_data[s][train_end:end] for s in self._symbols
            }
            test_funding: dict[str, list[float]] | None = None
            if self._funding_rates is not None:
                test_funding = {
                    s: self._funding_rates[s][train_end:end]
                    for s in self._symbols
                    if s in self._funding_rates
                }

            backtester = Backtester(config)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = backtester.run(
                    test_candles,
                    funding_rates=test_funding,
                    exchange_params=self._exchange_params,
                )

            test_scores.append(self._compute_score(result))

        if not test_scores:
            return -1e6

        return sum(test_scores) / len(test_scores)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _compute_score(self, result: BacktestResult) -> float:
        """Combine backtest metrics into a single scalar.

        score = sortino * 0.35
               + calmar * 0.25
               + adg * 365 * 10 * 0.25
               - max_drawdown * 0.15
        """
        sortino = result.sortino_ratio if math.isfinite(result.sortino_ratio) else 0.0
        calmar = result.calmar_ratio if math.isfinite(result.calmar_ratio) else 0.0
        adg = result.adg if math.isfinite(result.adg) else 0.0
        max_dd = result.max_drawdown if math.isfinite(result.max_drawdown) else 1.0

        return (
            sortino * _SCORE_SORTINO_W
            + calmar * _SCORE_CALMAR_W
            + adg * _ADG_ANNUALIZE * _ADG_SCALE * _SCORE_ADG_W
            - max_dd * _SCORE_DRAWDOWN_W
        )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_optimization_report(study: Study) -> None:  # type: ignore[type-arg]
    """Pretty-print a summary of the optimization study."""
    if not _OPTUNA_AVAILABLE or study is None:
        logger.warning("Cannot print report: optuna not available or study is None.")
        return

    border = "=" * 68
    thin = "-" * 68

    print(f"\n{border}")
    print(f"  OPTIMIZATION REPORT  --  {study.study_name}")
    print(border)

    best = study.best_trial
    print(f"  Best trial:    #{best.number}")
    print(f"  Best score:    {best.value:.6f}")
    print(f"  Total trials:  {len(study.trials)}")
    print(thin)

    print("  Best parameters:")
    for name, value in sorted(best.params.items()):
        if isinstance(value, float):
            print(f"    {name:45s}  {value:.6f}")
        else:
            print(f"    {name:45s}  {value}")

    print(thin)

    # Top-5 trials
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top_n = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    print("  Top 5 trials:")
    for t in top_n:
        print(f"    trial #{t.number:>4d}   score={t.value:>10.6f}")

    # Pruning stats
    n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    n_failed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL)
    print(thin)
    print(f"  Completed: {len(completed)}   Pruned: {n_pruned}   Failed: {n_failed}")
    print(f"{border}\n")
