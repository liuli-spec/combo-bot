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
# Multi-objective (Pareto) support
# ---------------------------------------------------------------------------
# Map an objective metric name to a getter on BacktestResult. Aliases let
# configs use either the short or the full field name.
_METRIC_GETTERS = {
    "adg": lambda r: r.adg,
    "sortino": lambda r: r.sortino_ratio,
    "sortino_ratio": lambda r: r.sortino_ratio,
    "calmar": lambda r: r.calmar_ratio,
    "calmar_ratio": lambda r: r.calmar_ratio,
    "sharpe": lambda r: r.sharpe_ratio,
    "sharpe_ratio": lambda r: r.sharpe_ratio,
    "max_drawdown": lambda r: r.max_drawdown,
    "drawdown": lambda r: r.max_drawdown,
    "win_rate": lambda r: r.win_rate,
    "total_pnl": lambda r: r.total_pnl,
    "n_trades": lambda r: float(r.n_trades),
}

# Penalty assigned to a failed trial, per optimization direction. Maximize
# objectives get a very negative value; minimize objectives a very positive
# one — so a hopeless parameter set is dominated on every axis.
_PENALTY_MAXIMIZE = -1e6
_PENALTY_MINIMIZE = 1e6


def _parse_objectives(specs: list[str] | None) -> list[tuple[str, str]] | None:
    """Parse ``["adg:max", "max_drawdown:min"]`` into ``[(metric, direction)]``.

    Returns None for an empty/None spec (legacy single-scalar mode). Raises
    ValueError on an unknown metric so a typo fails loudly at startup rather
    than silently optimizing the wrong thing.
    """
    if not specs:
        return None
    out: list[tuple[str, str]] = []
    for raw in specs:
        metric, _, goal = str(raw).partition(":")
        metric = metric.strip()
        goal = (goal.strip().lower() or "max")
        if metric not in _METRIC_GETTERS:
            valid = ", ".join(sorted(_METRIC_GETTERS))
            raise ValueError(
                f"unknown optimization metric {metric!r}; valid: {valid}"
            )
        direction = "maximize" if goal in ("max", "maximize") else "minimize"
        out.append((metric, direction))
    return out


def _metric_value(result: BacktestResult, metric: str) -> float:
    getter = _METRIC_GETTERS.get(metric)
    if getter is None:
        return 0.0
    value = getter(result)
    return float(value) if math.isfinite(value) else 0.0


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
    # Round-29: sampler choice.
    #   "tpe"    — Tree-structured Parzen Estimator (Bayesian). Default.
    #              Best sample efficiency on a small trial budget.
    #   "cmaes"  — CMA-ES (Covariance-Matrix-Adaptation Evolution Strategy).
    #              Continuous-parameter evolutionary algorithm; very strong
    #              when most params are floats (grid spacing, WEL, etc).
    #   "nsga2"  — NSGA-II genetic algorithm. Multi-objective oriented;
    #              works as a pure GA when given one objective. Higher
    #              trial budget recommended (1000+).
    #   "random" — Random search baseline. Useful sanity check.
    sampler: str = "tpe"
    # Multi-objective (Pareto) optimization. A list like
    # ["adg:max", "max_drawdown:min"] turns the search into a true
    # multi-objective NSGA-II run whose output is the Pareto front — no
    # arbitrary scalar weighting. None keeps the legacy single-scalar
    # ``combined`` score (sortino/calmar/adg/drawdown weighting). Valid
    # metrics: adg, sortino_ratio, calmar_ratio, sharpe_ratio,
    # max_drawdown, win_rate, total_pnl, n_trades.
    objectives: list[str] | None = None


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
        # None → legacy single-scalar score; otherwise [(metric, direction)].
        self._objective_specs = _parse_objectives(config.objectives)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _build_sampler(self):
        """Construct the Optuna sampler from the configured name.

        Falls back to TPE on unknown values with a warning, so a typo
        in the config doesn't silently degrade to random search.
        """
        name = (self._config.sampler or "tpe").lower()
        if name == "tpe":
            return optuna.samplers.TPESampler()
        if name == "cmaes":
            return optuna.samplers.CmaEsSampler()
        if name == "nsga2":
            return optuna.samplers.NSGAIISampler()
        if name == "random":
            return optuna.samplers.RandomSampler()
        logger.warning(
            "Unknown sampler %r — falling back to TPESampler. Valid: "
            "tpe / cmaes / nsga2 / random.",
            name,
        )
        return optuna.samplers.TPESampler()

    def run(self, callbacks: list | None = None) -> dict:
        """Run the full optimization and return the best parameter set.

        ``callbacks`` is forwarded to ``study.optimize`` and receives
        ``(study, trial)`` after each completed trial — useful for
        progress reporting in the web UI without polling Optuna internals.
        """
        sampler = self._build_sampler()
        specs = self._objective_specs

        common = dict(
            study_name=self._config.study_name,
            storage=self._config.storage,
            load_if_exists=True,
            sampler=sampler,
        )
        if specs is None:
            study = optuna.create_study(direction="maximize", **common)
        else:
            study = optuna.create_study(
                directions=[direction for _, direction in specs], **common
            )

        study.optimize(
            self._objective,
            n_trials=self._config.n_trials,
            n_jobs=self._config.n_jobs,
            show_progress_bar=False,
            callbacks=callbacks or [],
        )

        if specs is None:
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

        # Multi-objective: the result is the Pareto front (non-dominated
        # trials). No arbitrary scalar weighting — the caller/operator picks
        # a point on the risk/return trade-off curve.
        front = study.best_trials
        logger.info(
            "Optimization complete  |  pareto_front=%d  trials=%d  objectives=%s",
            len(front),
            len(study.trials),
            [f"{m}:{'max' if d == 'maximize' else 'min'}" for m, d in specs],
        )
        solutions = []
        for t in sorted(front, key=lambda tr: tr.number):
            solutions.append(
                {
                    "params": t.params,
                    "objectives": {
                        metric: value for (metric, _), value in zip(specs, t.values)
                    },
                    "grid": _grid_config_from_params(t.params).__dict__,
                    "trend": _trend_config_from_params(t.params).__dict__,
                    "merger": _merger_config_from_params(t.params).__dict__,
                }
            )
        return {
            "pareto_front": solutions,
            "objectives": [
                f"{m}:{'max' if d == 'maximize' else 'min'}" for m, d in specs
            ],
            "n_trials": len(study.trials),
            "study_name": study.study_name,
        }

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    def _objective(self, trial: Trial):
        """Evaluate a parameter set. Returns a scalar in legacy mode, or a
        tuple of objective values in multi-objective (Pareto) mode."""
        grid_cfg = self._suggest_grid_params(trial)
        trend_cfg = self._suggest_trend_params(trial)
        merger_cfg = self._suggest_merger_params(trial)

        bt_config = BacktestConfig(
            grid=grid_cfg,
            trend=trend_cfg,
            merger=merger_cfg,
            symbols=list(self._symbols),
        )

        metrics = self._walk_forward_evaluate(bt_config)
        specs = self._objective_specs

        if specs is None:
            # Legacy single-scalar path (with pruning).
            score = -1e6 if metrics is None else metrics["_combined"]
            trial.report(score, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()
            return score

        # Multi-objective: return one value per objective. Failed trials
        # get a dominated penalty on every axis. (Optuna's standard pruners
        # don't apply to multi-objective studies, so we don't report/prune.)
        if metrics is None:
            return tuple(
                _PENALTY_MAXIMIZE if direction == "maximize" else _PENALTY_MINIMIZE
                for _, direction in specs
            )
        return tuple(metrics.get(metric, 0.0) for metric, _ in specs)

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
    def _walk_forward_evaluate(self, config: BacktestConfig) -> dict[str, float] | None:
        """Split data chronologically, train on front, score on back.

        Repeats across ``walk_forward_splits`` rolling windows and returns
        the per-metric mean across splits (including the legacy ``_combined``
        scalar). Returns None when the parameter set is unevaluable
        (insufficient data, or zero in-sample trades) so the caller can
        assign a per-objective penalty.
        """
        n_splits = self._config.walk_forward_splits
        total_len = self._min_length

        if total_len < 2:
            return None

        window_size = total_len // n_splits
        if window_size < 2:
            return None

        metric_rows: list[dict[str, float]] = []

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
                    return None

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

            metric_rows.append(self._result_metrics(result))

        if not metric_rows:
            return None

        keys = metric_rows[0].keys()
        return {k: sum(row[k] for row in metric_rows) / len(metric_rows) for k in keys}

    def _result_metrics(self, result: BacktestResult) -> dict[str, float]:
        """All optimizable metrics for one backtest plus the legacy scalar."""
        metrics = {name: _metric_value(result, name) for name in _METRIC_GETTERS}
        metrics["_combined"] = self._compute_score(result)
        return metrics

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
