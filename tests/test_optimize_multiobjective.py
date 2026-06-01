"""Multi-objective (Pareto) optimization tests.

The legacy single-scalar path collapses sortino/calmar/adg/drawdown into one
weighted number. Multi-objective mode optimizes the raw metrics jointly and
returns the Pareto front — no arbitrary weighting. These tests verify the
result contract for both modes and that objective parsing fails loudly.
"""

from __future__ import annotations

import pytest

from tests.conftest import make_oscillating_candles


def _candle_data():
    return {"BTC/USDT:USDT": make_oscillating_candles(n=4000, amplitude=900)}


def test_single_objective_mode_returns_scalar_best():
    pytest.importorskip("optuna")
    from combo_bot.optimize import Optimizer, OptimizeConfig

    cfg = OptimizeConfig(
        n_trials=6, n_jobs=1, sampler="random", walk_forward_splits=2
    )
    result = Optimizer(cfg, _candle_data()).run()
    # Legacy contract unchanged.
    assert "best_value" in result
    assert "best_params" in result
    assert "grid" in result and "trend" in result and "merger" in result
    assert "pareto_front" not in result


def test_multi_objective_mode_returns_pareto_front():
    pytest.importorskip("optuna")
    from combo_bot.optimize import Optimizer, OptimizeConfig

    cfg = OptimizeConfig(
        n_trials=10,
        n_jobs=1,
        sampler="nsga2",
        walk_forward_splits=2,
        objectives=["adg:max", "max_drawdown:min"],
    )
    result = Optimizer(cfg, _candle_data()).run()

    assert "pareto_front" in result
    assert result["objectives"] == ["adg:max", "max_drawdown:min"]
    front = result["pareto_front"]
    assert isinstance(front, list) and len(front) >= 1
    for sol in front:
        # Each Pareto solution carries its params, the two objective values,
        # and the reconstructed config blocks.
        assert set(sol["objectives"]) == {"adg", "max_drawdown"}
        assert "grid" in sol and "trend" in sol and "merger" in sol
        assert "wallet_exposure_limit" in sol["grid"]

    # Pareto front must be non-dominated: no solution beats another on BOTH
    # axes (higher adg AND lower drawdown).
    for i, a in enumerate(front):
        for j, b in enumerate(front):
            if i == j:
                continue
            a_adg, a_dd = a["objectives"]["adg"], a["objectives"]["max_drawdown"]
            b_adg, b_dd = b["objectives"]["adg"], b["objectives"]["max_drawdown"]
            dominates = (b_adg >= a_adg and b_dd <= a_dd) and (
                b_adg > a_adg or b_dd < a_dd
            )
            assert not dominates, "Pareto front contains a dominated solution"


def test_invalid_objective_metric_raises():
    pytest.importorskip("optuna")
    from combo_bot.optimize import Optimizer, OptimizeConfig

    cfg = OptimizeConfig(objectives=["not_a_metric:max"])
    with pytest.raises(ValueError, match="unknown optimization metric"):
        Optimizer(cfg, _candle_data())
