"""Tests for the ML signal layer.

The two things that make or break a financial ML signal are tested
explicitly: triple-barrier label correctness, and the no-look-ahead
guard (features causal, tail rows excluded from training). Model
fit/predict tests skip cleanly when scikit-learn isn't installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from combo_bot.ml_signal import (
    MLSignalConfig,
    MLSignalModel,
    compute_features,
    last_complete_label_index,
    triple_barrier_labels,
)


# ── triple-barrier labeling ──────────────────────────────────────────


def test_triple_barrier_profit_take_label():
    """A bar followed by a clear upward break (high crosses the PT barrier
    before any low crosses the SL barrier) is labeled +1."""
    cfg = MLSignalConfig(horizon_bars=5, pt_mult=2.0, sl_mult=2.0, vol_window=5)
    # Flat-ish history to seed nonzero vol, then a sharp rally.
    closes = np.array([100, 101, 99, 100, 101, 100, 100, 100, 100, 100], dtype=float)
    highs = closes + 1.0
    lows = closes - 1.0
    # At t=4, push the next bars' highs way up (rally) and keep lows benign.
    highs[5:] = closes[5:] + 50.0
    labels = triple_barrier_labels(closes, highs, lows, cfg)
    assert labels[4] == 1


def test_triple_barrier_stop_loss_label():
    """A bar followed by a downward break is labeled -1."""
    cfg = MLSignalConfig(horizon_bars=5, pt_mult=2.0, sl_mult=2.0, vol_window=5)
    closes = np.array([100, 99, 101, 100, 99, 100, 100, 100, 100, 100], dtype=float)
    highs = closes + 1.0
    lows = closes - 1.0
    lows[5:] = closes[5:] - 50.0  # crash → low crosses SL barrier
    highs[5:] = closes[5:] + 0.1
    labels = triple_barrier_labels(closes, highs, lows, cfg)
    assert labels[4] == -1


def test_triple_barrier_timeout_label():
    """When neither barrier is touched within the horizon, label is 0."""
    cfg = MLSignalConfig(horizon_bars=5, pt_mult=5.0, sl_mult=5.0, vol_window=5)
    closes = np.array([100, 101, 99, 100, 101, 100, 100, 101, 99, 100], dtype=float)
    highs = closes + 0.2  # tiny ranges, wide barriers → never touched
    lows = closes - 0.2
    labels = triple_barrier_labels(closes, highs, lows, cfg)
    assert labels[3] == 0


def test_tail_rows_have_no_forward_window():
    """The last `horizon` rows can't have a complete label window — they
    must be excluded from training (look-ahead guard)."""
    cfg = MLSignalConfig(horizon_bars=10)
    n = 100
    assert last_complete_label_index(n, cfg) == 100 - 10 - 1  # 89


# ── causal features (no look-ahead) ──────────────────────────────────


def test_features_are_causal():
    """Changing a FUTURE bar must not change an earlier feature row —
    the defining property of a leak-free feature matrix."""
    cfg = MLSignalConfig()
    rng = np.random.default_rng(0)
    closes = 100 + np.cumsum(rng.normal(0, 1, 300))
    highs = closes + 1.0
    lows = closes - 1.0
    vols = np.abs(rng.normal(100, 10, 300))

    feats_a = compute_features(closes, highs, lows, vols, cfg)
    # Mutate only the LAST bar.
    closes2 = closes.copy()
    closes2[-1] += 50.0
    feats_b = compute_features(closes2, highs.copy(), lows.copy(), vols, cfg)

    # All rows except the last must be identical.
    assert np.allclose(feats_a[:-1], feats_b[:-1])


def test_features_finite_and_shaped():
    cfg = MLSignalConfig()
    n = 120
    closes = np.linspace(100, 110, n)
    feats = compute_features(closes, closes + 1, closes - 1, np.ones(n) * 100, cfg)
    assert feats.shape[0] == n
    assert np.all(np.isfinite(feats))


# ── model train / predict ────────────────────────────────────────────


def _trending_ohlcv(n: int = 1500, seed: int = 1):
    """Synthetic series with regime-dependent drift so there's learnable
    structure (not pure noise) for the smoke test."""
    rng = np.random.default_rng(seed)
    drift = np.where((np.arange(n) // 100) % 2 == 0, 0.001, -0.001)
    rets = drift + rng.normal(0, 0.004, n)
    closes = 100 * np.exp(np.cumsum(rets))
    highs = closes * (1 + np.abs(rng.normal(0, 0.003, n)))
    lows = closes * (1 - np.abs(rng.normal(0, 0.003, n)))
    vols = np.abs(rng.normal(1000, 100, n))
    return closes, highs, lows, vols


def test_model_untrained_returns_zero():
    """Untrained model (or no sklearn) yields a flat 0 score so the overlay
    never fires accidentally."""
    cfg = MLSignalConfig(horizon_bars=12)
    model = MLSignalModel(cfg)
    closes, highs, lows, vols = _trending_ohlcv(200)
    assert model.is_trained is False
    assert model.predict_score(closes, highs, lows, vols) == 0.0


def test_model_trains_and_scores_in_range():
    pytest.importorskip("sklearn")
    cfg = MLSignalConfig(
        horizon_bars=12,
        vol_window=20,
        min_train_samples=100,
        n_estimators=30,
        max_depth=2,
    )
    model = MLSignalModel(cfg)
    closes, highs, lows, vols = _trending_ohlcv(1500)
    assert model.train(closes, highs, lows, vols) is True
    assert model.is_trained is True
    score = model.predict_score(closes, highs, lows, vols)
    assert -1.0 <= score <= 1.0


def test_maybe_retrain_respects_interval():
    pytest.importorskip("sklearn")
    cfg = MLSignalConfig(
        horizon_bars=12,
        vol_window=20,
        min_train_samples=100,
        retrain_interval=500,
        train_window=1000,
        n_estimators=20,
        max_depth=2,
    )
    model = MLSignalModel(cfg)
    closes, highs, lows, vols = _trending_ohlcv(1500)
    assert model.maybe_retrain(closes, highs, lows, vols, bar_index=1000) is True
    # Too soon — within retrain_interval of the last train.
    assert model.maybe_retrain(closes, highs, lows, vols, bar_index=1200) is False
    # Far enough out — retrains again.
    assert model.maybe_retrain(closes, highs, lows, vols, bar_index=1600) is True
