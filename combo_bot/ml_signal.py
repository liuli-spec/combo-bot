"""Machine-learning signal layer (triple-barrier supervised model).

A self-contained ML signal generator inspired by FreqAI's supervised
pipeline and López de Prado's triple-barrier labeling. It turns an OHLCV
window into a directional conviction score ``ml_score ∈ [-1, 1]`` that an
ML-driven overlay can act on.

Design priorities, in order:

1. **No look-ahead.** This is the single failure mode that makes a backtest
   look great and live lose money. Two guards:
     * Features at bar ``t`` use only data up to and including ``t``
       (causal rolling windows).
     * Triple-barrier labels at bar ``t`` look *forward* up to
       ``horizon_bars``, so the last ``horizon_bars`` rows have no complete
       label and are excluded from training. ``train()`` only fits on rows
       whose forward window is fully observed.
2. **Anti-overfit defaults.** Shallow trees, low learning rate, a minimum
   training-sample floor. Financial return signal-to-noise is tiny.
3. **Graceful degradation.** scikit-learn is an optional dependency; if it
   is missing, the model stays untrained and ``predict_score`` returns 0.0
   so the overlay simply never fires.

This module is pure model logic — it does NOT place orders. Wiring the
score into the decision pipeline (the ML overlay) is a separate layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import GradientBoostingClassifier

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    GradientBoostingClassifier = None  # type: ignore[assignment,misc]
    _SKLEARN_AVAILABLE = False


@dataclass
class MLSignalConfig:
    """Configuration for the ML signal model.

    The triple-barrier widths scale with realized volatility so the labels
    adapt to regime — a fixed % barrier would be all-stop-loss in calm
    markets and all-timeout in volatile ones.
    """

    enabled: bool = False
    # ── Triple-barrier labeling ──────────────────────────────────────
    horizon_bars: int = 24  # vertical (time) barrier: max look-forward
    pt_mult: float = 2.0  # profit-take barrier = pt_mult * vol
    sl_mult: float = 2.0  # stop-loss barrier = sl_mult * vol
    vol_window: int = 50  # realized-vol window setting barrier width
    # ── Training ─────────────────────────────────────────────────────
    train_window: int = 2000  # rolling training window (bars)
    retrain_interval: int = 500  # retrain every N bars
    min_train_samples: int = 200  # refuse to fit below this
    # ── Model (anti-overfit defaults) ────────────────────────────────
    n_estimators: int = 100
    max_depth: int = 3
    learning_rate: float = 0.05
    random_state: int = 42
    # ── Feature lags ─────────────────────────────────────────────────
    return_lags: tuple[int, ...] = (1, 2, 3, 5, 10)
    ema_spans: tuple[int, ...] = (10, 30)
    rsi_period: int = 14
    # ── Overlay action ───────────────────────────────────────────────
    # |ml_score| must exceed this before the ML overlay takes a side. The
    # conviction above the threshold maps linearly to the entry qty scale.
    score_threshold: float = 0.3


def ml_overlay_decision(score: float, threshold: float) -> tuple[str | None, float]:
    """Map a directional ml_score to an overlay (side, qty_scale).

    Returns ("long"/"short"/None, scale) where scale ∈ [0, 1] grows with
    conviction above the threshold. Side strings (not the Side enum) keep
    this module import-light; callers translate.
    """
    span = max(1.0 - threshold, 1e-9)
    if score > threshold:
        return "long", min(1.0, (score - threshold) / span)
    if score < -threshold:
        return "short", min(1.0, (-score - threshold) / span)
    return None, 0.0


def apply_ml_overlay(regime_view, score: float, threshold: float):
    """Replace the trend-overlay fields on a RegimeView from an ml_score.

    Shared by Backtester and LiveTrader so both drive the overlay
    identically. Returns a NEW RegimeView (the original is frozen).
    """
    from dataclasses import replace

    from combo_bot.types import Side

    side_str, scale = ml_overlay_decision(score, threshold)
    if side_str is None:
        return replace(regime_view, trend_overlay=None, trend_qty_scale=0.0)
    side = Side.LONG if side_str == "long" else Side.SHORT
    return replace(regime_view, trend_overlay=side, trend_qty_scale=scale)


# ---------------------------------------------------------------------------
# Causal feature engineering (each row uses only past+current data)
# ---------------------------------------------------------------------------
def _ema(values: np.ndarray, span: int) -> np.ndarray:
    """Causal EMA. out[t] depends only on values[:t+1]."""
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values, dtype=float)
    if len(values) == 0:
        return out
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling std (population). out[t] uses values[t-window+1:t+1]."""
    n = len(values)
    out = np.zeros(n, dtype=float)
    if n == 0:
        return out
    csum = np.cumsum(values)
    csum2 = np.cumsum(values * values)
    for t in range(n):
        lo = max(0, t - window + 1)
        cnt = t - lo + 1
        s = csum[t] - (csum[lo - 1] if lo > 0 else 0.0)
        s2 = csum2[t] - (csum2[lo - 1] if lo > 0 else 0.0)
        mean = s / cnt
        var = max(0.0, s2 / cnt - mean * mean)
        out[t] = var**0.5
    return out


def _rsi(closes: np.ndarray, period: int) -> np.ndarray:
    """Causal Wilder-style RSI in [0, 100]."""
    n = len(closes)
    out = np.full(n, 50.0, dtype=float)
    if n < 2:
        return out
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(1, n):
        g, ls = gains[i - 1], losses[i - 1]
        if i <= period:
            avg_gain += g / period
            avg_loss += ls / period
        else:
            avg_gain = (avg_gain * (period - 1) + g) / period
            avg_loss = (avg_loss * (period - 1) + ls) / period
        if avg_loss <= 1e-12:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out


def compute_features(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    config: MLSignalConfig,
) -> np.ndarray:
    """Build a causal (N, F) feature matrix from OHLCV arrays.

    Every column at row ``t`` is a function of data at indices ``<= t``
    only. NaN/inf are scrubbed to 0 so the model never sees garbage.
    """
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    eps = 1e-12
    log_close = np.log(np.maximum(closes, eps))
    log_ret = np.zeros(n)
    log_ret[1:] = np.diff(log_close)

    cols: list[np.ndarray] = []
    # Multi-lag returns (momentum at several horizons).
    for lag in config.return_lags:
        lagged = np.zeros(n)
        if lag < n:
            lagged[lag:] = log_close[lag:] - log_close[:-lag]
        cols.append(lagged)
    # EMA distance (price relative to trend).
    for span in config.ema_spans:
        ema = _ema(closes, span)
        cols.append(closes / np.maximum(ema, eps) - 1.0)
    # RSI centered to [-0.5, 0.5].
    cols.append(_rsi(closes, config.rsi_period) / 100.0 - 0.5)
    # Realized volatility of returns.
    cols.append(_rolling_std(log_ret, config.vol_window))
    # High-low range relative to close (intrabar range proxy).
    rng = (np.asarray(highs, dtype=float) - np.asarray(lows, dtype=float)) / np.maximum(
        closes, eps
    )
    cols.append(rng)
    # Volume z-score (causal).
    vol = np.asarray(volumes, dtype=float)
    vmean = _ema(vol, config.vol_window)
    vstd = _rolling_std(vol, config.vol_window)
    cols.append((vol - vmean) / np.maximum(vstd, eps))

    feats = np.column_stack(cols)
    feats[~np.isfinite(feats)] = 0.0
    return feats


# ---------------------------------------------------------------------------
# Triple-barrier labeling (López de Prado)
# ---------------------------------------------------------------------------
def triple_barrier_labels(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    config: MLSignalConfig,
) -> np.ndarray:
    """Label each bar by which barrier its forward path touches first.

    For bar ``t`` with realized vol ``v_t``:
        upper = close_t * (1 + pt_mult * v_t)   (profit-take)
        lower = close_t * (1 - sl_mult * v_t)   (stop-loss)
    Scan bars ``t+1 .. t+horizon``:
        +1  if a bar's high reaches ``upper`` before any low reaches ``lower``
        -1  if a bar's low reaches ``lower`` first
         0  if the time (vertical) barrier is hit first (neither touched)

    The last ``horizon`` rows look past the end of the data and get label 0
    AND are reported as having an incomplete forward window (see
    ``last_complete_index``) so training can exclude them.
    """
    closes = np.asarray(closes, dtype=float)
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    n = len(closes)
    labels = np.zeros(n, dtype=int)
    h = config.horizon_bars

    log_close = np.log(np.maximum(closes, 1e-12))
    log_ret = np.zeros(n)
    log_ret[1:] = np.diff(log_close)
    vol = _rolling_std(log_ret, config.vol_window)

    for t in range(n):
        end = min(t + h, n - 1)
        if end <= t:
            continue  # no forward window (tail) → leave as 0
        v = vol[t]
        if v <= 0:
            continue
        upper = closes[t] * (1.0 + config.pt_mult * v)
        lower = closes[t] * (1.0 - config.sl_mult * v)
        label = 0
        for j in range(t + 1, end + 1):
            hit_up = highs[j] >= upper
            hit_dn = lows[j] <= lower
            if hit_up and hit_dn:
                # Both barriers inside the same bar — ambiguous; treat as
                # stop-loss (conservative: assume the adverse touch first).
                label = -1
                break
            if hit_up:
                label = 1
                break
            if hit_dn:
                label = -1
                break
        labels[t] = label
    return labels


def last_complete_label_index(n: int, config: MLSignalConfig) -> int:
    """Highest row index whose full forward (horizon) window is observed.

    Rows beyond this have truncated triple-barrier windows and must be
    excluded from training to avoid leaking the (missing) future.
    """
    return n - config.horizon_bars - 1


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MLSignalModel:
    """Triple-barrier classifier producing a directional score.

    Lifecycle: ``maybe_retrain(...)`` on a schedule, then ``predict_score``
    each bar. Pure model logic — no order placement.
    """

    def __init__(self, config: MLSignalConfig | None = None) -> None:
        self.config = config or MLSignalConfig()
        self._model = None
        self._last_train_index: int = -(10**9)
        self._classes: np.ndarray | None = None

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def train(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
    ) -> bool:
        """Fit on the supplied window. Returns True on a successful fit.

        Only rows with a fully-observed forward label window are used
        (look-ahead guard). Refuses to fit below ``min_train_samples`` or
        when only one class is present.
        """
        if not _SKLEARN_AVAILABLE:
            logger.warning(
                "[ml] scikit-learn not installed — ML signal disabled. "
                "Install with: pip install scikit-learn"
            )
            return False
        n = len(closes)
        cutoff = last_complete_label_index(n, self.config)
        if cutoff < self.config.min_train_samples:
            return False

        feats = compute_features(closes, highs, lows, volumes, self.config)
        labels = triple_barrier_labels(closes, highs, lows, self.config)
        # Exclude tail rows whose forward window is incomplete.
        x = feats[: cutoff + 1]
        y = labels[: cutoff + 1]
        if len(np.unique(y)) < 2:
            return False  # degenerate (all one class) — nothing to learn

        model = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            random_state=self.config.random_state,
        )
        try:
            model.fit(x, y)
        except Exception:
            logger.exception("[ml] model fit failed")
            return False
        self._model = model
        self._classes = model.classes_
        return True

    def maybe_retrain(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        bar_index: int,
    ) -> bool:
        """Retrain if ``retrain_interval`` bars have elapsed. Uses the most
        recent ``train_window`` bars (plus the horizon tail that gets
        excluded). Returns True if a (re)train happened."""
        if bar_index - self._last_train_index < self.config.retrain_interval:
            return False
        window = self.config.train_window + self.config.horizon_bars
        lo = max(0, len(closes) - window)
        ok = self.train(closes[lo:], highs[lo:], lows[lo:], volumes[lo:])
        if ok:
            self._last_train_index = bar_index
        return ok

    def predict_score(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
    ) -> float:
        """Directional conviction for the LATEST bar: ``P(+1) - P(-1)``.

        Uses only the most recent feature row (causal). Returns 0.0 when
        untrained so an ML overlay simply stays flat.
        """
        if self._model is None or self._classes is None:
            return 0.0
        feats = compute_features(closes, highs, lows, volumes, self.config)
        if len(feats) == 0:
            return 0.0
        try:
            proba = self._model.predict_proba(feats[-1:])[0]
        except Exception:
            logger.exception("[ml] predict failed")
            return 0.0
        p_up = 0.0
        p_dn = 0.0
        for cls, p in zip(self._classes, proba):
            if cls == 1:
                p_up = float(p)
            elif cls == -1:
                p_dn = float(p)
        return p_up - p_dn
