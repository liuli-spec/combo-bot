from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from combo_bot.types import TrendRegime, TrendSignal


@dataclass
class TrendConfig:
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    ema_fast: int = 20
    ema_slow: int = 50
    atr_period: int = 14
    strong_threshold: float = 0.6
    weak_threshold: float = 0.25


class TrendEngine:
    def __init__(self, config: TrendConfig | None = None):
        self.config = config or TrendConfig()
        self._history: dict[str, list[float]] = {}

    def update(self, symbol: str, close: float):
        if symbol not in self._history:
            self._history[symbol] = []
        self._history[symbol].append(close)
        max_needed = (
            max(
                self.config.macd_slow + self.config.macd_signal,
                self.config.bb_period,
                self.config.ema_slow,
                self.config.atr_period + 1,
            )
            + 10
        )
        # Prune infrequently to avoid EMA re-seed jumps (each truncation
        # resets the seeding index of _ema(), producing a signal artefact).
        # 20× leaves ~920 bars (≈15 hours of 1m data) between truncations
        # for typical configs — the jump becomes biologically irrelevant.
        if len(self._history[symbol]) > max_needed * 20:
            self._history[symbol] = self._history[symbol][-max_needed:]

    def compute(self, symbol: str) -> TrendSignal:
        prices = self._history.get(symbol, [])
        if len(prices) < self.config.ema_slow + 5:
            return TrendSignal(direction=0.0, strength=0.0, regime=TrendRegime.NEUTRAL)

        arr = np.array(prices, dtype=np.float64)
        scores = []

        rsi = _calc_rsi(arr, self.config.rsi_period)
        if np.isfinite(rsi):
            scores.append((rsi - 50.0) / 50.0)

        macd_hist = _calc_macd_histogram(
            arr, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal
        )
        if np.isfinite(macd_hist) and arr[-1] > 0:
            scores.append(np.clip(macd_hist / (arr[-1] * 0.01), -1.0, 1.0))

        bb_pos = _calc_bb_position(arr, self.config.bb_period, self.config.bb_std)
        if np.isfinite(bb_pos):
            scores.append(np.clip(bb_pos, -1.0, 1.0))

        ema_score = _calc_ema_trend(arr, self.config.ema_fast, self.config.ema_slow)
        if np.isfinite(ema_score):
            scores.append(np.clip(ema_score, -1.0, 1.0))

        if not scores:
            return TrendSignal(direction=0.0, strength=0.0, regime=TrendRegime.NEUTRAL)

        direction = float(np.mean(scores))
        strength = float(min(abs(direction), 1.0))
        direction = float(np.clip(direction, -1.0, 1.0))

        regime = _classify_regime(
            direction, self.config.strong_threshold, self.config.weak_threshold
        )
        return TrendSignal(direction=direction, strength=strength, regime=regime)

    def reset(self, symbol: str | None = None):
        if symbol:
            self._history.pop(symbol, None)
        else:
            self._history.clear()


def _calc_rsi(prices: np.ndarray, period: int) -> float:
    """Wilder's RSI — matches talib.RSI (freqtrade canonical).

    Seeds avg gain/loss with a simple mean over the first `period` deltas,
    then applies Wilder's smoothing (alpha = 1/period) for the remainder.
    Equivalent to a recursive EMA with the Wilder convention.
    """
    if len(prices) < period + 1:
        return np.nan
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    inv_p = 1.0 / period
    keep = 1.0 - inv_p
    for i in range(period, len(deltas)):
        avg_gain = avg_gain * keep + gains[i] * inv_p
        avg_loss = avg_loss * keep + losses[i] * inv_p

    if avg_loss < 1e-12:
        return 100.0 if avg_gain > 1e-12 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _calc_macd_histogram(
    prices: np.ndarray, fast: int, slow: int, signal: int
) -> float:
    if len(prices) < slow + signal:
        return np.nan
    ema_fast = _ema(prices, fast)
    ema_slow = _ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    return float(macd_line[-1] - signal_line[-1])


def _calc_bb_position(prices: np.ndarray, period: int, num_std: float) -> float:
    if len(prices) < period:
        return np.nan
    window = prices[-period:]
    mid = np.mean(window)
    std = np.std(window)
    if std < 1e-12:
        return 0.0
    return (prices[-1] - mid) / (num_std * std)


def _calc_ema_trend(prices: np.ndarray, fast: int, slow: int) -> float:
    if len(prices) < slow + 1:
        return np.nan
    ema_f = _ema(prices, fast)
    ema_s = _ema(prices, slow)
    diff = ema_f[-1] - ema_s[-1]
    mid = (ema_f[-1] + ema_s[-1]) / 2.0
    if mid < 1e-12:
        return 0.0
    return diff / mid * 20.0


def _classify_regime(direction: float, strong: float, weak: float) -> TrendRegime:
    if direction > strong:
        return TrendRegime.STRONG_BULL
    if direction > weak:
        return TrendRegime.BULL
    if direction < -strong:
        return TrendRegime.STRONG_BEAR
    if direction < -weak:
        return TrendRegime.BEAR
    return TrendRegime.NEUTRAL
