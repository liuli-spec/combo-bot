from __future__ import annotations
import numpy as np
import pytest
from combo_bot.trend_signal import TrendConfig, TrendEngine, _calc_rsi
from combo_bot.types import TrendRegime


class TestRSI:
    def test_all_gains_returns_100(self):
        prices = np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                16.0,
            ]
        )
        assert _calc_rsi(prices, 14) == 100.0

    def test_all_losses_returns_near_0(self):
        prices = np.array(
            [
                16.0,
                15.0,
                14.0,
                13.0,
                12.0,
                11.0,
                10.0,
                9.0,
                8.0,
                7.0,
                6.0,
                5.0,
                4.0,
                3.0,
                2.0,
                1.0,
            ]
        )
        assert _calc_rsi(prices, 14) < 1.0

    def test_flat_returns_50(self):
        prices = np.array([50.0] * 20)
        assert _calc_rsi(prices, 14) == 50.0

    def test_insufficient_data_returns_nan(self):
        prices = np.array([1.0, 2.0])
        assert np.isnan(_calc_rsi(prices, 14))

    def test_matches_pandas_wilder_smoothing(self):
        """Wilder's RSI must agree with pandas ewm(alpha=1/period) seeded by SMA.

        This is the same formula talib.RSI uses internally; matching pandas
        here is how we keep our composite TrendEngine RSI consistent with
        IStrategy.populate_indicators implementations that compute RSI in
        pandas (the freqtrade convention).
        """
        import pandas as pd

        rng = np.random.default_rng(7)
        prices = 100.0 + np.cumsum(rng.normal(0, 0.5, 200))

        period = 14
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        seed_gain = float(np.mean(gains[:period]))
        seed_loss = float(np.mean(losses[:period]))

        # Reference: apply Wilder's smoothing iteratively in pure pandas style.
        gain_series = pd.Series([seed_gain] + list(gains[period:]))
        loss_series = pd.Series([seed_loss] + list(losses[period:]))
        ref_gain = gain_series.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
        ref_loss = loss_series.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
        ref_rsi = 100.0 - 100.0 / (1.0 + ref_gain / ref_loss)

        assert _calc_rsi(prices, period) == pytest.approx(ref_rsi, rel=1e-10)


class TestTrendEngine:
    def test_neutral_on_flat_data(self):
        engine = TrendEngine(TrendConfig())
        for i in range(100):
            engine.update("BTC", 50000.0 + np.random.default_rng(42).normal(0, 1))
        signal = engine.compute("BTC")
        assert signal.regime == TrendRegime.NEUTRAL

    def test_bullish_on_rising_data(self):
        engine = TrendEngine(TrendConfig(strong_threshold=0.4))
        for i in range(200):
            engine.update("BTC", 50000.0 + i * 50.0)
        signal = engine.compute("BTC")
        assert signal.direction > 0
        assert signal.regime in (TrendRegime.BULL, TrendRegime.STRONG_BULL)

    def test_bearish_on_falling_data(self):
        engine = TrendEngine(TrendConfig(strong_threshold=0.4))
        for i in range(200):
            engine.update("BTC", 60000.0 - i * 50.0)
        signal = engine.compute("BTC")
        assert signal.direction < 0
        assert signal.regime in (TrendRegime.BEAR, TrendRegime.STRONG_BEAR)

    def test_insufficient_data_returns_neutral(self):
        engine = TrendEngine(TrendConfig())
        engine.update("BTC", 50000.0)
        signal = engine.compute("BTC")
        assert signal.regime == TrendRegime.NEUTRAL
        assert signal.strength == 0.0

    def test_reset_clears_history(self):
        engine = TrendEngine(TrendConfig())
        for i in range(100):
            engine.update("BTC", 50000.0 + i)
        engine.reset("BTC")
        signal = engine.compute("BTC")
        assert signal.strength == 0.0

    def test_multiple_symbols_independent(self):
        engine = TrendEngine(TrendConfig(strong_threshold=0.4))
        for i in range(200):
            engine.update("BTC", 50000.0 + i * 50.0)
            engine.update("ETH", 3000.0 - i * 20.0)
        btc = engine.compute("BTC")
        eth = engine.compute("ETH")
        assert btc.direction > 0
        assert eth.direction < 0
