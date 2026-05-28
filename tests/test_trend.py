from __future__ import annotations
import numpy as np
import pytest
from combo_bot.trend_signal import TrendConfig, TrendEngine, _calc_rsi, _calc_macd_histogram
from combo_bot.types import TrendRegime


class TestRSI:
    def test_all_gains_returns_100(self):
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
        assert _calc_rsi(prices, 14) == 100.0

    def test_all_losses_returns_near_0(self):
        prices = np.array([16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0,
                           8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        assert _calc_rsi(prices, 14) < 1.0

    def test_flat_returns_50(self):
        prices = np.array([50.0] * 20)
        assert _calc_rsi(prices, 14) == 50.0

    def test_insufficient_data_returns_nan(self):
        prices = np.array([1.0, 2.0])
        assert np.isnan(_calc_rsi(prices, 14))


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
