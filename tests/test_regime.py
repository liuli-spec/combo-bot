from __future__ import annotations


from combo_bot.data_provider import DataProvider
from combo_bot.regime import RegimeArbiter, RegimeArbiterConfig, read_strategy_signals
from combo_bot.types import Candle, Side, TradingMode, TrendRegime, TrendSignal


def _signal(direction: float, strength: float, regime: TrendRegime) -> TrendSignal:
    return TrendSignal(direction=direction, strength=strength, regime=regime)


class TestNeutralRegime:
    def test_neutral_keeps_defaults(self):
        a = RegimeArbiter()
        view = a.compute(_signal(0.0, 0.0, TrendRegime.NEUTRAL))
        assert view.long_mode == TradingMode.NORMAL
        assert view.short_mode == TradingMode.NORMAL
        assert view.trend_overlay is None
        assert view.close_aggressiveness == 1.0
        assert view.allow_grid_long is True
        assert view.allow_grid_short is True


class TestBullRegime:
    def test_bull_puts_short_in_tp_only(self):
        a = RegimeArbiter()
        view = a.compute(_signal(0.35, 0.35, TrendRegime.BULL))
        assert view.long_mode == TradingMode.NORMAL
        assert view.short_mode == TradingMode.TP_ONLY
        assert view.close_aggressiveness == 0.85
        assert view.trend_overlay is None

    def test_bear_puts_long_in_tp_only(self):
        a = RegimeArbiter()
        view = a.compute(_signal(-0.35, 0.35, TrendRegime.BEAR))
        assert view.long_mode == TradingMode.TP_ONLY
        assert view.short_mode == TradingMode.NORMAL


class TestStrongRegime:
    def test_strong_bull_panic_closes_short_by_default(self):
        a = RegimeArbiter()
        view = a.compute(_signal(0.7, 0.7, TrendRegime.STRONG_BULL))
        assert view.short_mode == TradingMode.PANIC
        assert view.allow_grid_short is False

    def test_strong_bull_can_opt_out_of_panic_close(self):
        a = RegimeArbiter(RegimeArbiterConfig(panic_close_opposite_on_strong=False))
        view = a.compute(_signal(0.7, 0.7, TrendRegime.STRONG_BULL))
        assert view.short_mode == TradingMode.TP_ONLY
        assert view.allow_grid_short is True

    def test_strong_bull_promotes_long_to_aggressive(self):
        a = RegimeArbiter()
        view = a.compute(_signal(0.7, 0.7, TrendRegime.STRONG_BULL))
        assert view.long_mode == TradingMode.AGGRESSIVE

    def test_strong_bull_below_aggressive_threshold_stays_normal(self):
        a = RegimeArbiter(RegimeArbiterConfig(aggressive_strength=0.9))
        view = a.compute(_signal(0.7, 0.7, TrendRegime.STRONG_BULL))
        assert view.long_mode == TradingMode.NORMAL

    def test_strong_bull_activates_long_overlay_when_conviction_high(self):
        a = RegimeArbiter()
        view = a.compute(_signal(0.7, 0.7, TrendRegime.STRONG_BULL))
        assert view.trend_overlay == Side.LONG
        assert view.trend_qty_scale > 0

    def test_strong_bear_mirrors_bull(self):
        a = RegimeArbiter()
        view = a.compute(_signal(-0.7, 0.7, TrendRegime.STRONG_BEAR))
        assert view.long_mode == TradingMode.PANIC
        assert view.short_mode == TradingMode.AGGRESSIVE
        assert view.trend_overlay == Side.SHORT

    def test_close_aggressiveness_tighter_in_strong(self):
        a = RegimeArbiter()
        bull = a.compute(_signal(0.35, 0.35, TrendRegime.BULL))
        strong = a.compute(_signal(0.7, 0.7, TrendRegime.STRONG_BULL))
        assert strong.close_aggressiveness < bull.close_aggressiveness


class TestFundingVeto:
    def test_positive_funding_vetoes_long_overlay(self):
        # funding > veto threshold means longs pay shorts → don't open long overlay.
        a = RegimeArbiter(RegimeArbiterConfig(funding_overlay_veto_pct=0.001))
        view = a.compute(
            _signal(0.7, 0.7, TrendRegime.STRONG_BULL),
            funding_rate=0.002,
        )
        assert view.trend_overlay is None
        assert any("funding" in r for r in view.veto_reasons)

    def test_negative_funding_vetoes_short_overlay(self):
        a = RegimeArbiter(RegimeArbiterConfig(funding_overlay_veto_pct=0.001))
        view = a.compute(
            _signal(-0.7, 0.7, TrendRegime.STRONG_BEAR),
            funding_rate=-0.002,
        )
        assert view.trend_overlay is None

    def test_neutral_funding_does_not_veto(self):
        a = RegimeArbiter()
        view = a.compute(
            _signal(0.7, 0.7, TrendRegime.STRONG_BULL),
            funding_rate=0.0001,
        )
        assert view.trend_overlay == Side.LONG


class TestStrategyOverrides:
    def test_strategy_exit_long_forces_tp_only(self):
        a = RegimeArbiter()
        view = a.compute(
            _signal(0.0, 0.0, TrendRegime.NEUTRAL),
            strategy_exit_long=True,
        )
        assert view.long_mode == TradingMode.TP_ONLY
        assert view.short_mode == TradingMode.NORMAL

    def test_strategy_exit_does_not_override_panic(self):
        a = RegimeArbiter()
        view = a.compute(
            _signal(-0.7, 0.7, TrendRegime.STRONG_BEAR),
            strategy_exit_long=True,
        )
        # PANIC was already set; exit signal shouldn't downgrade to TP_ONLY
        assert view.long_mode == TradingMode.PANIC

    def test_strategy_exit_long_cancels_long_overlay(self):
        a = RegimeArbiter()
        view = a.compute(
            _signal(0.7, 0.7, TrendRegime.STRONG_BULL),
            strategy_exit_long=True,
        )
        assert view.trend_overlay is None


class TestReadStrategySignals:
    def test_empty_provider_returns_all_false(self):
        dp = DataProvider()
        assert read_strategy_signals(dp, "BTC") == (False, False, False, False)

    def test_no_signal_columns_returns_all_false(self):
        dp = DataProvider()
        dp.append("BTC", Candle(1_000, 100, 101, 99, 100, 10))
        assert read_strategy_signals(dp, "BTC") == (False, False, False, False)

    def test_reads_enter_long_from_latest_row(self):

        dp = DataProvider()
        for ts in (1_000, 2_000, 3_000):
            dp.append("BTC", Candle(ts, 100, 101, 99, 100, 10))
        # Mutate the cached DataFrame as a strategy's populate_entry_trend would.
        df = dp.get_dataframe("BTC")
        df["enter_long"] = [0, 0, 1]
        df["exit_short"] = [0, 1, 0]
        # Note: the cached df is mutated in place, so subsequent reads see these.
        assert read_strategy_signals(dp, "BTC") == (True, False, False, False)

    def test_reads_exit_long_from_latest_row(self):
        dp = DataProvider()
        dp.append("BTC", Candle(1_000, 100, 101, 99, 100, 10))
        dp.append("BTC", Candle(2_000, 100, 101, 99, 100, 10))
        df = dp.get_dataframe("BTC")
        df["exit_long"] = [0, 1]
        _, _, exit_long, _ = read_strategy_signals(dp, "BTC")
        assert exit_long is True
