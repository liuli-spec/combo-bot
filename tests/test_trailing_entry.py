"""Stage 8 passivbot-style trailing re-entry tests.

Covers:

  * :class:`TrailingState` bundle semantics - reset on new extreme,
    track recovery from extreme.
  * :meth:`GridEngine.compute_trailing_entry` - fires only when both
    stages trigger; long/short symmetry; disabled when thresholds 0.
  * WEL ceiling; mode gating; cost crop.
  * Backtester integration - bundle resets on fresh grid open, bundle
    updates fed from candle high/low.
"""

from __future__ import annotations

import pytest

from combo_bot.backtest import Backtester, BacktestConfig
from combo_bot.grid_engine import GridConfig, GridEngine
from combo_bot.strategy import DefaultStrategy
from combo_bot.types import (
    Candle,
    EMAState,
    ExchangeParams,
    Position,
    Side,
    TradingMode,
    TrailingState,
    VolatilityState,
)


def _ema(lower=49_000.0, upper=49_500.0) -> EMAState:
    return EMAState(
        spans=[100.0, 200.0],
        values=[lower, upper],
        alphas=[0.02, 0.01],
        initialized=True,
    )


def _ep(**overrides) -> ExchangeParams:
    return ExchangeParams(**overrides)


# ---------------------------------------------------------------------------
# TrailingState bundle semantics
# ---------------------------------------------------------------------------


class TestTrailingStateLong:
    def test_reset_seeds_extreme_and_recovery(self):
        s = TrailingState()
        s.reset(100.0)
        assert s.extreme == 100.0
        assert s.recovery == 100.0
        assert s.initialized is True

    def test_uninitialized_update_is_noop(self):
        s = TrailingState()
        s.update_long(110.0, 90.0)
        assert s.initialized is False
        assert s.extreme == 0.0
        assert s.recovery == 0.0

    def test_long_tracks_min_extreme(self):
        s = TrailingState()
        s.reset(100.0)
        s.update_long(105.0, 95.0)
        assert s.extreme == 95.0  # new low
        # recovery resets to the new low; bounce is measured from here.
        assert s.recovery == 95.0

    def test_long_recovery_climbs_after_new_low(self):
        s = TrailingState()
        s.reset(100.0)
        s.update_long(105.0, 95.0)
        s.update_long(98.0, 96.0)
        # No new low; recovery climbs to this candle's high (98).
        assert s.extreme == 95.0
        assert s.recovery == 98.0

    def test_long_new_low_resets_recovery(self):
        s = TrailingState()
        s.reset(100.0)
        s.update_long(101.0, 95.0)
        s.update_long(99.0, 97.0)  # recovery becomes 99
        assert s.recovery == 99.0
        s.update_long(96.0, 90.0)  # new low -> reset
        assert s.extreme == 90.0
        assert s.recovery == 90.0  # reset


class TestTrailingStateShort:
    def test_short_tracks_max_extreme(self):
        s = TrailingState()
        s.reset(100.0)
        s.update_short(105.0, 95.0)
        assert s.extreme == 105.0  # new high
        assert s.recovery == 105.0

    def test_short_recovery_falls_after_new_high(self):
        s = TrailingState()
        s.reset(100.0)
        s.update_short(110.0, 95.0)
        s.update_short(108.0, 100.0)
        # No new high; recovery (running min after high) falls to 100.
        assert s.extreme == 110.0
        assert s.recovery == 100.0


# ---------------------------------------------------------------------------
# GridEngine.compute_trailing_entry
# ---------------------------------------------------------------------------


def _engine(**overrides) -> GridEngine:
    defaults = dict(
        entry_trailing_threshold_pct=0.02,
        entry_trailing_retracement_pct=0.005,
        entry_trailing_double_down_factor=1.3,
        wallet_exposure_limit=5.0,
    )
    defaults.update(overrides)
    return GridEngine(GridConfig(**defaults))


class TestTrailingEntryLong:
    def test_disabled_returns_none_when_threshold_zero(self):
        engine = _engine(entry_trailing_threshold_pct=0.0)
        pos = Position(0.1, 50_000.0)
        trail = TrailingState()
        trail.reset(50_000.0)
        trail.update_long(50_000.0, 48_500.0)
        result = engine.compute_trailing_entry(
            "BTC", Side.LONG, pos, trail,
            balance=10_000.0, wallet_exposure=0.5,
            exchange_params=_ep(), mark_price=48_900.0,
        )
        assert result is None

    def test_no_entry_when_position_empty(self):
        engine = _engine()
        result = engine.compute_trailing_entry(
            "BTC", Side.LONG, Position(), TrailingState(),
            balance=10_000.0, wallet_exposure=0.0,
            exchange_params=_ep(), mark_price=50_000.0,
        )
        assert result is None

    def test_no_entry_when_threshold_not_met(self):
        engine = _engine()  # threshold 2%
        pos = Position(0.1, 50_000.0)
        trail = TrailingState()
        trail.reset(50_000.0)
        # Price dropped only 1% to 49_500, below the 2% threshold.
        trail.update_long(50_000.0, 49_500.0)
        # Bounce to 49_900 (would satisfy retracement)
        trail.update_long(49_900.0, 49_500.0)
        result = engine.compute_trailing_entry(
            "BTC", Side.LONG, pos, trail,
            balance=10_000.0, wallet_exposure=0.5,
            exchange_params=_ep(), mark_price=49_700.0,
        )
        assert result is None

    def test_no_entry_when_no_bounce(self):
        engine = _engine()
        pos = Position(0.1, 50_000.0)
        trail = TrailingState()
        trail.reset(50_000.0)
        trail.update_long(50_000.0, 48_500.0)  # 3% drop
        # No bounce; recovery still at extreme.
        result = engine.compute_trailing_entry(
            "BTC", Side.LONG, pos, trail,
            balance=10_000.0, wallet_exposure=0.5,
            exchange_params=_ep(), mark_price=48_500.0,
        )
        assert result is None

    def test_entry_fires_when_both_conditions_met(self):
        engine = _engine()  # threshold 2%, retracement 0.5%
        pos = Position(0.1, 50_000.0)
        trail = TrailingState()
        trail.reset(50_000.0)
        # Drop 3% to 48_500.
        trail.update_long(50_000.0, 48_500.0)
        # Bounce 1% to 48_985, past the 0.5% retracement threshold.
        trail.update_long(48_985.0, 48_500.0)
        result = engine.compute_trailing_entry(
            "BTC", Side.LONG, pos, trail,
            balance=10_000.0, wallet_exposure=0.5,
            exchange_params=_ep(), mark_price=48_985.0,
        )
        assert result is not None
        assert result.side == Side.LONG
        # qty = pos.size * DDF (1.3) at minimum
        assert result.qty >= 0.13 - 1e-9
        # Order price = min(mark_price, entry*(1-threshold+retracement)).
        # raw = 50000 * 0.985 = 49250; mark = 48985 -> use 48985 so the
        # limit doesn't fill instantly against the current ask.
        assert result.price == pytest.approx(48_985.0, abs=1.0)

    def test_panic_mode_blocks_trailing_entry(self):
        engine = _engine()
        pos = Position(0.1, 50_000.0)
        trail = TrailingState()
        trail.reset(50_000.0)
        trail.update_long(50_000.0, 48_500.0)
        trail.update_long(48_985.0, 48_500.0)
        result = engine.compute_trailing_entry(
            "BTC", Side.LONG, pos, trail,
            balance=10_000.0, wallet_exposure=0.5,
            exchange_params=_ep(), mark_price=48_985.0,
            mode=TradingMode.PANIC,
        )
        assert result is None

    def test_wel_ceiling_blocks_entry(self):
        engine = _engine(wallet_exposure_limit=0.6)
        pos = Position(0.12, 50_000.0)  # WE = 0.6
        trail = TrailingState()
        trail.reset(50_000.0)
        trail.update_long(50_000.0, 48_500.0)
        trail.update_long(48_985.0, 48_500.0)
        result = engine.compute_trailing_entry(
            "BTC", Side.LONG, pos, trail,
            balance=10_000.0, wallet_exposure=0.6,
            exchange_params=_ep(), mark_price=48_985.0,
        )
        assert result is None


class TestTrailingEntryShort:
    def test_short_fires_on_high_then_retracement(self):
        engine = _engine()
        pos = Position(0.1, 50_000.0)
        trail = TrailingState()
        trail.reset(50_000.0)
        # Spike up 3% to 51_500.
        trail.update_short(51_500.0, 50_000.0)
        # Retrace down 1% to 50_985, past the 0.5% retracement.
        trail.update_short(51_500.0, 50_985.0)
        result = engine.compute_trailing_entry(
            "BTC", Side.SHORT, pos, trail,
            balance=10_000.0, wallet_exposure=0.5,
            exchange_params=_ep(), mark_price=50_985.0,
        )
        assert result is not None
        assert result.side == Side.SHORT
        # Order price = max(mark, entry*(1+threshold-retracement)).
        # raw = 50000 * 1.015 = 50750; mark = 50985 -> use 50985 so the
        # short limit doesn't fill instantly against the bid.
        assert result.price == pytest.approx(50_985.0, abs=1.0)


# ---------------------------------------------------------------------------
# Backtester integration
# ---------------------------------------------------------------------------


class TestBacktesterTrailingIntegration:
    def test_trailing_entry_produces_an_additional_fill_when_enabled(self):
        """A V-shaped price series after a grid entry should trigger a
        passivbot trailing re-entry that wouldn't fire in disabled mode."""
        from tests.conftest import make_candles

        # Phase 1: drop from 50000 to 48500 (3% below initial entry)
        # Phase 2: bounce from 48500 back to 49000 (1% retracement)
        # Phase 3: keep climbing.
        prices = [50_000 - i * 50 for i in range(30)]   # 50000 -> 48550
        prices += [48_500 + i * 50 for i in range(30)]  # 48500 -> 49950
        candles = make_candles(prices)

        base_cfg = dict(
            entry_initial_ema_dist=0.001,
            entry_grid_spacing_pct=0.05,  # wide -> grid only takes initial
            wallet_exposure_limit=0.5,
            max_grid_levels=1,
        )

        cfg_off = BacktestConfig(
            starting_balance=10_000, symbols=["BTC"],
            grid=GridConfig(
                **base_cfg,
                entry_trailing_threshold_pct=0.0,
                entry_trailing_retracement_pct=0.0,
            ),
        )
        cfg_on = BacktestConfig(
            starting_balance=10_000, symbols=["BTC"],
            grid=GridConfig(
                **base_cfg,
                entry_trailing_threshold_pct=0.02,
                entry_trailing_retracement_pct=0.005,
                entry_trailing_double_down_factor=1.3,
            ),
        )

        result_off = Backtester(cfg_off).run({"BTC": candles})
        result_on = Backtester(cfg_on).run({"BTC": candles})

        # Trailing on should produce strictly more fills than off; the
        # extra trailing re-entry fires during the bounce phase.
        assert result_on.n_trades > result_off.n_trades
        assert any(fill.qty > 0.001 for fill in result_on.fills)

    def test_trailing_entry_respects_strategy_entry_veto(self):
        from tests.conftest import make_candles

        class RejectLargeEntries(DefaultStrategy):
            def confirm_trade_entry(self, ctx, proposed_qty, proposed_price):
                return proposed_qty <= 0.001 + 1e-12

        prices = [50_000 - i * 50 for i in range(30)]
        prices += [48_500 + i * 50 for i in range(30)]
        candles = make_candles(prices)

        cfg = BacktestConfig(
            starting_balance=10_000,
            symbols=["BTC"],
            grid=GridConfig(
                entry_initial_ema_dist=0.001,
                entry_grid_spacing_pct=0.05,
                wallet_exposure_limit=0.5,
                max_grid_levels=1,
                entry_trailing_threshold_pct=0.02,
                entry_trailing_retracement_pct=0.005,
                entry_trailing_double_down_factor=1.3,
            ),
        )

        result = Backtester(cfg, strategy=RejectLargeEntries()).run({"BTC": candles})

        assert all(fill.qty <= 0.001 + 1e-12 for fill in result.fills)
