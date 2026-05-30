"""Stage 6 hard-stop tests: EMA-smoothed drawdown + RED latch.

Mirrors passivbot's ``equity_hard_stop_loss.rs`` semantics:

  * the tier classifier consumes ``score = min(raw, ema)`` so a single
    flash spike on raw doesn't latch RED while EMA is still calm, and a
    stale EMA after recovery doesn't keep us pinned to RED;
  * once RED is reached, the tier latches RED until ``reset_red_latch``
    is called — preventing the bot from re-entering during the bleed
    just because a cooldown timer ticked over.
"""

from __future__ import annotations


from combo_bot.risk import RiskConfig, RiskManager, RiskTier
from combo_bot.types import AccountState, Order, OrderSource, Side, SymbolState


def _acc(
    equity: float, peak: float = 10_000.0, balance: float = 10_000.0
) -> AccountState:
    return AccountState(balance=balance, equity=equity, equity_peak=peak)


def _risk(**overrides) -> RiskManager:
    defaults = dict(
        yellow_threshold=0.10,
        orange_threshold=0.18,
        red_threshold=0.25,
        dd_ema_span_minutes=30.0,
        red_latch_enabled=True,
    )
    defaults.update(overrides)
    return RiskManager(RiskConfig(**defaults))


# ---------------------------------------------------------------------------
# First-call backward compat
# ---------------------------------------------------------------------------


class TestFirstCallBehavior:
    def test_first_call_uses_raw_dd(self):
        """A single assess() must produce a deterministic tier without
        needing a tick history — keeps single-shot tests sane."""
        risk = _risk()
        tier = risk.assess(_acc(equity=7_000.0), timestamp_ms=0)
        # raw dd = 0.3 >= red_threshold 0.25 -> RED
        assert tier == RiskTier.RED

    def test_smoothing_disabled_uses_raw(self):
        risk = _risk(dd_ema_span_minutes=0.0)
        tier = risk.assess(_acc(equity=7_500.0), timestamp_ms=60_000)
        # raw dd = 0.25
        assert tier == RiskTier.RED


# ---------------------------------------------------------------------------
# EMA smoothing prevents flash-wick RED
# ---------------------------------------------------------------------------


class TestEMASmoothing:
    def test_single_flash_spike_does_not_latch_red(self):
        """Account is healthy for 30 ticks, then a single bar prints a
        30% drawdown, then immediately recovers. EMA should keep us out
        of RED."""
        risk = _risk(red_latch_enabled=True, dd_ema_span_minutes=30.0)
        # Warm up the EMA at low drawdown.
        for minute in range(30):
            risk.assess(_acc(equity=9_950.0), timestamp_ms=minute * 60_000)
        # Now a flash spike at minute 30.
        spike_tier = risk.assess(_acc(equity=7_000.0), timestamp_ms=30 * 60_000)
        # score = min(raw=0.30, ema≈0.005) ≈ 0.005 → Green
        assert spike_tier == RiskTier.GREEN
        assert risk.red_latched is False

    def test_sustained_drawdown_eventually_triggers_red(self):
        """If the drawdown persists, EMA catches up and RED triggers."""
        risk = _risk(dd_ema_span_minutes=10.0)
        tier = None
        # Hold equity at 30% drawdown for 60 minutes — well past the EMA
        # span so ema converges close to raw.
        for minute in range(60):
            tier = risk.assess(_acc(equity=7_000.0), timestamp_ms=minute * 60_000)
        assert tier == RiskTier.RED
        assert risk.red_latched is True

    def test_recovery_after_long_drawdown_does_not_falsely_red(self):
        """When EMA is high from past drawdown but raw has fully
        recovered, min(raw, ema) keeps us out of RED. (Latch off so we
        can observe the underlying classifier.)"""
        risk = _risk(red_latch_enabled=False, dd_ema_span_minutes=10.0)
        # Build up EMA at high drawdown (but below red so we don't latch).
        for minute in range(60):
            risk.assess(_acc(equity=8_300.0), timestamp_ms=minute * 60_000)
        assert risk.dd_ema > 0.15  # EMA has converged near 0.17
        # Recover fully.
        tier = risk.assess(_acc(equity=10_000.0), timestamp_ms=120 * 60_000)
        # raw = 0, ema is still high — but score = min(0, high) = 0 → Green
        assert tier == RiskTier.GREEN


# ---------------------------------------------------------------------------
# RED latch persistence
# ---------------------------------------------------------------------------


class TestRedLatch:
    def test_red_latch_persists_after_recovery(self):
        risk = _risk(red_latch_enabled=True, dd_ema_span_minutes=5.0)
        # Sustained drawdown → RED + latch.
        for minute in range(30):
            risk.assess(_acc(equity=7_000.0), timestamp_ms=minute * 60_000)
        assert risk.red_latched is True

        # Full recovery → tier still RED.
        tier = risk.assess(_acc(equity=10_000.0), timestamp_ms=120 * 60_000)
        assert tier == RiskTier.RED

    def test_reset_red_latch_clears_state(self):
        risk = _risk()
        risk.red_latched = True
        risk.tier = RiskTier.RED
        risk.reset_red_latch()
        assert risk.red_latched is False
        assert risk.tier == RiskTier.GREEN

    def test_after_reset_tier_re_evaluates_from_current_dd(self):
        risk = _risk(red_latch_enabled=True)
        # First latch on.
        risk.assess(_acc(equity=7_000.0), timestamp_ms=0)
        assert risk.red_latched is True
        # Reset.
        risk.reset_red_latch()
        # Re-assess at healthy state.
        tier = risk.assess(_acc(equity=10_000.0), timestamp_ms=60_000)
        assert tier == RiskTier.GREEN

    def test_latch_disabled_allows_recovery_without_reset(self):
        risk = _risk(red_latch_enabled=False, dd_ema_span_minutes=5.0)
        # Push into RED.
        for minute in range(20):
            risk.assess(_acc(equity=7_000.0), timestamp_ms=minute * 60_000)
        # Recover → tier drops below RED automatically.
        tier = risk.assess(_acc(equity=10_000.0), timestamp_ms=120 * 60_000)
        assert tier != RiskTier.RED
        assert risk.red_latched is False


# ---------------------------------------------------------------------------
# Integration: latched RED still produces panic close via filter_orders
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_filter_orders_in_latched_red_returns_panic_closes(self):
        risk = _risk()
        acc = _acc(equity=7_000.0)
        acc.symbols["BTC"] = SymbolState("BTC", last_price=50_000.0)
        from combo_bot.types import Position

        acc.symbols["BTC"].position_long = Position(0.1, 50_000.0)

        # First call latches.
        out = risk.filter_orders([], acc, timestamp=0)
        assert any(o.is_market and o.reduce_only for o in out)
        assert risk.red_latched is True

        # Recover account fully — latched, so filter still panic closes.
        acc.equity = 10_000.0
        acc.equity_peak = 10_000.0
        out2 = risk.filter_orders(
            [Order("BTC", Side.LONG, 49_000, 0.01, OrderSource.GRID)],
            acc,
            timestamp=60_000,
        )
        # Latched, so any non-reduce-only entry should be dropped via panic close.
        assert risk.tier == RiskTier.RED
        # Output is the panic-close set (no entry).
        for o in out2:
            assert o.reduce_only

    def test_filter_orders_after_reset_lets_entries_through(self):
        risk = _risk(red_latch_enabled=True, dd_ema_span_minutes=0.0)
        acc = _acc(equity=7_000.0)
        # Latch.
        risk.filter_orders([], acc, timestamp=0)
        assert risk.red_latched is True

        # Reset + healthy account.
        risk.reset_red_latch()
        acc.equity = 10_000.0
        acc.equity_peak = 10_000.0
        order = Order("BTC", Side.LONG, 49_000, 0.01, OrderSource.GRID)
        out = risk.filter_orders([order], acc, timestamp=60_000)
        assert len(out) == 1
        assert out[0].reduce_only is False
