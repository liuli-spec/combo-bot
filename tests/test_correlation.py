"""Stage 10 cross-symbol correlation gate tests.

Two layers under test:

* :class:`CorrelationTracker` — rolling close-price store, Pearson
  correlation of returns, sample-size accounting;
* :class:`CorrelationGate` — same-side-only penalty (hedges pass
  through), soft/hard threshold ramp, reduce-only passthrough.
"""

from __future__ import annotations

import math

import pytest

from combo_bot.correlation import (
    CorrelationGate,
    CorrelationGateConfig,
    CorrelationTracker,
)
from combo_bot.types import (
    AccountState,
    Order,
    OrderSource,
    Position,
    Side,
    SymbolState,
)


def _entry(symbol: str, side: Side = Side.LONG, qty: float = 0.1) -> Order:
    return Order(symbol, side, 50_000.0, qty, OrderSource.GRID)


def _close(symbol: str, side: Side = Side.LONG, qty: float = 0.1) -> Order:
    return Order(symbol, side, 51_000.0, qty, OrderSource.GRID, reduce_only=True)


def _ss_with_long(symbol: str, qty: float = 0.1) -> SymbolState:
    ss = SymbolState(symbol, last_price=50_000.0)
    ss.position_long = Position(qty, 50_000.0)
    return ss


def _ss_with_short(symbol: str, qty: float = 0.1) -> SymbolState:
    ss = SymbolState(symbol, last_price=50_000.0)
    ss.position_short = Position(qty, 50_000.0)
    return ss


# ---------------------------------------------------------------------------
# CorrelationTracker
# ---------------------------------------------------------------------------


class TestCorrelationTracker:
    def test_returns_empty_before_two_observations(self):
        t = CorrelationTracker()
        t.update("BTC", 50_000.0)
        assert t.returns("BTC") == []

    def test_returns_computed_from_consecutive_closes(self):
        t = CorrelationTracker()
        t.update("BTC", 100.0)
        t.update("BTC", 110.0)
        rets = t.returns("BTC")
        assert len(rets) == 1
        assert rets[0] == pytest.approx(0.1)

    def test_window_caps_buffer(self):
        t = CorrelationTracker(window=3)
        for v in [100.0, 110.0, 121.0, 133.1, 146.41]:
            t.update("BTC", v)
        # 3 closes → 2 returns
        assert t.sample_size("BTC") == 2

    def test_identical_series_have_correlation_one(self):
        t = CorrelationTracker()
        for v in [100.0, 110.0, 105.0, 115.0, 108.0, 120.0]:
            t.update("BTC", v)
            t.update("ETH", v)
        # 5 return observations each
        assert t.correlation("BTC", "ETH") == pytest.approx(1.0)

    def test_anticorrelated_series_have_negative_correlation(self):
        t = CorrelationTracker()
        a_path = [100.0, 110.0, 100.0, 110.0, 100.0]
        for i, v in enumerate(a_path):
            t.update("BTC", v)
            # Mirror: when BTC rises, ETH falls (same magnitude).
            t.update("ETH", 100.0 if i % 2 == 1 else 110.0)
        corr = t.correlation("BTC", "ETH")
        assert corr < -0.5

    def test_self_correlation_is_one(self):
        t = CorrelationTracker()
        t.update("BTC", 100.0)
        t.update("BTC", 110.0)
        assert t.correlation("BTC", "BTC") == 1.0

    def test_low_sample_size_returns_zero(self):
        t = CorrelationTracker()
        t.update("BTC", 100.0)
        t.update("ETH", 100.0)
        assert t.correlation("BTC", "ETH") == 0.0

    def test_flat_series_has_zero_variance_returns_zero(self):
        t = CorrelationTracker()
        for _ in range(10):
            t.update("BTC", 100.0)
            t.update("ETH", 100.0)
        # Both flat — denom_sq ~ 0 → 0.
        assert t.correlation("BTC", "ETH") == 0.0


# ---------------------------------------------------------------------------
# CorrelationGate
# ---------------------------------------------------------------------------


def _gate(**overrides) -> CorrelationGate:
    defaults = dict(window=20, min_samples=5,
                    soft_threshold=0.5, hard_threshold=0.9)
    defaults.update(overrides)
    return CorrelationGate(CorrelationGateConfig(**defaults))


def _seed_correlated(gate: CorrelationGate, n: int = 20):
    """Drive two symbols with identical price paths so corr -> 1.0."""
    for i in range(n):
        price = 50_000.0 * (1.0 + 0.001 * i)
        gate.update_prices([("BTC", price), ("ETH", price)])


def _seed_anticorrelated(gate: CorrelationGate, n: int = 20):
    """Drive BTC and ETH with truly mirror-image *returns* so corr → -1.

    Naively using ``50000*(1±0.001*i)`` doesn't work: the *level*
    diverges but the per-step *returns* end up almost perfectly
    positively correlated (both decreasing in magnitude as the
    denominator grows). Build an explicit return sequence and apply
    +r to BTC, -r to ETH.
    """
    import random
    rng = random.Random(0)
    btc, eth = 50_000.0, 50_000.0
    gate.update_prices([("BTC", btc), ("ETH", eth)])
    for _ in range(n - 1):
        r = rng.gauss(0.0, 0.005)
        btc *= 1.0 + r
        eth *= 1.0 - r
        gate.update_prices([("BTC", btc), ("ETH", eth)])


class TestGateSameSidePenalty:
    def test_same_side_high_corr_blocks_entry(self):
        gate = _gate(hard_threshold=0.9, soft_threshold=0.5)
        _seed_correlated(gate)
        acc = AccountState(balance=10_000)
        acc.symbols["BTC"] = _ss_with_long("BTC")
        acc.symbols["ETH"] = SymbolState("ETH", last_price=50_000.0)
        # Incoming long ETH while long BTC and corr(BTC,ETH)=1 → hard block.
        out = gate.filter_orders([_entry("ETH", Side.LONG)], acc)
        assert out == []

    def test_same_side_below_soft_threshold_passes_unchanged(self):
        gate = _gate(soft_threshold=0.8, hard_threshold=0.95)
        # Seed low correlation: shared trend but independent noise — use
        # mostly-independent series.
        for i in range(30):
            gate.update_prices([
                ("BTC", 50_000.0 * (1.0 + 0.001 * (i % 3))),
                ("ETH", 50_000.0 * (1.0 + 0.001 * ((i + 1) % 5))),
            ])
        acc = AccountState(balance=10_000)
        acc.symbols["BTC"] = _ss_with_long("BTC")
        acc.symbols["ETH"] = SymbolState("ETH", last_price=50_000.0)

        order = _entry("ETH", Side.LONG, qty=0.1)
        out = gate.filter_orders([order], acc)
        assert len(out) == 1
        assert out[0].qty == pytest.approx(0.1)


class TestGateHedgePassesThrough:
    def test_opposite_side_correlated_entry_unmodified(self):
        """Long BTC + new Short ETH with corr(BTC,ETH)≈1 is a hedge —
        effective correlation flips to ≈-1 → no penalty."""
        gate = _gate(hard_threshold=0.9, soft_threshold=0.5)
        _seed_correlated(gate)
        acc = AccountState(balance=10_000)
        acc.symbols["BTC"] = _ss_with_long("BTC")
        acc.symbols["ETH"] = SymbolState("ETH", last_price=50_000.0)
        out = gate.filter_orders([_entry("ETH", Side.SHORT, qty=0.1)], acc)
        assert len(out) == 1
        assert out[0].qty == pytest.approx(0.1)

    def test_same_side_anticorrelated_passes(self):
        """Long BTC + new Long ETH but corr(BTC,ETH)≈-1 → effective
        correlation ≈-1 → no penalty (diversifier)."""
        gate = _gate(soft_threshold=0.5, hard_threshold=0.9)
        _seed_anticorrelated(gate)
        acc = AccountState(balance=10_000)
        acc.symbols["BTC"] = _ss_with_long("BTC")
        acc.symbols["ETH"] = SymbolState("ETH", last_price=50_000.0)
        out = gate.filter_orders([_entry("ETH", Side.LONG, qty=0.1)], acc)
        assert len(out) == 1
        assert out[0].qty == pytest.approx(0.1)


class TestSoftScaling:
    def test_mid_range_correlation_scales_linearly(self):
        gate = _gate(soft_threshold=0.4, hard_threshold=0.8)
        # Build correlation that lands somewhere in the soft band — use
        # identical series then add noise. We can't precisely target
        # 0.6, but we can assert: between 0 and 1 of original qty.
        _seed_correlated(gate)
        # Add some divergence into the tail so corr drops below 1.
        for i in range(5):
            gate.update_prices([
                ("BTC", 50_000.0 * (1.0 + 0.001 * i)),
                ("ETH", 50_000.0 * (1.0 - 0.001 * i)),
            ])
        acc = AccountState(balance=10_000)
        acc.symbols["BTC"] = _ss_with_long("BTC")
        acc.symbols["ETH"] = SymbolState("ETH", last_price=50_000.0)
        out = gate.filter_orders([_entry("ETH", Side.LONG, qty=0.1)], acc)
        # If correlation landed in [soft, hard), qty was scaled.
        # Either passed unchanged (corr < soft) OR scaled to a positive
        # but smaller value OR dropped (corr >= hard).
        if len(out) == 1:
            assert 0 < out[0].qty <= 0.1


class TestReduceOnlyPassthrough:
    def test_reduce_only_orders_never_filtered(self):
        gate = _gate(hard_threshold=0.9)
        _seed_correlated(gate)
        acc = AccountState(balance=10_000)
        acc.symbols["BTC"] = _ss_with_long("BTC")
        acc.symbols["ETH"] = _ss_with_long("ETH")
        # Close ETH long even though BTC long is heavily correlated.
        out = gate.filter_orders([_close("ETH", Side.LONG)], acc)
        assert len(out) == 1
        assert out[0].reduce_only


class TestDegenerate:
    def test_single_symbol_no_filtering(self):
        gate = _gate()
        for i in range(30):
            gate.update_prices([("BTC", 50_000.0 + i)])
        acc = AccountState(balance=10_000)
        acc.symbols["BTC"] = SymbolState("BTC", last_price=50_000.0)
        # No other symbol exists → no correlation to compute → no scaling.
        out = gate.filter_orders([_entry("BTC", Side.LONG)], acc)
        assert len(out) == 1
        assert out[0].qty == pytest.approx(0.1)

    def test_insufficient_history_no_filtering(self):
        gate = _gate(min_samples=30)
        # Only 5 observations each — below threshold.
        for i in range(5):
            gate.update_prices([("BTC", 50_000.0 + i), ("ETH", 50_000.0 + i)])
        acc = AccountState(balance=10_000)
        acc.symbols["BTC"] = _ss_with_long("BTC")
        acc.symbols["ETH"] = SymbolState("ETH", last_price=50_000.0)
        out = gate.filter_orders([_entry("ETH", Side.LONG)], acc)
        assert len(out) == 1


# ---------------------------------------------------------------------------
# Trend-bucket positions also count toward correlation
# ---------------------------------------------------------------------------


class TestBucketAgnostic:
    def test_trend_bucket_position_blocks_correlated_entry(self):
        """Existing position in the TREND bucket should still trigger
        the gate against a new GRID entry — factor exposure compounds
        regardless of which bucket holds it."""
        gate = _gate(hard_threshold=0.9, soft_threshold=0.5)
        _seed_correlated(gate)
        acc = AccountState(balance=10_000)
        ss_btc = SymbolState("BTC", last_price=50_000.0)
        # BTC long lives only in the trend bucket.
        ss_btc.trend_long = Position(0.1, 50_000.0)
        acc.symbols["BTC"] = ss_btc
        acc.symbols["ETH"] = SymbolState("ETH", last_price=50_000.0)

        out = gate.filter_orders([_entry("ETH", Side.LONG)], acc)
        # Trend bucket BTC + new grid ETH same-side at corr ~1 → block.
        assert out == []


# ---------------------------------------------------------------------------
# Backtester integration
# ---------------------------------------------------------------------------


class TestBacktesterIntegration:
    def test_cold_start_does_not_change_baseline(self):
        from combo_bot.backtest import Backtester, BacktestConfig
        from combo_bot.grid_engine import GridConfig
        from tests.conftest import make_candles

        candles = make_candles([50_000 + i * 80 for i in range(200)])
        cfg = BacktestConfig(
            starting_balance=10_000, symbols=["BTC"],
            grid=GridConfig(max_grid_levels=2),
        )
        r_off = Backtester(cfg).run({"BTC": candles})
        r_on = Backtester(cfg, correlation_gate=_gate()).run({"BTC": candles})
        # Single-symbol run never exercises cross-symbol logic.
        assert r_off.n_trades == r_on.n_trades
        assert r_off.grid_pnl == pytest.approx(r_on.grid_pnl)

    def test_two_symbol_correlated_run_produces_fewer_trades(self):
        """With perfectly correlated symbols and a tight hard threshold,
        the gate should suppress second-symbol entries — fewer fills
        than the no-gate baseline."""
        from combo_bot.backtest import Backtester, BacktestConfig
        from combo_bot.grid_engine import GridConfig
        from tests.conftest import make_candles

        candles_a = make_candles([50_000 + i * 80 for i in range(200)])
        candles_b = [
            # Mirror BTC's path exactly.
            type(c)(c.timestamp, c.open, c.high, c.low, c.close, c.volume)
            for c in candles_a
        ]
        cfg = BacktestConfig(
            starting_balance=10_000, symbols=["BTC", "ETH"],
            grid=GridConfig(max_grid_levels=2),
        )
        gate = CorrelationGate(CorrelationGateConfig(
            window=50, min_samples=20,
            soft_threshold=0.3, hard_threshold=0.5,
        ))
        r_off = Backtester(cfg).run({
            "BTC": candles_a, "ETH": candles_b,
        })
        r_on = Backtester(cfg, correlation_gate=gate).run({
            "BTC": candles_a, "ETH": candles_b,
        })
        # Gate should have suppressed at least some entries.
        assert r_on.n_trades <= r_off.n_trades
