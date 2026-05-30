"""Round-18 tests:

* Incomplete resolved_via_fetch markers survive past TTL.
* Same-ms-stuck pagination escalates to a parked symbol after N polls.
* strategy config supports ``params`` kwargs.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

# ────────────────────────────────────────────────────────────────────
# P0 — incomplete RVF marker not aged out
# ────────────────────────────────────────────────────────────────────


def test_incomplete_rvf_marker_survives_ttl_age_out():
    """seen < qty markers must NOT be evicted by the TTL sweep.
    Otherwise a delayed trade stream would lose the dedup guard and
    double-write the bucket on arrival."""
    from combo_bot.live import LiveTrader

    incomplete = {"qty": 0.05, "seen": 0.02, "ts": 1.0}
    complete = {"qty": 0.05, "seen": 0.05, "ts": 1.0}
    # Cutoff far in the future — both markers' ts is way before it.
    cutoff = 1_700_000_000_000.0
    assert LiveTrader._rvf_marker_alive(incomplete, cutoff) is True, (
        "incomplete marker must survive TTL — delayed trades still need "
        "the dedup guard"
    )
    assert (
        LiveTrader._rvf_marker_alive(complete, cutoff) is False
    ), "fully-resolved marker should age out normally"


def test_incomplete_marker_via_sink_after_long_delay():
    """End-to-end: marker created at T=0, trade arrives at T=2h.
    Without the round-18 fix the marker would have been aged out at
    T=1h, the trade would have written the bucket → double position."""
    from combo_bot.fill_events_manager import FillEventManagerConfig
    from combo_bot.live import LiveConfig, LiveTrader
    from combo_bot.types import ExchangeParams, Position, SymbolState

    class _Stub:
        def __init__(self):
            self.trades_by_call: list[list[dict]] = []

        async def load_markets(self):
            return {}

        def market(self, _):
            return {
                "precision": {"amount": 0.001, "price": 0.01},
                "limits": {"amount": {"min": 0.001}, "cost": {"min": 5.0}},
                "maker": 0.0002,
                "taker": 0.0005,
            }

        async def fetch_balance(self, _=None):
            return {"USDT": {"total": 10_000.0}}

        async def fetch_positions(self, _):
            return []

        async def fetch_funding_rate(self, _):
            return {"fundingRate": 0.0}

        async def fetch_ohlcv(self, *a, **k):
            return []

        async def fetch_my_trades(self, *a, **k):
            return self.trades_by_call.pop(0) if self.trades_by_call else []

        async def fetch_open_orders(self, _):
            return []

        async def create_order(self, *a, **k):
            return {"id": "ex-1", "status": "open"}

        async def cancel_order(self, *a, **k):
            return {}

        async def set_leverage(self, *a, **k):
            return {}

        async def set_margin_mode(self, *a, **k):
            return {}

    with tempfile.TemporaryDirectory() as tmp:
        ex = _Stub()
        cfg = LiveConfig(
            symbols=["BTC/USDT:USDT"],
            dry_run=False,
            state_file=str(Path(tmp) / "state.json"),
        )
        trader = LiveTrader(
            cfg,
            ex,
            fill_events_config=FillEventManagerConfig(
                poll_interval_ms=0,
            ),
        )
        trader.exchange_params["BTC/USDT:USDT"] = ExchangeParams(
            qty_step=0.001,
            price_step=0.01,
            min_qty=0.001,
            min_cost=5.0,
        )
        trader.account.symbols["BTC/USDT:USDT"] = SymbolState(
            symbol="BTC/USDT:USDT",
        )
        # Manually install an incomplete marker and a populated bucket
        # (as if fetch_order had already written 0.05 but trade stream
        # hasn't surfaced any matching fills yet).
        marker = {"qty": 0.05, "seen": 0.0, "ts": 1.0}  # very old ts
        trader._resolved_via_fetch["cb-delayed"] = marker
        trader._resolved_via_fetch["ex-delayed"] = marker
        trader.account.symbols["BTC/USDT:USDT"].trend_long = Position(
            size=0.05,
            entry_price=50_000.0,
        )

        # Trade finally arrives a long time after the marker's ts.
        ex.trades_by_call = [
            [
                {
                    "id": "t-late",
                    "timestamp": 10_000_000,
                    "side": "buy",
                    "price": 50_000.0,
                    "amount": 0.05,
                    "fee": {"cost": 0.0},
                    "order": "ex-delayed",
                    "clientOrderId": "cb-delayed",
                    "info": {},
                }
            ]
        ]
        asyncio.run(trader._refresh_fills())
        # Bucket must still be 0.05, NOT 0.10.
        assert trader.account.symbols["BTC/USDT:USDT"].trend_long.size == pytest.approx(
            0.05
        ), (
            "incomplete marker must survive arbitrary TTL so the late "
            "trade doesn't double the bucket"
        )


# ────────────────────────────────────────────────────────────────────
# P1 — stuck-pagination escalates after N consecutive polls
# ────────────────────────────────────────────────────────────────────


def test_stuck_pagination_parks_symbol_after_escalation():
    from combo_bot.fill_events_manager import (
        FillEventManager,
        FillEventManagerConfig,
    )

    burst_ts = 1_000

    class _StuckEx:
        """Models an exchange that doesn't honor fromId: every call
        returns the same full page of same-ms trades. This is exactly
        the pathological case the escalation guards against."""

        def __init__(self):
            self.calls = 0

        async def fetch_my_trades(self, symbol, since=None, limit=None, params=None):
            self.calls += 1
            return [
                {
                    "id": f"burst-{i}",
                    "timestamp": burst_ts,
                    "side": "buy",
                    "price": 50.0,
                    "amount": 0.001,
                    "fee": {"cost": 0.0},
                    "order": "o",
                    "info": {},
                }
                for i in range(2)
            ]

    mgr = FillEventManager(
        _StuckEx(),
        FillEventManagerConfig(poll_interval_ms=0, page_size=2),
    )
    mgr._stuck_escalate_after = 2  # speed up the test
    sym = "BTC/USDT:USDT"

    # First poll establishes the watermark at burst_ts (the very first
    # full-page same-ms isn't "stuck" yet — cursor was None going in).
    asyncio.run(mgr.poll(sym, now_ms=0, sink=lambda fs: None))
    # Subsequent polls: cursor IS set, fromId returns identical ids
    # → stuck branch fires, counter increments.
    asyncio.run(mgr.poll(sym, now_ms=1, sink=lambda fs: None))
    asyncio.run(mgr.poll(sym, now_ms=2, sink=lambda fs: None))
    assert sym in mgr._stuck_symbols, (
        f"symbol should be parked after {mgr._stuck_escalate_after} "
        f"consecutive stuck polls; counters={mgr._stuck_count}"
    )

    # Once parked, polls short-circuit — exchange isn't queried.
    pre_call_count = mgr.exchange.calls
    captured: list = []
    asyncio.run(mgr.poll(sym, now_ms=3, sink=captured.extend))
    assert captured == []
    assert (
        mgr.exchange.calls == pre_call_count
    ), "stuck symbol must not be polled again until clear_stuck"

    # clear_stuck unblocks the next poll.
    mgr.clear_stuck(sym)
    assert sym not in mgr._stuck_symbols


def test_stuck_count_resets_on_normal_short_page_progress():
    """A clean drain (short page) should clear the consecutive-stuck
    counter so a single transient blip doesn't escalate."""
    from combo_bot.fill_events_manager import (
        FillEventManager,
        FillEventManagerConfig,
    )

    class _MixedEx:
        """Phase 1: full stuck page (same ids every call). Phase 2:
        starts returning short pages with different timestamps."""

        def __init__(self):
            self.phase = "stuck"

        async def fetch_my_trades(self, symbol, since=None, limit=None, params=None):
            if self.phase == "stuck":
                return [
                    {
                        "id": f"stuck-{i}",
                        "timestamp": 1_000,
                        "side": "buy",
                        "price": 50.0,
                        "amount": 0.001,
                        "fee": {"cost": 0.0},
                        "order": "o",
                        "info": {},
                    }
                    for i in range(2)
                ]
            return [
                {
                    "id": "drain-1",
                    "timestamp": 2_000,
                    "side": "buy",
                    "price": 50.0,
                    "amount": 0.001,
                    "fee": {"cost": 0.0},
                    "order": "o",
                    "info": {},
                }
            ]

    ex = _MixedEx()
    mgr = FillEventManager(
        ex,
        FillEventManagerConfig(poll_interval_ms=0, page_size=2),
    )
    mgr._stuck_escalate_after = 10  # plenty of headroom
    sym = "BTC/USDT:USDT"
    # First poll: iter 1 sets cursor, iter 2 detects same-ms stuck via
    # fromId-not-helping path → counter starts at 1.
    asyncio.run(mgr.poll(sym, now_ms=0, sink=lambda fs: None))
    assert mgr._stuck_count.get(sym, 0) >= 1
    # Exchange now drains cleanly via a short page.
    ex.phase = "drain"
    asyncio.run(mgr.poll(sym, now_ms=1, sink=lambda fs: None))
    assert (
        sym not in mgr._stuck_count
    ), f"clean drain must reset stuck counter; got {mgr._stuck_count!r}"
    assert sym not in mgr._stuck_symbols


# ────────────────────────────────────────────────────────────────────
# P2 — strategy config supports params
# ────────────────────────────────────────────────────────────────────


def test_build_strategy_passes_params_kwargs():
    from combo_bot.fusion_config import build_strategy
    from combo_bot.strategy import IStrategy

    class _ParamStrategy(IStrategy):
        def __init__(self, rsi_period: int = 14, threshold: float = 0.5):
            super().__init__() if hasattr(IStrategy, "__init__") else None
            self.rsi_period = rsi_period
            self.threshold = threshold

        def populate_indicators(self, df, m):
            return df

        def populate_entry_trend(self, df, m):
            return df

        def populate_exit_trend(self, df, m):
            return df

    # Register class so the dotted-path lookup works.
    import combo_bot.fusion_config as fc

    fc._STRATEGY_REGISTRY["_ParamStrategy"] = _ParamStrategy
    try:
        built = build_strategy(
            {
                "strategy": {
                    "class": "_ParamStrategy",
                    "params": {"rsi_period": 21, "threshold": 0.75},
                },
            }
        )
        assert isinstance(built, _ParamStrategy)
        assert built.rsi_period == 21
        assert built.threshold == 0.75

        # No params → defaults stay.
        default_built = build_strategy({"strategy": {"class": "_ParamStrategy"}})
        assert default_built.rsi_period == 14
    finally:
        fc._STRATEGY_REGISTRY.pop("_ParamStrategy", None)


def test_build_strategy_raises_on_bad_params():
    """Unknown kwargs should bubble up as a TypeError so a misspelled
    parameter doesn't silently get dropped."""
    from combo_bot.fusion_config import build_strategy
    from combo_bot.strategy import IStrategy

    class _StrictStrategy(IStrategy):
        def __init__(self, rsi_period: int = 14):
            self.rsi_period = rsi_period

        def populate_indicators(self, df, m):
            return df

        def populate_entry_trend(self, df, m):
            return df

        def populate_exit_trend(self, df, m):
            return df

    import combo_bot.fusion_config as fc

    fc._STRATEGY_REGISTRY["_StrictStrategy"] = _StrictStrategy
    try:
        with pytest.raises(TypeError):
            build_strategy(
                {
                    "strategy": {
                        "class": "_StrictStrategy",
                        "params": {"misspelled_key": 123},
                    },
                }
            )
    finally:
        fc._STRATEGY_REGISTRY.pop("_StrictStrategy", None)
