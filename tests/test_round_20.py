"""Round-20 tests — fail-closed across the rest of the ingestion
surface:

* fetch_my_trades exceptions escalate into a parked symbol.
* fetch_open_orders failure also blocks limit reduce_only ladders
  (only market reduce-only passes through as the emergency-exit path).
* intent_journal replay parse failure sets _persistence_failed so
  live blocks every new entry until manual reset.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

# ────────────────────────────────────────────────────────────────────
# P0: fetch_my_trades failure → STUCK after N tries
# ────────────────────────────────────────────────────────────────────


def test_fetch_my_trades_exception_escalates_to_stuck():
    from combo_bot.fill_events_manager import (
        FillEventManager,
        FillEventManagerConfig,
    )

    class _BrokenEx:
        def __init__(self):
            self.calls = 0

        async def fetch_my_trades(self, *a, **k):
            self.calls += 1
            raise RuntimeError("network down")

    mgr = FillEventManager(
        _BrokenEx(),
        FillEventManagerConfig(poll_interval_ms=0),
    )
    mgr._stuck_escalate_after = 3
    sym = "BTC/USDT:USDT"
    # 3 consecutive failures escalate.
    for i in range(3):
        asyncio.run(mgr.poll(sym, now_ms=i, sink=lambda fs: None))
    assert sym in mgr._stuck_symbols, (
        "fetch_my_trades failures must accumulate in the same stuck "
        "counter same-ms-pagination uses; both signals mean the fill "
        "ledger is unsafe"
    )


def test_pagination_interrupted_holds_watermark_inside_stuck_ms():
    """Round-20 P0: page 1 succeeds with same-ms trades, page 2
    raises. Watermark MUST NOT advance past the millisecond — the
    real ChatGPT-found bug was that ``max_ts + 1`` jumped to 1001
    and a same-ms sibling on page 2 was lost forever."""
    from combo_bot.fill_events_manager import (
        FillEventManager,
        FillEventManagerConfig,
    )

    burst_ts = 1_000

    class _PartialEx:
        def __init__(self):
            self.calls = 0

        async def fetch_my_trades(self, *a, **k):
            self.calls += 1
            if self.calls == 1:
                return [
                    {
                        "id": f"p1-{i}",
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
            raise RuntimeError("page-2 fetch failed")

    mgr = FillEventManager(
        _PartialEx(),
        FillEventManagerConfig(poll_interval_ms=0, page_size=2),
    )
    sym = "BTC/USDT:USDT"
    asyncio.run(mgr.poll(sym, now_ms=0, sink=lambda fs: None))
    # Watermark MUST be at burst_ts (held), NOT burst_ts + 1.
    assert mgr._last_ts_ms.get(sym) == burst_ts, (
        f"watermark must hold at burst_ts={burst_ts} after a page-2 "
        f"failure; got {mgr._last_ts_ms.get(sym)} — same-ms siblings "
        f"would be permanently lost"
    )
    # last_poll_failed reflects the single-tick fail-closed signal.
    assert mgr.last_poll_failed(sym) is True


def test_last_poll_failed_clears_after_clean_poll():
    from combo_bot.fill_events_manager import (
        FillEventManager,
        FillEventManagerConfig,
    )

    class _RecoverEx:
        def __init__(self):
            self.calls = 0

        async def fetch_my_trades(self, *a, **k):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("once")
            return []

    mgr = FillEventManager(
        _RecoverEx(),
        FillEventManagerConfig(poll_interval_ms=0),
    )
    sym = "BTC/USDT:USDT"
    asyncio.run(mgr.poll(sym, now_ms=0, sink=lambda fs: None))
    assert mgr.last_poll_failed(sym) is True
    asyncio.run(mgr.poll(sym, now_ms=1, sink=lambda fs: None))
    assert mgr.last_poll_failed(sym) is False


def test_fetch_my_trades_transient_failure_does_not_escalate():
    """A single failure shouldn't park the symbol — operators don't
    want false alarms on every transient network blip."""
    from combo_bot.fill_events_manager import (
        FillEventManager,
        FillEventManagerConfig,
    )

    class _BlinkEx:
        def __init__(self):
            self.calls = 0

        async def fetch_my_trades(self, *a, **k):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient")
            return []

    mgr = FillEventManager(
        _BlinkEx(),
        FillEventManagerConfig(poll_interval_ms=0),
    )
    mgr._stuck_escalate_after = 3
    sym = "BTC/USDT:USDT"
    asyncio.run(mgr.poll(sym, now_ms=0, sink=lambda fs: None))
    asyncio.run(mgr.poll(sym, now_ms=1, sink=lambda fs: None))
    asyncio.run(mgr.poll(sym, now_ms=2, sink=lambda fs: None))
    assert sym not in mgr._stuck_symbols


# ────────────────────────────────────────────────────────────────────
# P1: fetch_open_orders failure also blocks LIMIT reduce_only
# ────────────────────────────────────────────────────────────────────


class _OpenFailEx:
    def __init__(self):
        self.created: list[dict] = []
        self.open_call_count = 0

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
        return []

    async def fetch_open_orders(self, _):
        self.open_call_count += 1
        raise RuntimeError("open orders API down")

    async def create_order(self, sym, ot, side, qty, price, params):
        self.created.append(
            {
                "symbol": sym,
                "type": ot,
                "side": side,
                "qty": qty,
                "price": price,
                "params": params,
            }
        )
        return {"id": f"ex-{len(self.created)}", "status": "open"}

    async def cancel_order(self, *a, **k):
        return {}

    async def set_leverage(self, *a, **k):
        return {}

    async def set_margin_mode(self, *a, **k):
        return {}


def _trader_open_fail(tmpdir: Path):
    from combo_bot.live import LiveConfig, LiveTrader
    from combo_bot.types import ExchangeParams, SymbolState

    ex = _OpenFailEx()
    cfg = LiveConfig(
        symbols=["BTC/USDT:USDT"],
        dry_run=False,
        state_file=str(tmpdir / "state.json"),
    )
    trader = LiveTrader(cfg, ex)
    trader.exchange_params["BTC/USDT:USDT"] = ExchangeParams(
        qty_step=0.001,
        price_step=0.01,
        min_qty=0.001,
        min_cost=5.0,
    )
    trader.account.symbols["BTC/USDT:USDT"] = SymbolState(symbol="BTC/USDT:USDT")
    return trader


def test_fetch_open_orders_failure_blocks_limit_reduce_only():
    from combo_bot.types import Order, OrderSource, Side

    with tempfile.TemporaryDirectory() as tmp:
        trader = _trader_open_fail(Path(tmp))
        limit_tp = Order(
            symbol="BTC/USDT:USDT",
            side=Side.LONG,
            price=51_000.0,
            qty=0.01,
            source=OrderSource.GRID,
            reduce_only=True,
            is_market=False,
        )
        asyncio.run(trader._reconcile_orders([limit_tp]))
        assert trader.exchange.created == [], (
            "limit reduce-only orders must NOT be sent when "
            "fetch_open_orders failed; duplicates the close ladder"
        )


def test_fetch_open_orders_failure_still_allows_market_reduce_only():
    """The emergency-exit path (panic close, trend SL/TP exit) must
    NOT be blocked — those are how the bot bails out of bad positions."""
    from combo_bot.types import Order, OrderSource, Side

    with tempfile.TemporaryDirectory() as tmp:
        trader = _trader_open_fail(Path(tmp))
        market_panic = Order(
            symbol="BTC/USDT:USDT",
            side=Side.LONG,
            price=50_000.0,
            qty=0.01,
            source=OrderSource.RISK,
            reduce_only=True,
            is_market=True,
        )
        asyncio.run(trader._reconcile_orders([market_panic]))
        assert len(trader.exchange.created) == 1


def test_fetch_open_orders_failure_blocks_non_reduce_entries():
    """The pre-existing non-reduce block stays in place."""
    from combo_bot.types import Order, OrderSource, Side

    with tempfile.TemporaryDirectory() as tmp:
        trader = _trader_open_fail(Path(tmp))
        new_entry = Order(
            symbol="BTC/USDT:USDT",
            side=Side.LONG,
            price=50_000.0,
            qty=0.01,
            source=OrderSource.GRID,
        )
        asyncio.run(trader._reconcile_orders([new_entry]))
        assert trader.exchange.created == []


# ────────────────────────────────────────────────────────────────────
# P1: intent_journal replay failure → _persistence_failed
# ────────────────────────────────────────────────────────────────────


def test_intent_journal_midstream_garbage_fails_closed():
    """Round-20 P1: a malformed line in the MIDDLE of the journal
    (not just the trailing partial-write case) must trigger
    fail-closed. Uses a real corrupted JSONL file — no monkeypatching
    so the production replay path is exercised end-to-end."""
    from combo_bot.live import LiveConfig, LiveTrader

    class _Ex:
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
            return []

        async def fetch_open_orders(self, _):
            return []

        async def create_order(self, *a, **k):
            return {"id": "ex", "status": "open"}

        async def cancel_order(self, *a, **k):
            return {}

        async def set_leverage(self, *a, **k):
            return {}

        async def set_margin_mode(self, *a, **k):
            return {}

    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "state.json"
        journal_path = state_path.with_suffix(".intent_journal.jsonl")
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        # Write a journal where line 2 is malformed (mid-stream), but
        # line 3 looks valid. Pre-fix this would have silently
        # processed lines 1 and 3 while skipping the corrupt line.
        good = (
            '{"ts":1,"kind":"submit","cid":"cb-a","symbol":"BTC",'
            '"side":"long","source":"trend","reduce_only":false,'
            '"is_market":true}\n'
        )
        garbage = "this is not valid json at all\n"
        also_good = (
            '{"ts":3,"kind":"submit","cid":"cb-b","symbol":"BTC",'
            '"side":"long","source":"trend","reduce_only":false,'
            '"is_market":true}\n'
        )
        journal_path.write_text(good + garbage + also_good)

        cfg = LiveConfig(
            symbols=["BTC/USDT:USDT"],
            dry_run=False,
            state_file=str(state_path),
        )
        trader = LiveTrader(cfg, _Ex())
        trader._persistence_failed = False
        trader._replay_intent_journal()
        assert trader._persistence_failed is True, (
            "midstream JSON corruption in the journal must trigger "
            "fail-closed; replay can't silently skip the bad line and "
            "keep going"
        )


def test_intent_journal_trailing_partial_line_tolerated():
    """Round-20 P1 boundary: the LAST line — if missing its trailing
    newline (i.e. a write was interrupted by SIGKILL) — must still be
    tolerated. Otherwise every crash mid-write would force operator
    intervention."""
    from combo_bot.intent_journal import IntentJournal

    with tempfile.TemporaryDirectory() as tmp:
        journal_path = Path(tmp) / "j.jsonl"
        good = (
            '{"ts":1,"kind":"submit","cid":"cb-x","symbol":"BTC",'
            '"side":"long","source":"trend","reduce_only":false,'
            '"is_market":true}\n'
        )
        trailing_partial = '{"ts":2,"kind":"submit","cid":"cb-y","sy'
        journal_path.write_text(good + trailing_partial)

        j = IntentJournal(journal_path)
        j.replay()
        assert j.last_replay_failed is False
        assert "cb-x" in j.records
        assert "cb-y" not in j.records
