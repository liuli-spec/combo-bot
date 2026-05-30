"""Tests for the live fill events manager — the bridge from exchange
trade history to the bot's protections / kelly / source-PnL ledger."""

from __future__ import annotations

import asyncio

import pytest

from combo_bot.fill_events_manager import (
    FillEventManager,
    FillEventManagerConfig,
)
from combo_bot.types import Fill, OrderSource, Side


class _StubExchange:
    """Minimal exchange that returns pre-canned trades."""

    def __init__(self, trades_per_call: list[list[dict]]):
        self._queue = list(trades_per_call)
        self.calls: list[dict] = []

    async def fetch_my_trades(self, symbol, since=None, limit=None):
        self.calls.append({"symbol": symbol, "since": since, "limit": limit})
        if self._queue:
            return self._queue.pop(0)
        return []


def _trade(
    tid: str,
    ts_ms: int,
    *,
    side: str = "buy",
    price: float = 50_000.0,
    amount: float = 0.01,
    fee_cost: float = 0.1,
    order_id: str = "ord-1",
    realized_pnl: float = 0.0,
) -> dict:
    return {
        "id": tid,
        "timestamp": ts_ms,
        "side": side,
        "price": price,
        "amount": amount,
        "fee": {"cost": fee_cost},
        "order": order_id,
        "info": {"realizedPnl": realized_pnl},
    }


def test_first_poll_returns_all_fresh_trades():
    ex = _StubExchange(
        [
            [
                _trade("t1", 1_000),
                _trade("t2", 2_000),
                _trade("t3", 3_000),
            ],
        ]
    )
    mgr = FillEventManager(ex, FillEventManagerConfig(poll_interval_ms=0))
    sink: list[Fill] = []
    fresh = asyncio.run(
        mgr.poll("BTC/USDT:USDT", now_ms=10_000, sink=sink.extend),
    )
    assert len(fresh) == 3
    assert {f.timestamp for f in fresh} == {1_000, 2_000, 3_000}
    assert len(sink) == 3


def test_dedup_drops_repeated_trade_ids():
    """A second poll that re-serves the same trade ids must not re-emit."""
    repeated_trades = [_trade("t1", 1_000), _trade("t2", 2_000)]
    ex = _StubExchange([repeated_trades, list(repeated_trades)])
    mgr = FillEventManager(ex, FillEventManagerConfig(poll_interval_ms=0))
    asyncio.run(mgr.poll("BTC/USDT:USDT", now_ms=10_000, sink=lambda _: None))
    second_round: list[Fill] = []
    asyncio.run(
        mgr.poll("BTC/USDT:USDT", now_ms=20_000, sink=second_round.extend),
    )
    assert second_round == []


def test_watermark_advances_past_latest_trade():
    """After ingesting up to ts=3000, the next poll's `since` is > 3000."""
    ex = _StubExchange(
        [
            [_trade("t1", 1_000), _trade("t2", 3_000)],
            [],
        ]
    )
    mgr = FillEventManager(ex, FillEventManagerConfig(poll_interval_ms=0))
    asyncio.run(mgr.poll("BTC/USDT:USDT", now_ms=10_000, sink=lambda _: None))
    asyncio.run(mgr.poll("BTC/USDT:USDT", now_ms=20_000, sink=lambda _: None))
    assert ex.calls[-1]["since"] == 3_001


def test_poll_interval_throttles_consecutive_calls():
    """Two polls within poll_interval_ms must result in only one fetch."""
    ex = _StubExchange([[_trade("t1", 1_000)]])
    mgr = FillEventManager(ex, FillEventManagerConfig(poll_interval_ms=30_000))
    asyncio.run(mgr.poll("BTC/USDT:USDT", now_ms=0, sink=lambda _: None))
    asyncio.run(mgr.poll("BTC/USDT:USDT", now_ms=10_000, sink=lambda _: None))
    assert len(ex.calls) == 1


def test_source_attribution_via_register_outgoing():
    """When a trade's order_id is registered with source=TREND, the
    emitted Fill must carry source=TREND."""
    ex = _StubExchange([[_trade("t1", 1_000, order_id="ord-trend-1")]])
    mgr = FillEventManager(ex, FillEventManagerConfig(poll_interval_ms=0))
    mgr.register_outgoing(
        "ord-trend-1",
        OrderSource.TREND,
        Side.LONG,
        reduce_only=False,
    )
    captured: list[Fill] = []
    asyncio.run(mgr.poll("BTC/USDT:USDT", now_ms=0, sink=captured.extend))
    assert len(captured) == 1
    assert captured[0].source == OrderSource.TREND


def test_snapshot_restores_watermark_and_order_attribution():
    first = _StubExchange([[_trade("t1", 1_000, order_id="ord-trend-1")]])
    mgr = FillEventManager(first, FillEventManagerConfig(poll_interval_ms=0))
    mgr.register_outgoing(
        "ord-trend-1",
        OrderSource.TREND,
        Side.LONG,
        reduce_only=False,
    )
    asyncio.run(mgr.poll("BTC/USDT:USDT", now_ms=0, sink=lambda _: None))

    restored_exchange = _StubExchange(
        [
            [_trade("t2", 2_000, order_id="ord-trend-1")],
        ]
    )
    restored = FillEventManager(
        restored_exchange,
        FillEventManagerConfig(poll_interval_ms=0),
    )
    restored.load_snapshot(mgr.snapshot())

    captured: list[Fill] = []
    asyncio.run(restored.poll("BTC/USDT:USDT", now_ms=10_000, sink=captured.extend))

    assert restored_exchange.calls[0]["since"] == 1_001
    assert len(captured) == 1
    assert captured[0].source == OrderSource.TREND


def test_fill_carries_reduce_only_metadata_from_order_registry():
    ex = _StubExchange(
        [
            [_trade("t1", 1_000, side="sell", order_id="ord-long-close")],
        ]
    )
    mgr = FillEventManager(ex, FillEventManagerConfig(poll_interval_ms=0))
    mgr.register_outgoing(
        "ord-long-close",
        OrderSource.TREND,
        Side.LONG,
        reduce_only=True,
    )
    captured: list[Fill] = []
    asyncio.run(mgr.poll("BTC/USDT:USDT", now_ms=0, sink=captured.extend))
    assert captured[0].reduce_only is True


def test_unknown_order_id_defaults_to_grid_source():
    ex = _StubExchange([[_trade("t1", 1_000, order_id="unknown")]])
    mgr = FillEventManager(ex, FillEventManagerConfig(poll_interval_ms=0))
    captured: list[Fill] = []
    asyncio.run(mgr.poll("BTC/USDT:USDT", now_ms=0, sink=captured.extend))
    assert captured[0].source == OrderSource.GRID


def test_realized_pnl_extracted_from_info_field():
    """Binance puts realized PnL in info.realizedPnl; honor it."""
    ex = _StubExchange(
        [
            [_trade("t1", 1_000, realized_pnl=12.5)],
        ]
    )
    mgr = FillEventManager(ex, FillEventManagerConfig(poll_interval_ms=0))
    captured: list[Fill] = []
    asyncio.run(mgr.poll("BTC/USDT:USDT", now_ms=0, sink=captured.extend))
    assert captured[0].realized_pnl == pytest.approx(12.5)


def test_side_inference_for_buy_entry_vs_short_close():
    """A `buy` trade with reduce_only metadata must be a SHORT close."""
    ex = _StubExchange(
        [
            [_trade("t1", 1_000, side="buy", order_id="ord-short-close")],
        ]
    )
    mgr = FillEventManager(ex, FillEventManagerConfig(poll_interval_ms=0))
    mgr.register_outgoing(
        "ord-short-close",
        OrderSource.GRID,
        Side.SHORT,
        reduce_only=True,
    )
    captured: list[Fill] = []
    asyncio.run(mgr.poll("BTC/USDT:USDT", now_ms=0, sink=captured.extend))
    assert captured[0].side == Side.SHORT


def test_empty_response_emits_nothing():
    ex = _StubExchange([[]])
    mgr = FillEventManager(ex, FillEventManagerConfig(poll_interval_ms=0))
    captured: list[Fill] = []
    asyncio.run(mgr.poll("BTC/USDT:USDT", now_ms=0, sink=captured.extend))
    assert captured == []


def test_exchange_exception_does_not_crash_poll():
    class _ErrorExchange:
        async def fetch_my_trades(self, *_a, **_k):
            raise RuntimeError("boom")

    mgr = FillEventManager(_ErrorExchange(), FillEventManagerConfig(poll_interval_ms=0))
    captured: list[Fill] = []
    fresh = asyncio.run(
        mgr.poll("BTC/USDT:USDT", now_ms=0, sink=captured.extend),
    )
    assert fresh == []
    assert captured == []
