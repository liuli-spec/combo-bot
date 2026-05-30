"""Round-17 tests:

* resolved_via_fetch survives multiple partial fills of the same order
* resolved_via_fetch survives a process restart
* _resolve_unknowns clears UNKNOWN only when ALL cIDs resolved
* market exit's confirm_trade_exit gets ctx.current_price, not threshold
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest


class _Stub:
    def __init__(self):
        self.opens_by_call: list[list[dict]] = []
        self.positions: list[dict] = []
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
        return self.positions

    async def fetch_funding_rate(self, _):
        return {"fundingRate": 0.0}

    async def fetch_ohlcv(self, *a, **k):
        return []

    async def fetch_my_trades(self, *a, **k):
        return self.trades_by_call.pop(0) if self.trades_by_call else []

    async def fetch_open_orders(self, _):
        return self.opens_by_call.pop(0) if self.opens_by_call else []

    async def create_order(self, *a, **k):
        return {"id": "ex-1", "status": "open"}

    async def cancel_order(self, *a, **k):
        return {}

    async def set_leverage(self, *a, **k):
        return {}

    async def set_margin_mode(self, *a, **k):
        return {}


def _trader(tmp: Path):
    from combo_bot.fill_events_manager import FillEventManagerConfig
    from combo_bot.live import LiveConfig, LiveTrader
    from combo_bot.types import ExchangeParams, SymbolState

    ex = _Stub()
    cfg = LiveConfig(
        symbols=["BTC/USDT:USDT"],
        dry_run=False,
        state_file=str(tmp / "state.json"),
    )
    # poll_interval_ms=0 so tests can _refresh_fills multiple times
    # back to back without hitting the live rate-limit throttle.
    trader = LiveTrader(
        cfg,
        ex,
        fill_events_config=FillEventManagerConfig(poll_interval_ms=0),
    )
    trader.exchange_params["BTC/USDT:USDT"] = ExchangeParams(
        qty_step=0.001,
        price_step=0.01,
        min_qty=0.001,
        min_cost=5.0,
    )
    trader.account.symbols["BTC/USDT:USDT"] = SymbolState(symbol="BTC/USDT:USDT")
    return trader


# ────────────────────────────────────────────────────────────────────
# P0 #1 — multi-fill (partial) trades don't double-write bucket
# ────────────────────────────────────────────────────────────────────


def test_resolved_via_fetch_survives_split_trade_stream():
    """fetch_order says qty=0.02; the exchange returns 2 partial
    trades of 0.01 each (same cID). Neither should re-apply to the
    bucket — marker should only be released after the second trade
    accumulates the full filled qty."""
    from combo_bot.types import Side

    with tempfile.TemporaryDirectory() as tmp:
        trader = _trader(Path(tmp))
        trader.intent_journal.submit(
            cid="cb-split",
            symbol="BTC/USDT:USDT",
            side="long",
            source="trend",
            reduce_only=False,
            is_market=True,
            now_ms=1,
        )
        trader.intent_journal.open(
            cid="cb-split",
            exchange_id="ex-99",
            now_ms=2,
        )
        trader._unknown_overlay[("BTC/USDT:USDT", Side.LONG)] = 1.0

        async def fo(*a, **k):
            return {
                "status": "closed",
                "id": "ex-99",
                "filled": 0.02,
                "average": 50_000.0,
            }

        trader.exchange.fetch_order = fo  # type: ignore[method-assign]
        asyncio.run(trader._resolve_unknowns())
        ss = trader.account.symbols["BTC/USDT:USDT"]
        assert ss.trend_long.size == pytest.approx(0.02)
        # Marker present.
        assert "cb-split" in trader._resolved_via_fetch
        assert trader._resolved_via_fetch["cb-split"]["qty"] == pytest.approx(0.02)

        # First partial trade arrives — must NOT re-apply.
        trader.exchange.trades_by_call = [
            [
                {
                    "id": "t-a",
                    "timestamp": 9_000,
                    "side": "buy",
                    "price": 50_000.0,
                    "amount": 0.01,
                    "fee": {"cost": 0.05},
                    "order": "ex-99",
                    "clientOrderId": "cb-split",
                    "info": {"realizedPnl": "0"},
                },
            ]
        ]
        asyncio.run(trader._refresh_fills())
        assert ss.trend_long.size == pytest.approx(0.02)
        # Marker still present — only saw 0.01 of 0.02.
        assert "cb-split" in trader._resolved_via_fetch
        assert trader._resolved_via_fetch["cb-split"]["seen"] == pytest.approx(0.01)

        # Second partial trade arrives — STILL must not re-apply.
        trader.exchange.trades_by_call = [
            [
                {
                    "id": "t-b",
                    "timestamp": 9_100,
                    "side": "buy",
                    "price": 50_000.0,
                    "amount": 0.01,
                    "fee": {"cost": 0.05},
                    "order": "ex-99",
                    "clientOrderId": "cb-split",
                    "info": {"realizedPnl": "0"},
                },
            ]
        ]
        asyncio.run(trader._refresh_fills())
        assert ss.trend_long.size == pytest.approx(0.02), (
            "second partial trade for a fetch_order-resolved order "
            "must not double the bucket"
        )
        # Marker fully drained, can be released.
        assert "cb-split" not in trader._resolved_via_fetch


# ────────────────────────────────────────────────────────────────────
# P0 #2 — resolved_via_fetch round-trips through state file
# ────────────────────────────────────────────────────────────────────


def test_resolved_via_fetch_round_trips_through_state_file():

    with tempfile.TemporaryDirectory() as tmp:
        trader_a = _trader(Path(tmp))
        marker = {"qty": 0.02, "seen": 0.01, "ts": 1_700_000_000_000.0}
        trader_a._resolved_via_fetch["cb-restart"] = marker
        trader_a._resolved_via_fetch["ex-77"] = marker
        asyncio.run(trader_a._save_state())

        # Inspect raw JSON to confirm shape.
        on_disk = json.loads((Path(tmp) / "state.json").read_text())
        assert "resolved_via_fetch" in on_disk
        groups = on_disk["resolved_via_fetch"]
        assert len(groups) == 1
        assert set(groups[0]["keys"]) == {"cb-restart", "ex-77"}
        assert groups[0]["qty"] == 0.02
        assert groups[0]["seen"] == 0.01

        # Load into trader B and confirm both keys share the same marker.
        trader_b = _trader(Path(tmp))
        asyncio.run(trader_b._load_state())
        assert "cb-restart" in trader_b._resolved_via_fetch
        assert "ex-77" in trader_b._resolved_via_fetch
        # Mutating via one key visible through the other → shared dict.
        trader_b._resolved_via_fetch["cb-restart"]["seen"] += 0.001
        assert trader_b._resolved_via_fetch["ex-77"]["seen"] == pytest.approx(0.011)


# ────────────────────────────────────────────────────────────────────
# P1 #1 — UNKNOWN only clears when ALL cIDs resolved
# ────────────────────────────────────────────────────────────────────


def test_resolve_unknowns_holds_when_only_some_cids_resolved():
    from combo_bot.types import Side

    with tempfile.TemporaryDirectory() as tmp:
        trader = _trader(Path(tmp))
        # Two journal cIDs for the same (sym, side).
        for cid in ("cb-A", "cb-B"):
            trader.intent_journal.submit(
                cid=cid,
                symbol="BTC/USDT:USDT",
                side="long",
                source="trend",
                reduce_only=False,
                is_market=True,
                now_ms=1,
            )
            trader.intent_journal.open(
                cid=cid,
                exchange_id=f"ex-{cid}",
                now_ms=2,
            )
        trader._unknown_overlay[("BTC/USDT:USDT", Side.LONG)] = 1.0
        trader.exchange.opens_by_call = [[]]

        # fetch_order: cb-A resolves cancelled, cb-B returns
        # indeterminate (none status).
        async def fo(order_id=None, symbol=None, params=None):
            if order_id == "ex-cb-A":
                return {"status": "canceled", "id": order_id}
            return {"status": "open", "id": order_id}

        trader.exchange.fetch_order = fo  # type: ignore[method-assign]
        asyncio.run(trader._resolve_unknowns())

        # UNKNOWN must STILL be held — cb-B isn't resolved.
        assert ("BTC/USDT:USDT", Side.LONG) in trader._unknown_overlay
        # cb-A journal marked canceled; cb-B still submit/open.
        assert trader.intent_journal.records["cb-A"]["kind"] == "canceled"
        assert trader.intent_journal.records["cb-B"]["kind"] != "canceled"


# ────────────────────────────────────────────────────────────────────
# P1 #2 — market exit confirm_trade_exit sees ctx.current_price
# ────────────────────────────────────────────────────────────────────


def test_confirm_trade_exit_for_market_sees_current_price_not_threshold():
    from combo_bot.strategy import IStrategy, StrategyRunner, TradeContext
    from combo_bot.types import (
        AccountState,
        Candle,
        ExchangeParams,
        Order,
        OrderSource,
        Position,
        Side,
        SymbolState,
        TrendSignal,
        TrendRegime,
    )

    seen_prices: list[float] = []

    class _S(IStrategy):
        def populate_indicators(self, df, m):
            return df

        def populate_entry_trend(self, df, m):
            return df

        def populate_exit_trend(self, df, m):
            return df

        def confirm_trade_exit(self, ctx, qty, price, reason):
            seen_prices.append(price)
            return True

    runner = StrategyRunner(_S())
    pos = Position(size=0.01, entry_price=50_000.0)
    ss = SymbolState(symbol="BTC", position_long=pos)
    acc = AccountState(balance=10_000)
    acc.symbols["BTC"] = ss
    current = 50_400.0
    ctx = TradeContext(
        symbol="BTC",
        side=Side.LONG,
        position=pos,
        account=acc,
        candle=Candle(0, current, current, current, current, 0),
        signal=TrendSignal(direction=0, strength=0, regime=TrendRegime.NEUTRAL),
        current_time_ms=0,
        exchange_params=ExchangeParams(),
    )
    # Strategy custom_stoploss would build a market close at SL threshold.
    sl_threshold = 48_500.0
    market_exit = Order(
        symbol="BTC",
        side=Side.LONG,
        price=sl_threshold,
        qty=0.01,
        source=OrderSource.RISK,
        reduce_only=True,
        is_market=True,
    )
    runner.filter_exits([market_exit], ctx)
    assert seen_prices == [pytest.approx(current)], (
        f"confirm_trade_exit must see ctx.current_price={current} for "
        f"market exits, not the SL threshold; saw {seen_prices}"
    )
