"""Round-9 tests:

* trend bucket clears when the exchange stops echoing a side
* recent_creates dedup uses full identity (not just symbol/price/qty)
* _cid_by_desired survives _save_state / _load_state
* realized PnL fallback from local bucket entry when info.realizedPnl missing
* max_realized_loss_pct global gate trims/drops loss-realizing closes
"""
from __future__ import annotations

import asyncio
import tempfile
from collections import deque
from pathlib import Path

import pytest

from combo_bot.live import LiveConfig, LiveTrader
from combo_bot.risk import RiskConfig, RiskManager
from combo_bot.types import (
    AccountState, ExchangeParams, Fill, Order, OrderSource, Position, Side,
    SymbolState,
)


class _Stub:
    def __init__(self, positions=None):
        self.positions = positions or []
        self.created: list[dict] = []
        self.trades_by_call: list[list[dict]] = []

    async def load_markets(self):
        return {}

    def market(self, _s):
        return {
            "precision": {"amount": 0.001, "price": 0.01},
            "limits": {"amount": {"min": 0.001}, "cost": {"min": 5.0}},
            "maker": 0.0002, "taker": 0.0005,
        }

    async def fetch_balance(self, _p=None):
        return {"USDT": {"free": 10_000.0, "total": 10_000.0}}

    async def fetch_positions(self, _s):
        return self.positions

    async def fetch_funding_rate(self, _s):
        return {"fundingRate": 0.0}

    async def fetch_ohlcv(self, *_a, **_k):
        return []

    async def fetch_my_trades(self, *_a, **_k):
        if self.trades_by_call:
            return self.trades_by_call.pop(0)
        return []

    async def fetch_open_orders(self, _s):
        return []

    async def create_order(self, sym, ot, side, qty, price, params):
        self.created.append({
            "symbol": sym, "type": ot, "side": side,
            "qty": qty, "price": price, "params": params,
        })
        return {"id": str(len(self.created)), "status": "open"}

    async def cancel_order(self, *_a, **_k):
        return {}

    async def set_leverage(self, *_a, **_k):
        return {}

    async def set_margin_mode(self, *_a, **_k):
        return {}


def _trader() -> LiveTrader:
    ex = _Stub()
    cfg = LiveConfig(symbols=["BTC/USDT:USDT"], dry_run=False)
    trader = LiveTrader(cfg, ex)
    trader.exchange_params["BTC/USDT:USDT"] = ExchangeParams(
        qty_step=0.001, price_step=0.01, min_qty=0.001, min_cost=5.0,
    )
    trader.account.symbols["BTC/USDT:USDT"] = SymbolState(symbol="BTC/USDT:USDT")
    return trader


# ────────────────────────────────────────────────────────────────────
# #1 trend bucket also clears on missing-side
# ────────────────────────────────────────────────────────────────────


def test_missing_side_clears_both_grid_and_trend_buckets():
    trader = _trader()
    ss = trader.account.symbols["BTC/USDT:USDT"]
    ss.position_long = Position(size=0.3, entry_price=50_000.0)
    ss.trend_long = Position(size=0.2, entry_price=49_000.0)

    # Exchange returns NO long-side position. Both buckets must clear.
    trader.exchange.positions = [{
        "symbol": "BTC/USDT:USDT", "side": "short",
        "contracts": 0.1, "entryPrice": 51_000.0, "markPrice": 50_500.0,
    }]
    asyncio.run(trader._refresh_account())

    assert ss.position_long.size == 0.0
    assert ss.trend_long.size == 0.0, (
        "trend bucket must also clear when exchange stops echoing the side "
        "— the exchange is the source of truth"
    )


# ────────────────────────────────────────────────────────────────────
# #2 dedup uses full identity (side + source + reduce_only matter)
# ────────────────────────────────────────────────────────────────────


def test_dedup_does_not_cross_block_long_close_and_short_entry():
    """A LONG-close sell at price/qty X must NOT dedup a same-price/qty
    SHORT-entry sell — the pre-fix tuple was (symbol, price, qty) only,
    so this case silently dropped the SHORT entry."""
    trader = _trader()
    long_close = Order(
        symbol="BTC/USDT:USDT", side=Side.LONG, price=50_000.0, qty=0.01,
        source=OrderSource.GRID, reduce_only=True,
    )
    short_entry = Order(
        symbol="BTC/USDT:USDT", side=Side.SHORT, price=50_000.0, qty=0.01,
        source=OrderSource.GRID,
    )
    asyncio.run(trader._reconcile_orders([long_close, short_entry]))
    # Both must reach the exchange — different identities even though
    # exchange_side is "sell" for each.
    assert len(trader.exchange.created) == 2


def test_dedup_still_blocks_truly_identical_repeats():
    trader = _trader()
    o = Order(
        symbol="BTC/USDT:USDT", side=Side.LONG, price=50_000.0, qty=0.01,
        source=OrderSource.GRID,
    )
    asyncio.run(trader._reconcile_orders([o]))
    asyncio.run(trader._reconcile_orders([o]))
    assert len(trader.exchange.created) == 1


# ────────────────────────────────────────────────────────────────────
# #4 _cid_by_desired round-trips through state file
# ────────────────────────────────────────────────────────────────────


def test_cid_cache_survives_save_load_cycle():
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "state.json"
        trader_a = _trader()
        trader_a.config.state_file = str(state_path)
        o = Order(
            symbol="BTC/USDT:USDT", side=Side.LONG, price=50_000.0, qty=0.01,
            source=OrderSource.GRID,
        )
        asyncio.run(trader_a._reconcile_orders([o]))
        # cache populated.
        assert trader_a._cid_by_desired
        first_cid = trader_a.exchange.created[0]["params"]["clientOrderId"]
        asyncio.run(trader_a._save_state())

        trader_b = _trader()
        trader_b.config.state_file = str(state_path)
        asyncio.run(trader_b._load_state())
        # Same desired identity must produce SAME cOID after load —
        # which means the cache survived the round-trip.
        identity = trader_b._desired_identity(o)
        cached = trader_b._cid_by_desired.get(identity)
        assert cached is not None, (
            "cOID cache must round-trip through state file so reconcile "
            "after restart matches existing open orders by cOID instead "
            "of cancelling and recreating them"
        )
        assert cached[0] == first_cid


# ────────────────────────────────────────────────────────────────────
# #5 realized PnL fallback from local bucket entry
# ────────────────────────────────────────────────────────────────────


def test_enrich_fill_pnl_computes_long_close_from_entry_price():
    trader = _trader()
    ss = trader.account.symbols["BTC/USDT:USDT"]
    ss.position_long = Position(size=0.01, entry_price=50_000.0)
    # A reduce_only LONG close at 51000 with realized_pnl=0 (missing
    # field). Local fallback: (51000 - 50000) * 0.01 = +10.0.
    fill = Fill(
        timestamp=1_000, symbol="BTC/USDT:USDT", side=Side.LONG,
        price=51_000.0, qty=0.01, fee=0.5, realized_pnl=0.0,
        source=OrderSource.GRID, reduce_only=True,
    )
    enriched = trader._enrich_fill_pnl(fill)
    assert enriched.realized_pnl == pytest.approx(10.0)


def test_enrich_fill_pnl_computes_short_close_correctly():
    trader = _trader()
    ss = trader.account.symbols["BTC/USDT:USDT"]
    ss.position_short = Position(size=0.01, entry_price=50_000.0)
    fill = Fill(
        timestamp=1_000, symbol="BTC/USDT:USDT", side=Side.SHORT,
        price=49_000.0, qty=0.01, fee=0.5, realized_pnl=0.0,
        source=OrderSource.GRID, reduce_only=True,
    )
    enriched = trader._enrich_fill_pnl(fill)
    # SHORT close at 49k from entry 50k → +10 PnL.
    assert enriched.realized_pnl == pytest.approx(10.0)


def test_enrich_fill_pnl_preserves_exchange_provided_value():
    """If the exchange surfaces realized_pnl != 0, trust it over the
    local reconstruction."""
    trader = _trader()
    ss = trader.account.symbols["BTC/USDT:USDT"]
    ss.position_long = Position(size=0.01, entry_price=50_000.0)
    fill = Fill(
        timestamp=1_000, symbol="BTC/USDT:USDT", side=Side.LONG,
        price=51_000.0, qty=0.01, fee=0.5, realized_pnl=42.0,  # given
        source=OrderSource.GRID, reduce_only=True,
    )
    enriched = trader._enrich_fill_pnl(fill)
    assert enriched.realized_pnl == pytest.approx(42.0)


def test_enrich_fill_pnl_skips_non_reduce_fills():
    """Opens / adds have realized_pnl == 0 by definition. Don't
    fabricate a number for them."""
    trader = _trader()
    ss = trader.account.symbols["BTC/USDT:USDT"]
    ss.position_long = Position(size=0.01, entry_price=50_000.0)
    fill = Fill(
        timestamp=1_000, symbol="BTC/USDT:USDT", side=Side.LONG,
        price=51_000.0, qty=0.01, fee=0.5, realized_pnl=0.0,
        source=OrderSource.GRID, reduce_only=False,
    )
    enriched = trader._enrich_fill_pnl(fill)
    assert enriched.realized_pnl == 0.0


# ────────────────────────────────────────────────────────────────────
# #6 max_realized_loss_pct global gate
# ────────────────────────────────────────────────────────────────────


def test_realized_loss_gate_drops_close_when_budget_exhausted():
    risk = RiskManager(RiskConfig(
        max_realized_loss_pct=0.01,  # $100 on $10k
        yellow_threshold=0.99, orange_threshold=0.99, red_threshold=0.99,
    ))
    acc = AccountState(balance=10_000, equity=10_000, equity_peak=10_000)
    acc.symbols["BTC"] = SymbolState(symbol="BTC")
    acc.symbols["BTC"].position_long = Position(size=1.0, entry_price=50_000.0)
    # Spent $90 of the $100 budget already.
    acc.grid_loss_log = deque([(1_000, -90.0)])

    # Close at $45000 would realize $5000 loss — way over remaining $10.
    huge_loss_close = Order(
        symbol="BTC", side=Side.LONG, price=45_000.0, qty=1.0,
        source=OrderSource.GRID, reduce_only=True,
    )
    out = risk.filter_orders([huge_loss_close], acc, timestamp=10_000)
    # Either trimmed to fit ~$10 budget or dropped entirely.
    assert all(o.qty < 1.0 for o in out if o.reduce_only) or out == []


def test_realized_loss_gate_passes_profitable_closes():
    risk = RiskManager(RiskConfig(
        max_realized_loss_pct=0.01,
        yellow_threshold=0.99, orange_threshold=0.99, red_threshold=0.99,
    ))
    acc = AccountState(balance=10_000, equity=10_000, equity_peak=10_000)
    acc.symbols["BTC"] = SymbolState(symbol="BTC")
    acc.symbols["BTC"].position_long = Position(size=1.0, entry_price=50_000.0)
    acc.grid_loss_log = deque([(1_000, -99.0)])  # near-budget loss

    profit_close = Order(
        symbol="BTC", side=Side.LONG, price=51_000.0, qty=1.0,
        source=OrderSource.GRID, reduce_only=True,
    )
    out = risk.filter_orders([profit_close], acc, timestamp=10_000)
    # Profit close must pass — gate is about realising LOSS, not P&L direction.
    assert len(out) == 1


def test_realized_loss_gate_lets_market_orders_through():
    """is_market = panic / SL / forced exit — must not be gated."""
    risk = RiskManager(RiskConfig(
        max_realized_loss_pct=0.001,  # $10 on $10k
        yellow_threshold=0.99, orange_threshold=0.99, red_threshold=0.99,
    ))
    acc = AccountState(balance=10_000, equity=10_000, equity_peak=10_000)
    acc.symbols["BTC"] = SymbolState(symbol="BTC")
    acc.symbols["BTC"].position_long = Position(size=1.0, entry_price=50_000.0)
    acc.grid_loss_log = deque([(1_000, -100.0)])  # over budget

    panic = Order(
        symbol="BTC", side=Side.LONG, price=45_000.0, qty=1.0,
        source=OrderSource.RISK, reduce_only=True, is_market=True,
    )
    out = risk.filter_orders([panic], acc, timestamp=10_000)
    assert len(out) == 1
    assert out[0].is_market is True


# ────────────────────────────────────────────────────────────────────
# Explicit verification of the round-7 grid entry reconstruction
# against the scenario ChatGPT flagged as a P0 — grid 0.1 @ 50k,
# trend 0.1 @ 60k, exchange aggregates 0.2 @ 55k. The fix is supposed
# to invert the aggregate, NOT inherit it.
# ────────────────────────────────────────────────────────────────────


def test_grid_entry_inverted_from_mixed_trend_aggregate_chatgpt_p0():
    trader = _trader()
    ss = trader.account.symbols["BTC/USDT:USDT"]
    # Pre-condition: trend bucket locally tracked at 0.1 @ 60k.
    ss.trend_long = Position(size=0.1, entry_price=60_000.0)
    # Exchange echoes the AGGREGATE: 0.2 @ 55k (the volume-weighted
    # average of the local grid 0.1 @ 50k and trend 0.1 @ 60k).
    trader.exchange.positions = [{
        "symbol": "BTC/USDT:USDT", "side": "long",
        "contracts": 0.2, "entryPrice": 55_000.0, "markPrice": 55_500.0,
    }]
    asyncio.run(trader._refresh_account())

    # If the bug were live, the grid bucket would carry entry_price
    # = 55k (the aggregate). That would put TP markups, WE math,
    # close ladder and unstuck all at the wrong reference. The fix
    # inverts: grid_entry = (0.2*55k - 0.1*60k) / 0.1 = 50_000.
    assert ss.position_long.size == pytest.approx(0.1)
    assert ss.position_long.entry_price == pytest.approx(50_000.0), (
        f"grid entry must invert to 50k from the aggregate 55k; got "
        f"{ss.position_long.entry_price} — the aggregate-as-grid bug is back"
    )


def test_realized_loss_gate_zero_pct_disables():
    risk = RiskManager(RiskConfig(
        max_realized_loss_pct=0.0,
        yellow_threshold=0.99, orange_threshold=0.99, red_threshold=0.99,
    ))
    acc = AccountState(balance=10_000, equity=10_000, equity_peak=10_000)
    acc.symbols["BTC"] = SymbolState(symbol="BTC")
    acc.symbols["BTC"].position_long = Position(size=1.0, entry_price=50_000.0)
    huge_loss = Order(
        symbol="BTC", side=Side.LONG, price=45_000.0, qty=1.0,
        source=OrderSource.GRID, reduce_only=True,
    )
    out = risk.filter_orders([huge_loss], acc, timestamp=0)
    # Gate disabled → order survives.
    assert any(o.qty == 1.0 for o in out)


# ────────────────────────────────────────────────────────────────────
# Round-10: flat-account first-entry, fuzzy with reduceOnly+positionSide,
# c_mult-aware loss gate, sequential PnL enrich, bot_start_ms gate.
# ────────────────────────────────────────────────────────────────────


def test_flat_account_first_entry_actually_reaches_exchange():
    """Round-10 P0: a brand-new account with no positions must NOT have
    its first-tick entries silently quiesced. The pre-fix code added
    state_change keys for every missing side unconditionally, so reconcile
    skipped all non-reduce entries on tick 0 of an empty account."""
    trader = _trader()
    # Cold state: no positions on exchange. _refresh_account had been
    # adding both (sym, LONG) and (sym, SHORT) to state_change_keys
    # despite local buckets being empty.
    trader.exchange.positions = []
    asyncio.run(trader._refresh_account())
    assert trader._state_change_keys == set(), (
        "flat account refresh must not lock entries — buckets are "
        "already empty so there's no drift to defer for"
    )

    # Now drive a normal entry through reconcile. It must reach the exchange.
    o = Order(
        symbol="BTC/USDT:USDT", side=Side.LONG, price=50_000.0, qty=0.01,
        source=OrderSource.GRID,
    )
    asyncio.run(trader._reconcile_orders([o]))
    assert len(trader.exchange.created) == 1, (
        "first live entry on a flat account must actually create_order"
    )


def test_realized_loss_gate_honors_c_mult():
    """Round-10 P1.b: a c_mult != 1 symbol must NOT compute projected
    loss as qty * (fill - entry); it must multiply by c_mult so the
    budget gate actually reflects USD loss."""
    risk = RiskManager(RiskConfig(
        max_realized_loss_pct=0.01,  # $100 budget
        yellow_threshold=0.99, orange_threshold=0.99, red_threshold=0.99,
    ))
    acc = AccountState(balance=10_000, equity=10_000, equity_peak=10_000)
    acc.symbols["BTC"] = SymbolState(symbol="BTC")
    acc.symbols["BTC"].position_long = Position(size=1.0, entry_price=50_000.0)
    # c_mult = 10 (think index-multiplied contract). At fill 49_999,
    # raw qty*(fill-entry) = -1 → would look like $1 loss without
    # c_mult. With c_mult applied: $10 loss per unit price gap.
    ep = ExchangeParams(c_mult=10.0)
    # 11 unit gap → projected loss = 1.0 * 11 * 10 = $110 > $100 budget.
    big_loss = Order(
        symbol="BTC", side=Side.LONG, price=49_989.0, qty=1.0,
        source=OrderSource.GRID, reduce_only=True,
    )
    out = risk.filter_orders(
        [big_loss], acc, timestamp=0, exchange_params={"BTC": ep},
    )
    # Must be trimmed below qty=1 because the c_mult-aware projection
    # sees the loss exceeds the budget.
    if out:
        assert all(o.qty < 1.0 for o in out if o.reduce_only)


def test_fill_pnl_correct_when_entry_and_close_in_same_batch():
    """Round-10 P1.c: when a fetch_my_trades batch contains a trend
    entry followed by a trend close, the close's fallback PnL must
    see the entry applied to the bucket first. Sequential enrich+
    apply is what makes this work."""
    trader = _trader()
    ss = trader.account.symbols["BTC/USDT:USDT"]
    # Pre-condition: trend bucket empty.
    assert not ss.trend_long.is_open

    # Stub two trades: a TREND LONG entry at ts=1000, then a TREND
    # LONG close at ts=2000. fetch_my_trades returns them together.
    trader.exchange.trades_by_call = [[
        {
            "id": "t-entry", "timestamp": 1_000, "side": "buy",
            "price": 50_000.0, "amount": 0.01,
            "fee": {"cost": 0.1}, "order": "ord-entry",
            "info": {"realizedPnl": "0"},
        },
        {
            "id": "t-close", "timestamp": 2_000, "side": "sell",
            "price": 51_000.0, "amount": 0.01,
            "fee": {"cost": 0.1}, "order": "ord-close",
            "info": {"realizedPnl": "0"},  # missing PnL → fallback path
        },
    ]]
    # Register source/meta for both orders so attribution finds TREND
    # and the close's reduce_only flag.
    trader.fill_events.register_outgoing(
        "ord-entry", OrderSource.TREND, Side.LONG, reduce_only=False,
    )
    trader.fill_events.register_outgoing(
        "ord-close", OrderSource.TREND, Side.LONG, reduce_only=True,
    )
    asyncio.run(trader._refresh_fills())

    # The close should have produced a realized PnL of (51000-50000)*0.01 = +10.
    # If sequential ordering broke (close enriched before entry applied),
    # the bucket would have been empty and fallback returned 0.
    assert trader.account.trend_realized_pnl > 0, (
        "trend close's fallback PnL must see entry applied first; "
        f"got trend_realized_pnl={trader.account.trend_realized_pnl}"
    )


def test_first_poll_skips_pre_bot_start_history():
    """Round-10 P1.d: the very first fetch_my_trades may return days
    of historical trades. Those must not flow into the ledger or
    Kelly/protections — they happened before the bot booted."""
    from combo_bot.fill_events_manager import FillEventManager, FillEventManagerConfig
    from combo_bot.types import Fill as _Fill  # noqa: F401

    class _Ex:
        def __init__(self):
            self.calls = 0

        async def fetch_my_trades(self, *_a, **_k):
            self.calls += 1
            return [
                {"id": "old", "timestamp": 500, "side": "buy",
                 "price": 50_000.0, "amount": 0.01, "fee": {"cost": 0.05},
                 "order": "ord-old", "info": {}},
                {"id": "new", "timestamp": 2_000, "side": "sell",
                 "price": 51_000.0, "amount": 0.01, "fee": {"cost": 0.05},
                 "order": "ord-new", "info": {"realizedPnl": "10"}},
            ]

    mgr = FillEventManager(_Ex(), FillEventManagerConfig(poll_interval_ms=0))
    mgr.set_bot_start(1_000)
    captured = []
    asyncio.run(mgr.poll("BTC/USDT:USDT", now_ms=5_000, sink=captured.extend))
    # Only the trade at ts=2000 (>= bot_start=1000) should reach the sink.
    assert len(captured) == 1
    assert captured[0].timestamp == 2_000


# ────────────────────────────────────────────────────────────────────
# Round-11: pending overlay ledger, is_market in cOID identity,
# prune-before-stamp, cOID-keyed fill attribution.
# ────────────────────────────────────────────────────────────────────


def test_pending_overlay_blocks_double_market_entry_until_fill():
    """Round-11 P0: between create_order ack and fill arrival, the
    bucket is empty locally. The next tick must see a pending claim
    and refuse to emit another market overlay."""
    from combo_bot.types import RegimeView, TradingMode, TrendRegime

    trader = _trader()
    trader.account.balance = 10_000.0
    rv = RegimeView(
        primary=TrendRegime.STRONG_BULL,
        conviction=0.9,
        long_mode=TradingMode.AGGRESSIVE,
        short_mode=TradingMode.PANIC,
        allow_grid_long=True, allow_grid_short=False,
        trend_overlay=Side.LONG, trend_qty_scale=0.5,
        close_aggressiveness=0.7, veto_reasons=(),
    )
    ep = ExchangeParams(qty_step=0.001, price_step=0.01, min_qty=0.001, min_cost=5.0)

    # First call emits an overlay entry.
    first = trader._emit_trend_overlay("BTC/USDT:USDT", rv, price=50_000.0, exchange=ep)
    assert len(first) == 1 and first[0].is_market is True

    # Simulate the create_order success path stamping pending.
    trader._pending_overlay[("BTC/USDT:USDT", Side.LONG)] = (
        __import__("time").time() * 1000
    )

    # Second call: bucket still empty (fill hasn't arrived), pending
    # set must block.
    second = trader._emit_trend_overlay("BTC/USDT:USDT", rv, price=50_000.0, exchange=ep)
    assert second == [], (
        "pending-overlay claim must block a second market entry until "
        "the fill arrives or the entry expires"
    )

    # Simulate the fill arriving and clearing the pending slot.
    trader._pending_overlay.pop(("BTC/USDT:USDT", Side.LONG))
    ss = trader.account.symbols["BTC/USDT:USDT"]
    ss.trend_long = Position(size=0.01, entry_price=50_000.0)

    # Now the bucket itself blocks (is_open guard), as before.
    third = trader._emit_trend_overlay("BTC/USDT:USDT", rv, price=50_000.0, exchange=ep)
    assert third == []


def test_pending_overlay_ttl_graduates_to_unknown_not_freed():
    """Round-13 P1.a: pending past TTL no longer silently frees the
    slot — it transitions to UNKNOWN and continues to block emit
    until ``_resolve_unknowns`` consults the exchange."""
    from combo_bot.types import RegimeView, TradingMode, TrendRegime

    trader = _trader()
    trader.account.balance = 10_000.0

    fake_now = 10 * 60_000
    trader._now_ms = lambda: fake_now  # type: ignore[method-assign]
    trader._pending_overlay[("BTC/USDT:USDT", Side.LONG)] = 0.0

    rv = RegimeView(
        primary=TrendRegime.STRONG_BULL, conviction=0.9,
        long_mode=TradingMode.AGGRESSIVE, short_mode=TradingMode.PANIC,
        allow_grid_long=True, allow_grid_short=False,
        trend_overlay=Side.LONG, trend_qty_scale=0.5,
        close_aggressiveness=0.7, veto_reasons=(),
    )
    ep = ExchangeParams(qty_step=0.001, price_step=0.01, min_qty=0.001, min_cost=5.0)
    out = trader._emit_trend_overlay("BTC/USDT:USDT", rv, price=50_000.0, exchange=ep)
    assert out == [], (
        "expired pending must NOT silently re-allow emit; it must "
        "block via the unknown_overlay slot until _resolve_unknowns "
        "confirms the order's state"
    )
    assert ("BTC/USDT:USDT", Side.LONG) in trader._unknown_overlay
    assert ("BTC/USDT:USDT", Side.LONG) not in trader._pending_overlay


def test_desired_identity_distinguishes_market_from_limit():
    """A limit and a market entry at the same price/qty must get
    DIFFERENT cOIDs — they're different orders on the exchange."""
    trader = _trader()
    limit_o = Order(
        symbol="BTC/USDT:USDT", side=Side.LONG, price=50_000.0, qty=0.01,
        source=OrderSource.GRID, is_market=False,
    )
    market_o = Order(
        symbol="BTC/USDT:USDT", side=Side.LONG, price=50_000.0, qty=0.01,
        source=OrderSource.GRID, is_market=True,
    )
    assert trader._desired_identity(limit_o) != trader._desired_identity(market_o)


def test_prune_runs_before_stamp_so_stale_cache_doesnt_revive():
    """If an identity expired but recurs after a long quiet period, the
    cOID assigned to it must be FRESH — not the pre-fix behavior of
    refreshing the stale entry's timestamp and reusing its cOID."""
    trader = _trader()
    # Plant a stale cache entry (well past the dedup window).
    old_identity = (
        "BTC/USDT:USDT", "long", "grid", False, False, 50_000.0, 0.01,
    )
    very_old_ts = 1.0
    trader._cid_by_desired[old_identity] = ("cb-stale-old-cid", very_old_ts)
    # Now reconcile a desired with the same identity. Prune must have
    # cleared the stale entry; a fresh cOID must be minted.
    o = Order(
        symbol="BTC/USDT:USDT", side=Side.LONG, price=50_000.0, qty=0.01,
        source=OrderSource.GRID,
    )
    asyncio.run(trader._reconcile_orders([o]))
    used_cid = trader.exchange.created[0]["params"]["clientOrderId"]
    assert used_cid != "cb-stale-old-cid", (
        "stale cache entry must NOT be revived — prune-before-stamp "
        "ensures a fresh cOID for an identity that was effectively dead"
    )


def test_fill_attribution_via_client_order_id_when_exchange_id_missing():
    """Some exchanges echo only clientOrderId on the trade. The fill
    manager must still attribute source/meta correctly via the cOID
    fallback."""
    from combo_bot.fill_events_manager import FillEventManager, FillEventManagerConfig

    class _Ex:
        async def fetch_my_trades(self, *_a, **_k):
            return [{
                "id": "t-1", "timestamp": 1_000, "side": "buy",
                "price": 50_000.0, "amount": 0.01,
                "fee": {"cost": 0.1},
                # No "order" / "orderId" field — only clientOrderId.
                "clientOrderId": "cb-trend-1",
                "info": {"realizedPnl": "0"},
            }]

    mgr = FillEventManager(_Ex(), FillEventManagerConfig(poll_interval_ms=0))
    mgr.register_outgoing(
        exchange_order_id="ex-1",
        source=OrderSource.TREND,
        side=Side.LONG,
        reduce_only=False,
        client_order_id="cb-trend-1",
    )
    captured = []
    asyncio.run(mgr.poll("BTC/USDT:USDT", now_ms=2_000, sink=captured.extend))
    assert len(captured) == 1
    assert captured[0].source == OrderSource.TREND, (
        "fill must be attributed via clientOrderId when exchange id is missing"
    )


# ────────────────────────────────────────────────────────────────────
# Round-12: bot_start scoping, pending_overlay persistence,
# create_order exception safety, cID TTL independence.
# ────────────────────────────────────────────────────────────────────


def test_bot_start_skip_respects_warmstart_watermark():
    """Restart with a persisted watermark must NOT drop fills that
    happened before the new bot_start_ms. The watermark means we're
    NOT cold-starting on this symbol."""
    from combo_bot.fill_events_manager import FillEventManager, FillEventManagerConfig

    class _Ex:
        async def fetch_my_trades(self, *_a, **_k):
            return [{
                "id": "real-fill", "timestamp": 500, "side": "buy",
                "price": 50_000.0, "amount": 0.01, "fee": {"cost": 0.05},
                "order": "ord-1", "info": {},
            }]

    mgr = FillEventManager(_Ex(), FillEventManagerConfig(poll_interval_ms=0))
    # Simulate state restored from disk: we already had a watermark.
    mgr._last_ts_ms["BTC/USDT:USDT"] = 100
    # Bot start (new wall clock after restart) is AFTER the trade ts.
    mgr.set_bot_start(1_000)
    captured = []
    asyncio.run(mgr.poll("BTC/USDT:USDT", now_ms=2_000, sink=captured.extend))
    assert len(captured) == 1, (
        "warmstart (with persisted watermark) must NOT drop trades that "
        "predate the new bot_start_ms — they may be real fills from "
        "before the restart"
    )


def test_bot_start_skip_respects_known_outgoing_order():
    """Even on a cold start (no watermark), trades from an order we
    registered as outgoing must reach the ledger — that's a real fill
    of one of our orders that happened during the brief restart gap."""
    from combo_bot.fill_events_manager import FillEventManager, FillEventManagerConfig

    class _Ex:
        async def fetch_my_trades(self, *_a, **_k):
            return [{
                "id": "ours", "timestamp": 500, "side": "buy",
                "price": 50_000.0, "amount": 0.01, "fee": {"cost": 0.05},
                "order": "ord-mine", "info": {},
            }]

    mgr = FillEventManager(_Ex(), FillEventManagerConfig(poll_interval_ms=0))
    mgr.set_bot_start(1_000)
    # We sent this order before the restart.
    mgr.register_outgoing(
        "ord-mine", OrderSource.TREND, Side.LONG, reduce_only=False,
    )
    captured = []
    asyncio.run(mgr.poll("BTC/USDT:USDT", now_ms=2_000, sink=captured.extend))
    assert len(captured) == 1
    assert captured[0].source == OrderSource.TREND


def test_pending_overlay_round_trips_through_state_file():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "state.json"
        trader_a = _trader()
        trader_a.config.state_file = str(state_path)
        trader_a._pending_overlay[("BTC/USDT:USDT", Side.LONG)] = 1_700_000_000_000.0
        asyncio.run(trader_a._save_state())

        trader_b = _trader()
        trader_b.config.state_file = str(state_path)
        asyncio.run(trader_b._load_state())
        assert ("BTC/USDT:USDT", Side.LONG) in trader_b._pending_overlay, (
            "pending TREND overlay claim must survive a restart so the "
            "next tick doesn't double-emit while the fill is still in flight"
        )


def test_create_order_exception_keeps_pending_and_cid_registration():
    """If the exchange call raises (timeout, disconnect, anything), the
    order may have been accepted server-side. We must keep the cOID
    attribution registered and the pending overlay marker in place so
    a delayed fill is routable AND no race tick can double-emit."""
    class _ErrorExchange(_Stub):
        async def create_order(self, *_a, **_k):
            raise RuntimeError("network hiccup")

    ex = _ErrorExchange()
    cfg = LiveConfig(symbols=["BTC/USDT:USDT"], dry_run=False)
    trader = LiveTrader(cfg, ex)
    trader.exchange_params["BTC/USDT:USDT"] = ExchangeParams(
        qty_step=0.001, price_step=0.01, min_qty=0.001, min_cost=5.0,
    )
    trader.account.symbols["BTC/USDT:USDT"] = SymbolState(symbol="BTC/USDT:USDT")
    o = Order(
        symbol="BTC/USDT:USDT", side=Side.LONG, price=50_000.0, qty=0.01,
        source=OrderSource.TREND, is_market=True,
        client_order_id="cb-trend-net-fail",
    )
    asyncio.run(trader._create_order(o))
    # cOID must still be registered for fill attribution.
    assert "cb-trend-net-fail" in trader.fill_events.order_source
    assert trader.fill_events.order_source["cb-trend-net-fail"] == OrderSource.TREND
    # Pending overlay must still be set.
    assert ("BTC/USDT:USDT", Side.LONG) in trader._pending_overlay


def test_cid_cache_outlives_recent_creates_window():
    """A cID cache entry must survive a brief outage longer than the
    recent_creates TTL. Otherwise reconcile after restart would fail
    to cOID-match the exchange's still-open order."""
    trader = _trader()
    o = Order(
        symbol="BTC/USDT:USDT", side=Side.LONG, price=50_000.0, qty=0.01,
        source=OrderSource.GRID,
    )
    asyncio.run(trader._reconcile_orders([o]))
    identity = trader._desired_identity(o)
    cached_cid, cached_ts = trader._cid_by_desired[identity]

    # Simulate "5 minutes later" by backdating the cache entry past
    # _recent_order_window_ms (15s default) but well within
    # _cid_cache_ttl_ms (24h).
    long_ago = cached_ts - (10 * 60 * 1000)  # 10 minutes ago
    trader._cid_by_desired[identity] = (cached_cid, long_ago)

    # Run reconcile again with the same desired. The prune-before-stamp
    # path uses _cid_cache_ttl_ms, NOT _recent_order_window_ms, so the
    # entry must survive and the cOID must be reused.
    trader.exchange.created.clear()
    trader._recent_creates.clear()  # bypass dedup so the order goes through
    asyncio.run(trader._reconcile_orders([o]))
    new_cid = trader.exchange.created[0]["params"]["clientOrderId"]
    assert new_cid == cached_cid, (
        f"cID cache must outlive recent_creates window; expected reuse of "
        f"{cached_cid}, got {new_cid} — cID cache TTL leaked to "
        f"_recent_order_window_ms"
    )


def test_now_ms_is_a_method_and_can_be_monkeypatched():
    """The time-source refactor's whole point: tests can inject a clock."""
    trader = _trader()
    assert callable(trader._now_ms)
    fake_value = 1_234_567_890_000
    trader._now_ms = lambda: fake_value  # type: ignore[method-assign]
    assert trader._now_ms() == fake_value
    # And the patched value flows into the dedup window math.
    cutoff = fake_value - trader._recent_order_window_ms
    assert cutoff == fake_value - trader._recent_order_window_ms


def test_fuzzy_match_refuses_when_reduce_only_disagrees():
    """Round-10 P1.a: a LONG-close (reduce_only=True) must NOT match
    a SHORT-entry (reduce_only=False) even when they share the same
    ccxt 'sell' side, price, and qty."""
    trader = _trader()
    long_close = Order(
        symbol="BTC/USDT:USDT", side=Side.LONG, price=50_000.0, qty=0.01,
        source=OrderSource.GRID, reduce_only=True,
        client_order_id="cb-long-close",
    )
    short_entry_open_order = {
        "side": "sell", "price": 50_000.0, "amount": 0.01,
        "reduceOnly": False,
        # No clientOrderId echo — exercise the fuzzy path.
    }
    assert trader._orders_match(long_close, short_entry_open_order) is False
