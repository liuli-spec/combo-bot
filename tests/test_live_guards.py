"""Regression tests for the live-trader execution guards.

Covers the bugs fixed in this round:
  1. dedup key must include symbol — two symbols with coincident
     (price, qty) must NOT cross-block each other.
  2. RiskTier round-trips through save/load as an enum, not a bare str.
  3. _state_change_keys is per (symbol, side) — a long-side fill must
     not suppress fresh short-side entries on the same symbol.
  4. The dedup window auto-sizes to >= 2 * loop_interval so consecutive
     ticks actually see each other.
"""
from __future__ import annotations
import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from combo_bot.live import LiveConfig, LiveTrader
from combo_bot.regime import read_strategy_signals
from combo_bot.risk import RiskTier
from combo_bot.strategy import DefaultStrategy
from combo_bot.types import (
    Candle, ExchangeParams, Order, OrderSource, Position, Side, SymbolState,
)


class _StubExchange:
    """Minimal async-shaped stub for the reconciler tests."""

    def __init__(self):
        self.created: list[dict] = []
        self.cancelled: list[tuple[str, str]] = []
        self.open_orders_by_symbol: dict[str, list[dict]] = {}
        self.next_status: str = "open"
        self.next_id: int = 0
        self.balance_payload: dict = {"USDT": {"free": 10000.0}}
        self.trades_per_call: list[list[dict]] = []

    async def load_markets(self):
        return {}

    def market(self, _symbol):
        return {
            "precision": {"amount": 0.001, "price": 0.01},
            "limits": {"amount": {"min": 0.001}, "cost": {"min": 5.0}},
            "maker": 0.0002,
            "taker": 0.0005,
        }

    async def fetch_open_orders(self, symbol):
        return self.open_orders_by_symbol.get(symbol, [])

    async def fetch_balance(self, _params=None):
        return self.balance_payload

    async def fetch_positions(self, _symbols):
        return []

    async def fetch_funding_rate(self, _symbol):
        return {"fundingRate": 0.0}

    async def fetch_ohlcv(self, _symbol, _tf, limit=100):
        return []

    async def fetch_my_trades(self, symbol, since=None, limit=None):
        if self.trades_per_call:
            return self.trades_per_call.pop(0)
        return []

    async def create_order(self, symbol, order_type, side, qty, price, params):
        self.next_id += 1
        self.created.append({
            "symbol": symbol, "type": order_type, "side": side,
            "qty": qty, "price": price, "params": params,
        })
        return {"id": str(self.next_id), "status": self.next_status}

    async def cancel_order(self, order_id, symbol):
        self.cancelled.append((symbol, order_id))
        return {}

    async def set_leverage(self, *_a, **_k):
        return {}

    async def set_margin_mode(self, *_a, **_k):
        return {}


def _trader(symbols: list[str], *, dry_run: bool = False, loop_interval: float = 60.0) -> tuple[LiveTrader, _StubExchange]:
    ex = _StubExchange()
    cfg = LiveConfig(symbols=symbols, dry_run=dry_run, loop_interval_seconds=loop_interval)
    trader = LiveTrader(cfg, ex)
    # _init_exchange would normally populate this; do it directly.
    for s in symbols:
        trader.exchange_params[s] = ExchangeParams(
            qty_step=0.001, price_step=0.01, min_qty=0.001, min_cost=5.0,
        )
        trader.account.symbols[s] = SymbolState(symbol=s, last_price=50000.0)
    return trader, ex


# ── #1: dedup key must include symbol ────────────────────────────────

def test_dedup_does_not_cross_block_different_symbols():
    """Two symbols submitting the same (price, qty) must both go through."""
    trader, ex = _trader(["BTC/USDT:USDT", "ETH/USDT:USDT"])
    orders = [
        Order(symbol="BTC/USDT:USDT", side=Side.LONG, price=50000.0, qty=0.01,
              source=OrderSource.GRID),
        Order(symbol="ETH/USDT:USDT", side=Side.LONG, price=50000.0, qty=0.01,
              source=OrderSource.GRID),
    ]
    asyncio.run(trader._reconcile_orders(orders))
    sent_symbols = {c["symbol"] for c in ex.created}
    assert sent_symbols == {"BTC/USDT:USDT", "ETH/USDT:USDT"}


def test_dedup_blocks_same_symbol_repeat():
    """Repeating the exact same (symbol, price, qty) within the window is skipped."""
    trader, ex = _trader(["BTC/USDT:USDT"])
    o = Order(symbol="BTC/USDT:USDT", side=Side.LONG, price=50000.0, qty=0.01,
              source=OrderSource.GRID)
    asyncio.run(trader._reconcile_orders([o]))
    assert len(ex.created) == 1
    # Second pass with identical order — dedup should swallow it.
    asyncio.run(trader._reconcile_orders([o]))
    assert len(ex.created) == 1


# ── #2: RiskTier round-trips as enum ────────────────────────────────

def test_risk_tier_round_trips_through_state_file():
    """After load, risk.tier must be a RiskTier enum (so .value still works)."""
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "state.json"
        # Trader A: drive tier to RED via direct mutation, then save.
        ex_a = _StubExchange()
        cfg_a = LiveConfig(symbols=["BTC/USDT:USDT"], state_file=str(state_path))
        trader_a = LiveTrader(cfg_a, ex_a)
        trader_a.risk.tier = RiskTier.RED
        trader_a.risk.red_latched = True
        trader_a.risk.red_cooldown_until = 1
        trader_a.risk.dd_ema = 0.42
        asyncio.run(trader_a._save_state())

        # Trader B: cold start, then load. tier must come back as enum.
        ex_b = _StubExchange()
        cfg_b = LiveConfig(symbols=["BTC/USDT:USDT"], state_file=str(state_path))
        trader_b = LiveTrader(cfg_b, ex_b)
        asyncio.run(trader_b._load_state())

        assert isinstance(trader_b.risk.tier, RiskTier)
        assert trader_b.risk.tier == RiskTier.RED
        assert trader_b.risk.tier.value == "red"  # would AttributeError on a bare str pre-fix
        assert trader_b.risk.red_latched is True
        assert trader_b.risk.dd_ema == pytest.approx(0.42)


def test_risk_tier_load_ignores_unknown_value():
    """An unknown tier string on disk must not crash _load_state."""
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "state.json"
        state_path.write_text(json.dumps({
            "equity_peak": 1000.0,
            "risk_tier": "ultraviolet",
            "risk_red_latched": False,
        }))
        ex = _StubExchange()
        cfg = LiveConfig(symbols=["BTC/USDT:USDT"], state_file=str(state_path))
        trader = LiveTrader(cfg, ex)
        # Must not raise. Tier stays at the default (GREEN).
        asyncio.run(trader._load_state())
        assert trader.risk.tier == RiskTier.GREEN


def test_fill_event_state_round_trips_through_live_state_file():
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "state.json"
        trader_a, _ = _trader(["BTC/USDT:USDT"])
        trader_a.config.state_file = str(state_path)
        trader_a.fill_events.load_snapshot({
            "last_ts_ms": {"BTC/USDT:USDT": 1234},
            "seen_ids": {"BTC/USDT:USDT": ["fill-1"]},
            "order_source": {"ord-trend": "trend"},
            "order_meta": {
                "ord-trend": {"side": "long", "reduce_only": False},
            },
        })
        asyncio.run(trader_a._save_state())

        trader_b, _ = _trader(["BTC/USDT:USDT"])
        trader_b.config.state_file = str(state_path)
        asyncio.run(trader_b._load_state())

        snapshot = trader_b.fill_events.snapshot()
        assert snapshot["last_ts_ms"]["BTC/USDT:USDT"] == 1234
        assert snapshot["order_source"]["ord-trend"] == "trend"


# ── #3: state-change defer is per (symbol, side) ────────────────────

def test_state_change_defer_does_not_block_other_side():
    """A long-side fill drift must not suppress fresh short-side entries."""
    trader, ex = _trader(["BTC/USDT:USDT"])
    trader._state_change_keys.add(("BTC/USDT:USDT", Side.LONG))
    orders = [
        Order(symbol="BTC/USDT:USDT", side=Side.LONG, price=49000.0, qty=0.01,
              source=OrderSource.GRID),
        Order(symbol="BTC/USDT:USDT", side=Side.SHORT, price=51000.0, qty=0.01,
              source=OrderSource.GRID),
    ]
    asyncio.run(trader._reconcile_orders(orders))
    sent_sides = {c["side"] for c in ex.created}
    # SHORT entry must go through; LONG entry must be deferred.
    assert "sell" in sent_sides
    assert "buy" not in sent_sides


def test_state_change_defer_allows_reduce_only_on_blocked_side():
    """Reduce-only exits must NOT be blocked even on a quiesced (symbol, side)."""
    trader, ex = _trader(["BTC/USDT:USDT"])
    trader._state_change_keys.add(("BTC/USDT:USDT", Side.LONG))
    exit_order = Order(
        symbol="BTC/USDT:USDT", side=Side.LONG, price=52000.0, qty=0.01,
        source=OrderSource.GRID, reduce_only=True,
    )
    asyncio.run(trader._reconcile_orders([exit_order]))
    assert len(ex.created) == 1


# ── #4: window auto-sizes to 2 * loop_interval ──────────────────────

def test_dedup_window_grows_with_loop_interval():
    trader_fast, _ = _trader(["BTC/USDT:USDT"], loop_interval=5.0)
    trader_slow, _ = _trader(["BTC/USDT:USDT"], loop_interval=120.0)
    # Floor: 15s. Slow trader: 2 * 120s = 240s.
    assert trader_fast._recent_order_window_ms == 15_000
    assert trader_slow._recent_order_window_ms == 240_000


# ── #5: rejected-order path correctly identifies symbol ─────────────

def test_rejected_order_only_clears_its_own_symbol_record():
    """When BTC's order is rejected, ETH's same px/qty record must survive."""
    trader, ex = _trader(["BTC/USDT:USDT", "ETH/USDT:USDT"], dry_run=False)
    # Seed both symbols' recent_creates with full desired-identity tuples.
    now = 1_000_000.0
    btc_identity = trader._desired_identity(
        Order(symbol="BTC/USDT:USDT", side=Side.LONG, price=50000.0,
              qty=0.01, source=OrderSource.GRID)
    )
    eth_identity = trader._desired_identity(
        Order(symbol="ETH/USDT:USDT", side=Side.LONG, price=50000.0,
              qty=0.01, source=OrderSource.GRID)
    )
    trader._recent_creates.append((now, btc_identity))
    trader._recent_creates.append((now, eth_identity))
    # Now simulate BTC order rejected → only BTC entry should be removed.
    ex.next_status = "rejected"
    btc_order = Order(symbol="BTC/USDT:USDT", side=Side.LONG, price=50000.0,
                      qty=0.01, source=OrderSource.GRID)
    asyncio.run(trader._create_order(btc_order))
    remaining = list(trader._recent_creates)
    symbols_left = {ident[0] for (_, ident) in remaining}
    assert symbols_left == {"ETH/USDT:USDT"}


# ── #6: account balance must use wallet/total, not free margin ───────

def test_refresh_account_uses_wallet_total_not_free_margin():
    trader, ex = _trader(["BTC/USDT:USDT"])
    ex.balance_payload = {
        "USDT": {"free": 123.0, "total": 10_000.0},
        "info": {"totalWalletBalance": "9999.5"},
    }

    asyncio.run(trader._refresh_account())

    assert trader.account.balance == pytest.approx(9999.5)


# ── #7: create_order ack is not a fill event ─────────────────────────

def test_live_create_ack_does_not_update_trend_bucket_until_fill_event():
    trader, ex = _trader(["BTC/USDT:USDT"])
    order = Order(
        symbol="BTC/USDT:USDT", side=Side.LONG, price=50_000.0, qty=0.01,
        source=OrderSource.TREND, is_market=True,
    )

    asyncio.run(trader._create_order(order))

    ss = trader.account.symbols["BTC/USDT:USDT"]
    assert ss.trend_long.size == 0.0

    ex.trades_per_call = [[{
        "id": "fill-1", "timestamp": 1_000, "side": "buy",
        "price": 50_010.0, "amount": 0.01, "fee": {"cost": 0.1},
        "order": "1", "info": {"realizedPnl": "0"},
    }]]
    asyncio.run(trader._refresh_fills())

    assert ss.trend_long.size == pytest.approx(0.01)
    assert ss.trend_long.entry_price == pytest.approx(50_010.0)


def test_refresh_fills_does_not_mutate_exchange_authoritative_balance():
    trader, ex = _trader(["BTC/USDT:USDT"])
    trader.account.balance = 10_000.0
    ex.trades_per_call = [[{
        "id": "fill-1", "timestamp": 1_000, "side": "sell",
        "price": 51_000.0, "amount": 0.01, "fee": {"cost": 1.0},
        "order": "unknown", "info": {"realizedPnl": "50.0"},
    }]]

    asyncio.run(trader._refresh_fills())

    assert trader.account.balance == pytest.approx(10_000.0)
    assert trader.account.grid_realized_pnl == pytest.approx(49.0)


# ── #8: DefaultStrategy subclasses still get populate_* in live ──────

def test_live_runs_populate_for_default_strategy_subclass():
    class SignalStrategy(DefaultStrategy):
        def __init__(self):
            self.calls = 0

        def populate_entry_trend(self, dataframe, metadata):
            self.calls += 1
            dataframe["enter_long"] = 1
            return dataframe

    strategy = SignalStrategy()
    trader, _ = _trader(["BTC/USDT:USDT"])
    trader.strategy = strategy
    trader.strategy_runner.strategy = strategy
    trader.data_provider.append(
        "BTC/USDT:USDT",
        Candle(
            timestamp=1_000, open=50_000, high=50_100, low=49_900,
            close=50_000, volume=1,
        ),
    )

    trader._apply_strategy_populates("BTC/USDT:USDT")

    enter_long, _, _, _ = read_strategy_signals(
        trader.data_provider, "BTC/USDT:USDT",
    )
    assert strategy.calls == 1
    assert enter_long is True
