"""Tests for the round-7 ledger correctness work:

* Bucket reconstruction solves for grid entry_price from aggregate.
* Realized PnL ledger (per-source + 24h loss log) round-trips through
  the state file.
* Live timeframe / vol-target periods_per_year stay consistent.
* clientOrderId-based order identity replaces fuzzy matching when both
  sides carry it.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections import deque
from pathlib import Path

import pytest

from combo_bot.fusion_config import build_vol_target_sizer
from combo_bot.live import LiveConfig, LiveTrader
from combo_bot.types import (
    ExchangeParams,
    Order,
    OrderSource,
    Position,
    Side,
    SymbolState,
)


class _Stub:
    """Async stub exchange for the live tests in this file."""

    def __init__(self, positions=None):
        self.positions = positions or []
        self.created: list[dict] = []

    async def load_markets(self):
        return {}

    def market(self, _symbol):
        return {
            "precision": {"amount": 0.001, "price": 0.01},
            "limits": {"amount": {"min": 0.001}, "cost": {"min": 5.0}},
            "maker": 0.0002,
            "taker": 0.0005,
        }

    async def fetch_balance(self, _params=None):
        return {"USDT": {"free": 9_000.0, "total": 10_000.0}}

    async def fetch_positions(self, _syms):
        return self.positions

    async def fetch_funding_rate(self, _s):
        return {"fundingRate": 0.0}

    async def fetch_ohlcv(self, *_a, **_k):
        return []

    async def fetch_my_trades(self, *_a, **_k):
        return []

    async def fetch_open_orders(self, _s):
        return []

    async def create_order(self, symbol, ot, side, qty, price, params):
        self.created.append(
            {
                "symbol": symbol,
                "type": ot,
                "side": side,
                "qty": qty,
                "price": price,
                "params": params,
            }
        )
        return {"id": "ex-1", "status": "open"}

    async def cancel_order(self, *_a, **_k):
        return {}

    async def set_leverage(self, *_a, **_k):
        return {}

    async def set_margin_mode(self, *_a, **_k):
        return {}


def _trader_with_symbol() -> LiveTrader:
    ex = _Stub()
    cfg = LiveConfig(symbols=["BTC/USDT:USDT"], dry_run=False)
    trader = LiveTrader(cfg, ex)
    trader.exchange_params["BTC/USDT:USDT"] = ExchangeParams(
        qty_step=0.001,
        price_step=0.01,
        min_qty=0.001,
        min_cost=5.0,
    )
    trader.account.symbols["BTC/USDT:USDT"] = SymbolState(symbol="BTC/USDT:USDT")
    return trader


# ────────────────────────────────────────────────────────────────────
# 1. grid bucket entry_price derived from aggregate + trend bucket
# ────────────────────────────────────────────────────────────────────


def test_grid_entry_price_solved_from_aggregate():
    """Grid bucket entry must satisfy total_notional = grid + trend.

    Exchange shows size=1.0 at avg_entry=50000. Locally we know the trend
    bucket is 0.4 at entry 48000. So grid is 0.6, and the grid avg entry
    must satisfy 1.0 * 50000 = 0.4 * 48000 + 0.6 * grid_entry, i.e.
    grid_entry = (50000 - 0.4*48000) / 0.6 = 51333.33...
    """
    trader = _trader_with_symbol()
    ss = trader.account.symbols["BTC/USDT:USDT"]
    ss.trend_long = Position(size=0.4, entry_price=48_000.0)
    trader._rebuild_bucket(
        symbol="BTC/USDT:USDT",
        side=Side.LONG,
        exchange_size=1.0,
        exchange_entry=50_000.0,
        ss=ss,
    )
    assert ss.position_long.size == pytest.approx(0.6)
    expected = (1.0 * 50_000.0 - 0.4 * 48_000.0) / 0.6
    assert ss.position_long.entry_price == pytest.approx(expected, rel=1e-9)


def test_grid_entry_price_falls_back_when_trend_notional_exceeds_aggregate():
    """If our trend tracking is over-stale, derived grid_entry can go
    negative — fall back to the exchange avg rather than emit nonsense.

    Trigger: trend bucket entry far above the aggregate, so trend
    notional eats more than the aggregate notional.
    """
    trader = _trader_with_symbol()
    ss = trader.account.symbols["BTC/USDT:USDT"]
    # Trend bucket says 0.5 at 120k → 60k notional. Exchange aggregate
    # is 1.0 at 50k → 50k notional. (50k - 60k) / 0.5 = -20k → negative.
    ss.trend_long = Position(size=0.5, entry_price=120_000.0)
    trader._rebuild_bucket(
        symbol="BTC/USDT:USDT",
        side=Side.LONG,
        exchange_size=1.0,
        exchange_entry=50_000.0,
        ss=ss,
    )
    # grid_entry would compute negative; we fall back to exchange avg.
    assert ss.position_long.entry_price > 0
    assert ss.position_long.entry_price == pytest.approx(50_000.0)


def test_trend_tracking_clamped_when_it_exceeds_exchange_total():
    """tracked_trend > exchange_total is structurally impossible — clamp."""
    trader = _trader_with_symbol()
    ss = trader.account.symbols["BTC/USDT:USDT"]
    ss.trend_long = Position(size=2.0, entry_price=50_000.0)  # absurd
    trader._rebuild_bucket(
        symbol="BTC/USDT:USDT",
        side=Side.LONG,
        exchange_size=1.0,
        exchange_entry=50_000.0,
        ss=ss,
    )
    assert ss.trend_long.size == pytest.approx(1.0)
    assert ss.position_long.size == pytest.approx(0.0)


# ────────────────────────────────────────────────────────────────────
# 2. realized PnL ledger persistence
# ────────────────────────────────────────────────────────────────────


def test_realized_pnl_ledger_round_trips_through_state_file():
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "state.json"
        trader_a = _trader_with_symbol()
        trader_a.config.state_file = str(state_path)
        # Simulate a restart-resilient state.
        trader_a.account.grid_realized_pnl = 123.45
        trader_a.account.trend_realized_pnl = -67.89
        trader_a.account.grid_equity_peak = 200.0
        trader_a.account.trend_equity_peak = 50.0
        trader_a.account.grid_loss_log = deque(
            [(1_700_000_000_000, -10.0), (1_700_000_100_000, -5.0)]
        )
        trader_a.account.trend_loss_log = deque([(1_700_000_200_000, -3.0)])
        asyncio.run(trader_a._save_state())

        # Verify the JSON has them.
        on_disk = json.loads(state_path.read_text())
        assert on_disk["grid_realized_pnl"] == pytest.approx(123.45)
        assert on_disk["trend_realized_pnl"] == pytest.approx(-67.89)

        # Cold start, load — ledger comes back.
        trader_b = _trader_with_symbol()
        trader_b.config.state_file = str(state_path)
        asyncio.run(trader_b._load_state())
        assert trader_b.account.grid_realized_pnl == pytest.approx(123.45)
        assert trader_b.account.trend_realized_pnl == pytest.approx(-67.89)
        assert trader_b.account.grid_equity_peak == pytest.approx(200.0)
        assert len(trader_b.account.grid_loss_log) == 2
        assert trader_b.account.grid_loss_log[0] == (1_700_000_000_000, -10.0)
        assert len(trader_b.account.trend_loss_log) == 1


# ────────────────────────────────────────────────────────────────────
# 3. timeframe consistency between live config and vol-target sizer
# ────────────────────────────────────────────────────────────────────


def test_live_config_carries_timeframe_and_bar_interval():
    cfg = LiveConfig(
        candle_timeframe="1h",
        bar_interval_minutes=60.0,
    )
    assert cfg.candle_timeframe == "1h"
    assert cfg.bar_interval_minutes == 60.0


def test_vol_target_sizer_derives_periods_per_year_from_bar_interval():
    """When bar_interval_minutes is set in the top-level cfg, the sizer's
    periods_per_year must scale so Sharpe-style annualization stays
    correct (60-min bars → 8760 periods/year, not 525600)."""
    cfg = {
        "vol_target_sizer": {"enabled": True, "target_annual_vol": 0.3},
        "bar_interval_minutes": 60.0,
    }
    sizer = build_vol_target_sizer(cfg)
    assert sizer is not None
    assert sizer.config.periods_per_year == pytest.approx(8760, rel=0.01)


def test_vol_target_sizer_honors_explicit_periods_per_year():
    """User-set periods_per_year must not be overwritten by the
    bar_interval derivation."""
    cfg = {
        "vol_target_sizer": {
            "enabled": True,
            "periods_per_year": 12345,
        },
        "bar_interval_minutes": 60.0,
    }
    sizer = build_vol_target_sizer(cfg)
    assert sizer.config.periods_per_year == 12345


# ────────────────────────────────────────────────────────────────────
# 4. clientOrderId-based order identity
# ────────────────────────────────────────────────────────────────────


def test_create_order_stamps_client_order_id_into_ccxt_params():
    trader = _trader_with_symbol()
    o = Order(
        symbol="BTC/USDT:USDT",
        side=Side.LONG,
        price=50_000.0,
        qty=0.01,
        source=OrderSource.GRID,
    )
    asyncio.run(trader._create_order(o))
    sent = trader.exchange.created[0]
    cid = sent["params"].get("clientOrderId")
    assert cid is not None
    assert cid.startswith("cb-")
    assert len(cid) <= 36  # Binance USDM limit


def test_orders_match_uses_client_order_id_when_present():
    trader = _trader_with_symbol()
    desired = Order(
        symbol="BTC/USDT:USDT",
        side=Side.LONG,
        price=50_000.0,
        qty=0.01,
        source=OrderSource.GRID,
        client_order_id="cb-abc123",
    )
    # Same cOID — must match even with totally different fuzzy fields.
    existing = {
        "side": "buy",
        "price": 99_999.0,
        "amount": 99.0,  # would fail fuzzy
        "clientOrderId": "cb-abc123",
    }
    assert trader._orders_match(desired, existing) is True

    # Same fuzzy match but different cOID — must NOT match.
    other = {
        "side": "buy",
        "price": 50_000.0,
        "amount": 0.01,
        "clientOrderId": "cb-different",
    }
    assert trader._orders_match(desired, other) is False


def test_orders_match_falls_back_to_fuzzy_when_no_cid_but_reduce_only_present():
    """When the existing record lacks a cOID but DOES echo reduceOnly,
    fuzzy fallback still works."""
    trader = _trader_with_symbol()
    desired = Order(
        symbol="BTC/USDT:USDT",
        side=Side.LONG,
        price=50_000.0,
        qty=0.01,
        source=OrderSource.GRID,
        client_order_id="cb-abc123",
    )
    existing_no_cid = {
        "side": "buy",
        "price": 50_000.0,
        "amount": 0.01,
        "reduceOnly": False,
    }
    assert trader._orders_match(desired, existing_no_cid) is True


def test_orders_match_conservatively_refuses_when_no_cid_and_no_reduce_only():
    """Round-10 P1.a: if BOTH cOID AND reduceOnly are missing on the
    existing record, refuse to match. The safer outcome is sending the
    new order (recent_creates dedup will catch any real duplicate)
    than silently absorbing our desired into an unknown exchange order
    of mystery intent (could be a manual order, a leftover, etc.)."""
    trader = _trader_with_symbol()
    desired = Order(
        symbol="BTC/USDT:USDT",
        side=Side.LONG,
        price=50_000.0,
        qty=0.01,
        source=OrderSource.GRID,
        client_order_id="cb-abc123",
    )
    mystery_existing = {
        "side": "buy",
        "price": 50_000.0,
        "amount": 0.01,
        # No clientOrderId, no reduceOnly field at all.
    }
    assert trader._orders_match(desired, mystery_existing) is False


# ────────────────────────────────────────────────────────────────────
# 5. cOID persists across ticks for the same desired identity
#    (round-8 #1 — was missing in round-7).
# ────────────────────────────────────────────────────────────────────


def test_cid_persists_across_reconcile_ticks_for_same_desired():
    """A logical desired order recurring across ticks must reuse the
    same cOID so reconcile can match it against the exchange's
    clientOrderId echo instead of falling back to fuzzy heuristics."""
    trader = _trader_with_symbol()
    o = Order(
        symbol="BTC/USDT:USDT",
        side=Side.LONG,
        price=50_000.0,
        qty=0.01,
        source=OrderSource.GRID,
    )
    asyncio.run(trader._reconcile_orders([o]))
    first_cid = trader.exchange.created[0]["params"]["clientOrderId"]

    # Second tick: same logical desired. Without persistent cache, the
    # _assign_cid would emit a fresh UUID and the exchange would see
    # two distinct cOIDs for what is one ongoing entry.
    trader.exchange.created.clear()
    # Need to clear recent_creates dedup or it'll skip — that's a
    # SEPARATE guard. Bypass by changing qty so dedup misses.
    o2 = Order(
        symbol="BTC/USDT:USDT",
        side=Side.LONG,
        price=50_000.0,
        qty=0.01,
        source=OrderSource.GRID,
    )
    # Manually advance _recent_creates past its window.
    trader._recent_creates.clear()
    asyncio.run(trader._reconcile_orders([o2]))
    second_cid = trader.exchange.created[0]["params"]["clientOrderId"]
    assert (
        first_cid == second_cid
    ), f"same desired identity must reuse cOID; got {first_cid} vs {second_cid}"


def test_cid_differs_across_distinct_desired_identities():
    """Two orders with different (symbol, side, source, price, qty)
    must get different cOIDs."""
    # Use a fresh trader with both symbols registered in LiveConfig,
    # not just account.symbols — reconcile only iterates config.symbols.
    ex = _Stub()
    cfg = LiveConfig(
        symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
        dry_run=False,
    )
    trader = LiveTrader(cfg, ex)
    for s in cfg.symbols:
        trader.exchange_params[s] = ExchangeParams(
            qty_step=0.001,
            price_step=0.01,
            min_qty=0.001,
            min_cost=5.0,
        )
        trader.account.symbols[s] = SymbolState(symbol=s)
    a = Order(
        symbol="BTC/USDT:USDT",
        side=Side.LONG,
        price=50_000.0,
        qty=0.01,
        source=OrderSource.GRID,
    )
    b = Order(
        symbol="ETH/USDT:USDT",
        side=Side.LONG,
        price=50_000.0,
        qty=0.01,
        source=OrderSource.GRID,
    )
    asyncio.run(trader._reconcile_orders([a, b]))
    cids = [c["params"]["clientOrderId"] for c in trader.exchange.created]
    assert len(cids) == 2
    assert cids[0] != cids[1]


# ────────────────────────────────────────────────────────────────────
# 6. ghost position cleanup: side missing from fetch_positions clears
#    the local grid bucket (round-8 #2).
# ────────────────────────────────────────────────────────────────────


def test_missing_side_in_fetch_positions_clears_grid_bucket():
    """If the exchange stops returning a side, that bucket got closed.
    Locally we must clear it so the next overlay/grid decision doesn't
    keep referencing a ghost position."""
    trader = _trader_with_symbol()
    ss = trader.account.symbols["BTC/USDT:USDT"]
    ss.position_long = Position(size=0.5, entry_price=50_000.0)

    # Exchange returns positions: short only. Long is conspicuously absent.
    trader.exchange.positions = [
        {
            "symbol": "BTC/USDT:USDT",
            "side": "short",
            "contracts": 0.1,
            "entryPrice": 51_000.0,
            "markPrice": 50_500.0,
        }
    ]
    asyncio.run(trader._refresh_account())

    assert (
        ss.position_long.size == 0.0
    ), "long bucket must be cleared when the exchange stops echoing it"
    assert ss.position_short.size == pytest.approx(0.1)


def test_zero_size_row_does_not_keep_bucket_alive():
    """A zero-size echo (some exchanges include closed sides as size=0)
    must also count as 'not present' so the bucket clears."""
    trader = _trader_with_symbol()
    ss = trader.account.symbols["BTC/USDT:USDT"]
    ss.position_long = Position(size=0.5, entry_price=50_000.0)

    trader.exchange.positions = [
        {
            "symbol": "BTC/USDT:USDT",
            "side": "long",
            "contracts": 0.0,
            "entryPrice": 0.0,
            "markPrice": 50_500.0,
        }
    ]
    asyncio.run(trader._refresh_account())

    assert ss.position_long.size == 0.0


# ────────────────────────────────────────────────────────────────────
# 7. EMA / volatility spans come from grid config (round-8 #3).
# ────────────────────────────────────────────────────────────────────


def test_ema_and_volatility_spans_use_grid_config():
    """Live EMA/vol init must honor GridConfig.ema_span_0/1 and
    entry_volatility_ema_span_hours; previously they were hardcoded
    [385, 620] and 1000.0, so any user tuning in grid config was
    silently ignored in live."""
    from combo_bot.grid_engine import GridConfig

    grid = GridConfig(
        ema_span_0=100.0,
        ema_span_1=200.0,
        entry_volatility_ema_span_hours=42.0,
    )
    ex = _Stub()
    cfg = LiveConfig(
        symbols=["BTC/USDT:USDT"],
        dry_run=False,
        grid=grid,
        candle_timeframe="1m",
        bar_interval_minutes=1.0,
    )
    trader = LiveTrader(cfg, ex)
    trader.exchange_params["BTC/USDT:USDT"] = ExchangeParams()
    trader.account.symbols["BTC/USDT:USDT"] = SymbolState(symbol="BTC/USDT:USDT")

    # Inject one candle through _refresh_candles' fetch hook.
    class _OhlcvEx(_Stub):
        async def fetch_ohlcv(self, *_a, **_k):
            return [[60_000, 100.0, 101.0, 99.0, 100.0, 1.0]]

    trader.exchange = _OhlcvEx()
    asyncio.run(trader._refresh_candles())

    ss = trader.account.symbols["BTC/USDT:USDT"]
    assert ss.ema.spans == [100.0, 200.0]
    assert ss.volatility.ema_span_hours == 42.0


# ────────────────────────────────────────────────────────────────────
# 8. live record_equity for vol_target_sizer (round-8 #4).
# ────────────────────────────────────────────────────────────────────


def test_live_tick_calls_record_equity_on_vol_target_sizer():
    """The live tick loop must feed equity to the sizer or the sizer
    stays at cold-start scale=1.0 forever — Stage 11 silently disabled."""
    from combo_bot.vol_target import VolTargetSizer, VolTargetSizerConfig

    sizer = VolTargetSizer(VolTargetSizerConfig(min_samples=2))
    ex = _Stub()
    cfg = LiveConfig(symbols=["BTC/USDT:USDT"], dry_run=True)
    trader = LiveTrader(cfg, ex, vol_target_sizer=sizer)
    trader.exchange_params["BTC/USDT:USDT"] = ExchangeParams(
        qty_step=0.001,
        price_step=0.01,
        min_qty=0.001,
        min_cost=5.0,
    )
    trader.account.symbols["BTC/USDT:USDT"] = SymbolState(symbol="BTC/USDT:USDT")
    # Drive a tick. The sizer must register equity each call.
    assert sizer.sample_size() == 0
    asyncio.run(trader._tick())
    asyncio.run(trader._tick())
    assert sizer.sample_size() >= 1, (
        "vol_target_sizer.record_equity must be called per tick — "
        "otherwise live is stuck at cold-start scale=1.0"
    )
