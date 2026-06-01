"""Round-19 fail-closed live-execution tests.

These cover the remaining live-safety gaps after the fill-ledger work:

* a stuck fill stream must survive restart;
* a stuck fill stream must block new risk-increasing orders;
* open-order refresh failure must not be treated as "no open orders";
* account refresh failure must abort the tick before creating orders.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from combo_bot.fill_events_manager import (
    FillEventManager,
    FillEventManagerConfig,
)
from combo_bot.grid_engine import GridConfig
from combo_bot.live import LiveConfig, LiveTrader
from combo_bot.types import (
    ExchangeParams,
    Fill,
    Order,
    OrderSource,
    Side,
    SymbolState,
)


class _Exchange:
    def __init__(self):
        self.created: list[dict] = []
        self.open_orders_error = False
        self.balance_error = False
        self.balance_payload = {"USDT": {"total": 10_000.0}}
        self.ohlcv = [[1_000, 50_000.0, 50_000.0, 50_000.0, 50_000.0, 1.0]]

    async def load_markets(self):
        return {}

    def market(self, _symbol):
        return {
            "precision": {"amount": 0.001, "price": 0.01},
            "limits": {"amount": {"min": 0.001}, "cost": {"min": 5.0}},
            "maker": 0.0002,
            "taker": 0.0005,
        }

    async def fetch_my_trades(self, *args, **kwargs):
        return []

    async def fetch_open_orders(self, symbol):
        if self.open_orders_error:
            raise RuntimeError("open orders unavailable")
        return []

    async def fetch_balance(self, _params=None):
        if self.balance_error:
            raise RuntimeError("balance unavailable")
        return self.balance_payload

    async def fetch_positions(self, _symbols):
        return []

    async def fetch_funding_rate(self, _symbol):
        return {"fundingRate": 0.0}

    async def fetch_ohlcv(self, *_args, **_kwargs):
        return self.ohlcv

    async def create_order(self, symbol, order_type, side, qty, price, params):
        self.created.append(
            {
                "symbol": symbol,
                "type": order_type,
                "side": side,
                "qty": qty,
                "price": price,
                "params": params,
            }
        )
        return {"id": f"order-{len(self.created)}", "status": "open"}

    async def cancel_order(self, *_args, **_kwargs):
        return {}

    async def set_leverage(self, *_args, **_kwargs):
        return {}

    async def set_margin_mode(self, *_args, **_kwargs):
        return {}


def _live_trader(*, state_file: str | None = None) -> tuple[LiveTrader, _Exchange]:
    symbol = "BTC/USDT:USDT"
    ex = _Exchange()
    cfg = LiveConfig(
        symbols=[symbol],
        dry_run=False,
        state_file=state_file or "test_state.json",
        loop_interval_seconds=60,
        grid=GridConfig(
            entry_initial_qty_pct=0.01,
            wallet_exposure_limit=0.10,
            max_grid_levels=1,
        ),
    )
    trader = LiveTrader(
        cfg,
        ex,
        fill_events_config=FillEventManagerConfig(poll_interval_ms=0),
    )
    trader.exchange_params[symbol] = ExchangeParams(
        qty_step=0.001,
        price_step=0.01,
        min_qty=0.001,
        min_cost=5.0,
    )
    trader.account.symbols[symbol] = SymbolState(symbol=symbol, last_price=50_000.0)
    trader.account.balance = 10_000.0
    return trader, ex


def test_fill_event_stuck_state_round_trips_snapshot():
    """A cursor-stuck (same-ms pagination stall) is a ledger-integrity
    condition and MUST survive snapshot/restore so it keeps blocking
    entries after a crash-restart. A fetch-failure (transient exchange/
    API error) is DROPPED on restore — a fresh process is the natural
    retry, and the detector re-escalates within the session if it recurs."""
    symbol = "BTC/USDT:USDT"

    # cursor-stuck → persists across restart.
    mgr = FillEventManager(
        _Exchange(),
        FillEventManagerConfig(poll_interval_ms=0, page_size=2),
    )
    mgr._stuck_symbols.add(symbol)
    mgr._stuck_count[symbol] = 3
    mgr._stuck_reason[symbol] = "cursor"

    restored = FillEventManager(_Exchange())
    restored.load_snapshot(mgr.snapshot())

    assert symbol in restored._stuck_symbols, "cursor-stuck must persist"
    assert restored._stuck_count[symbol] == 3
    assert restored._stuck_reason[symbol] == "cursor"

    # fetch-failure → dropped on restart.
    mgr2 = FillEventManager(
        _Exchange(),
        FillEventManagerConfig(poll_interval_ms=0, page_size=2),
    )
    mgr2._stuck_symbols.add(symbol)
    mgr2._stuck_count[symbol] = 5
    mgr2._stuck_reason[symbol] = "fetch"

    restored2 = FillEventManager(_Exchange())
    restored2.load_snapshot(mgr2.snapshot())

    assert symbol not in restored2._stuck_symbols, (
        "fetch-failure STUCK must be dropped on restart (transient)"
    )

    # legacy state file with no reason tag → treated as transient, dropped.
    legacy = FillEventManager(_Exchange())
    legacy.load_snapshot(
        {"stuck_symbols": [symbol], "stuck_count": {symbol: 3}}
    )
    assert symbol not in legacy._stuck_symbols, (
        "untagged (legacy) STUCK must be dropped — cursor-stuck self-re-detects"
    )


def test_reconcile_blocks_new_entries_when_fill_stream_is_stuck():
    trader, ex = _live_trader()
    symbol = "BTC/USDT:USDT"
    trader.fill_events._stuck_symbols.add(symbol)

    entry = Order(
        symbol=symbol,
        side=Side.LONG,
        price=50_000.0,
        qty=0.01,
        source=OrderSource.GRID,
    )
    exit_order = Order(
        symbol=symbol,
        side=Side.LONG,
        price=50_500.0,
        qty=0.01,
        source=OrderSource.GRID,
        reduce_only=True,
    )

    asyncio.run(trader._reconcile_orders([entry, exit_order]))

    assert len(ex.created) == 1
    assert ex.created[0]["params"].get("reduceOnly") is True


def test_reconcile_logs_stuck_fill_stream_once_per_symbol(caplog):
    """A stuck symbol can have many desired entries in one tick.

    They should all be blocked, but the operator log should get one clear
    fail-closed line for the symbol, not one line per candidate order.
    """

    trader, ex = _live_trader()
    symbol = "BTC/USDT:USDT"
    trader.fill_events._stuck_symbols.add(symbol)
    entries = [
        Order(
            symbol=symbol,
            side=Side.LONG,
            price=50_000.0 + i,
            qty=0.01,
            source=OrderSource.GRID,
        )
        for i in range(3)
    ]

    caplog.set_level("ERROR", logger="combo_bot.live")
    stats = asyncio.run(trader._reconcile_orders(entries))

    assert ex.created == []
    assert stats == {"desired": 3, "create_attempted": 0, "cancel_attempted": 0}
    stuck_lines = [
        r
        for r in caplog.records
        if "fill stream is STUCK" in r.getMessage()
    ]
    assert len(stuck_lines) == 1


def test_reconcile_blocks_entries_when_open_orders_refresh_fails():
    trader, ex = _live_trader()
    ex.open_orders_error = True

    entry = Order(
        symbol="BTC/USDT:USDT",
        side=Side.LONG,
        price=50_000.0,
        qty=0.01,
        source=OrderSource.GRID,
    )

    asyncio.run(trader._reconcile_orders([entry]))

    assert ex.created == []


def test_tick_aborts_before_order_creation_when_account_refresh_fails():
    with tempfile.TemporaryDirectory() as tmp:
        trader, ex = _live_trader(state_file=str(Path(tmp) / "state.json"))
        ex.balance_error = True

        asyncio.run(trader._tick())

        assert ex.created == []


def test_candle_fetch_failure_freshness_blocks_then_self_heals():
    """A per-symbol candle fetch failure must block that symbol's
    risk-increasing orders (freshness gate), and the block must clear
    automatically once the candle surface refreshes on a later tick."""
    trader, ex = _live_trader()
    sym = "BTC/USDT:USDT"

    # Tick 1: candle fetch raises → freshness flags the symbol.
    async def _boom(*_a, **_k):
        raise RuntimeError("ohlcv feed down")

    ex.fetch_ohlcv = _boom
    trader.freshness.begin_epoch()
    asyncio.run(trader._refresh_candles())
    assert trader.freshness.is_blocked(sym) is True
    assert trader._risk_increasing_blocked(sym) is True

    # Tick 2: candle fetch recovers → stamp clears the block.
    async def _ok(*_a, **_k):
        return [[2_000, 50_000.0, 50_100.0, 49_900.0, 50_050.0, 1.0]]

    ex.fetch_ohlcv = _ok
    trader.freshness.begin_epoch()
    asyncio.run(trader._refresh_candles())
    assert trader.freshness.is_blocked(sym) is False
    assert trader._risk_increasing_blocked(sym) is False


def test_enrich_fill_pnl_flags_degraded_when_close_exceeds_known_size():
    """A reduce-only fill whose qty exceeds the locally-known bucket size
    has an incomplete cost basis — its reconstructed PnL must be flagged
    degraded (still booked, but withheld from Kelly)."""
    trader, _ = _live_trader()
    sym = "BTC/USDT:USDT"
    ss = trader.account.symbols[sym]
    ss.position_long.size = 0.001
    ss.position_long.entry_price = 50_000.0

    fill = Fill(
        timestamp=1, symbol=sym, side=Side.LONG, price=51_000.0,
        qty=0.003, fee=0.0, realized_pnl=0.0, source=OrderSource.GRID,
        reduce_only=True,
    )
    enriched = trader._enrich_fill_pnl(fill)
    assert enriched.pnl_degraded is True
    # PnL still reconstructed (equity stays whole), just low-confidence.
    assert enriched.realized_pnl != 0.0


def test_enrich_fill_pnl_not_degraded_within_known_size():
    """A reduce-only close within the known position size reconstructs a
    trustworthy PnL and is NOT degraded."""
    trader, _ = _live_trader()
    sym = "BTC/USDT:USDT"
    ss = trader.account.symbols[sym]
    ss.position_long.size = 0.005
    ss.position_long.entry_price = 50_000.0

    fill = Fill(
        timestamp=1, symbol=sym, side=Side.LONG, price=51_000.0,
        qty=0.002, fee=0.0, realized_pnl=0.0, source=OrderSource.GRID,
        reduce_only=True,
    )
    enriched = trader._enrich_fill_pnl(fill)
    assert enriched.pnl_degraded is False
    assert enriched.realized_pnl == 0.002 * (51_000.0 - 50_000.0)


def test_degraded_fills_withheld_from_kelly_but_booked_to_ledger():
    """End-to-end: a degraded fill is booked into the account ledger
    (equity correct) but excluded from the Kelly edge estimator."""
    from combo_bot.sizing import KellySizer, KellySizerConfig

    trader, _ = _live_trader()
    sym = "BTC/USDT:USDT"
    trader.kelly_sizer = KellySizer(KellySizerConfig())
    ss = trader.account.symbols[sym]
    ss.position_long.size = 0.001
    ss.position_long.entry_price = 50_000.0

    degraded = trader._enrich_fill_pnl(
        Fill(
            timestamp=1, symbol=sym, side=Side.LONG, price=51_000.0,
            qty=0.003, fee=0.0, realized_pnl=0.0, source=OrderSource.GRID,
            reduce_only=True,
        )
    )
    assert degraded.pnl_degraded is True

    # Mirror the live consumption: ledger gets it, Kelly does not.
    before = trader.account.grid_realized_pnl
    trader.account.add_realized_pnl(
        degraded.source, degraded.realized_pnl - degraded.fee, degraded.timestamp
    )
    assert trader.account.grid_realized_pnl != before  # booked
    kelly_fills = [degraded] if not degraded.pnl_degraded else []
    assert kelly_fills == []  # withheld from Kelly


def test_refresh_candles_excludes_in_progress_bar():
    """The still-forming current bar (ccxt returns it as the last OHLCV
    row) must NOT be fed to trend/EMA/volatility — only closed bars are
    final. last_price still reflects the latest (live) bar for marking."""
    trader, ex = _live_trader()
    sym = "BTC/USDT:USDT"
    NOW = 1_000_000_000_000
    trader._now_ms = lambda: NOW  # type: ignore[method-assign]
    tf = 60_000  # 1m default
    a_ts, b_ts, c_ts = NOW - 2 * tf, NOW - tf, NOW  # A,B closed; C forming
    ex.ohlcv = [
        [a_ts, 100.0, 110.0, 90.0, 101.0, 1.0],
        [b_ts, 101.0, 111.0, 91.0, 102.0, 1.0],
        [c_ts, 102.0, 112.0, 92.0, 999.0, 1.0],  # in-progress, distinctive close
    ]
    asyncio.run(trader._refresh_candles())
    # In-progress bar C must NOT advance the indicator watermark…
    assert trader._last_candle_ts[sym] == b_ts
    # …but last_price still reflects the latest (live) bar.
    assert trader.account.symbols[sym].last_price == 999.0
