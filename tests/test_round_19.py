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
from combo_bot.types import ExchangeParams, Order, OrderSource, Side, SymbolState


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
    mgr = FillEventManager(
        _Exchange(),
        FillEventManagerConfig(poll_interval_ms=0, page_size=2),
    )
    symbol = "BTC/USDT:USDT"
    mgr._stuck_symbols.add(symbol)
    mgr._stuck_count[symbol] = 3

    restored = FillEventManager(_Exchange())
    restored.load_snapshot(mgr.snapshot())

    assert symbol in restored._stuck_symbols
    assert restored._stuck_count[symbol] == 3


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
