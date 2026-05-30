"""Round-16 tests:

* fetch_order-written bucket isn't double-applied when the real
  trade later arrives via fetch_my_trades.
* stale UNKNOWN holds when exchange isn't confirmed flat.
* fill polling holds watermark on same-ms stuck pagination.
* custom_exit / custom_stoploss output passes through confirm_trade_exit.
* market exits skip custom_exit_price.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest


class _Stub:
    def __init__(self):
        self.created = []
        self.trades_by_call = []
        self.positions = []

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
        return []

    async def create_order(self, *a, **k):
        return {"id": f"ex-{len(self.created)+1}", "status": "open"}

    async def cancel_order(self, *a, **k):
        return {}

    async def set_leverage(self, *a, **k):
        return {}

    async def set_margin_mode(self, *a, **k):
        return {}


def _trader(tmpdir: Path):
    from combo_bot.live import LiveConfig, LiveTrader
    from combo_bot.types import ExchangeParams, SymbolState

    ex = _Stub()
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


# ────────────────────────────────────────────────────────────────────
# P0 #1: fetch_order bucket-write must dedup the later real trade
# ────────────────────────────────────────────────────────────────────


def test_fetch_order_resolved_bucket_not_double_applied_by_late_trade():
    """If fetch_order's filled/avg already wrote the trend bucket,
    the same order's trade arriving via fetch_my_trades MUST NOT
    re-apply it (would double the trend position)."""
    from combo_bot.types import Side

    with tempfile.TemporaryDirectory() as tmp:
        trader = _trader(Path(tmp))
        # Set up an UNKNOWN slot with a journal cID.
        trader.intent_journal.submit(
            cid="cb-target",
            symbol="BTC/USDT:USDT",
            side="long",
            source="trend",
            reduce_only=False,
            is_market=True,
            now_ms=1,
        )
        trader.intent_journal.open(
            cid="cb-target",
            exchange_id="ex-7",
            now_ms=2,
        )
        trader._unknown_overlay[("BTC/USDT:USDT", Side.LONG)] = 1.0

        # fetch_order returns terminal + qty/avg → bucket written.
        async def fo(*a, **k):
            return {
                "status": "closed",
                "id": "ex-7",
                "filled": 0.01,
                "average": 50_000.0,
            }

        trader.exchange.fetch_order = fo  # type: ignore[method-assign]
        asyncio.run(trader._resolve_unknowns())
        ss = trader.account.symbols["BTC/USDT:USDT"]
        assert ss.trend_long.size == pytest.approx(0.01)

        # Now the real trade arrives — must NOT double the bucket.
        trader.exchange.trades_by_call = [
            [
                {
                    "id": "t-1",
                    "timestamp": 9_000,
                    "side": "buy",
                    "price": 50_000.0,
                    "amount": 0.01,
                    "fee": {"cost": 0.05},
                    "order": "ex-7",
                    "clientOrderId": "cb-target",
                    "info": {"realizedPnl": "0"},
                },
            ]
        ]
        asyncio.run(trader._refresh_fills())
        # Bucket size still 0.01, not 0.02.
        assert ss.trend_long.size == pytest.approx(0.01), (
            "late real trade for an order already bucket-written via "
            "fetch_order must NOT double the trend position"
        )


# ────────────────────────────────────────────────────────────────────
# P0 #2: stale UNKNOWN holds when exchange shows non-flat position
# ────────────────────────────────────────────────────────────────────


def test_stale_unknown_holds_when_exchange_not_confirmed_flat():
    """Empty buckets + empty journal + exchange shows residual
    position on this (symbol, side) → HOLD UNKNOWN. Auto-clearing
    here would let a duplicate market entry leak through."""
    from combo_bot.types import Side

    with tempfile.TemporaryDirectory() as tmp:
        trader = _trader(Path(tmp))
        trader._unknown_overlay[("BTC/USDT:USDT", Side.LONG)] = 1.0
        # Exchange reports a residual long contract.
        trader.exchange.positions = [
            {
                "symbol": "BTC/USDT:USDT",
                "side": "long",
                "contracts": 0.05,
                "entryPrice": 50_000.0,
                "markPrice": 50_500.0,
            }
        ]
        asyncio.run(trader._resolve_unknowns())
        # MUST hold the block.
        assert ("BTC/USDT:USDT", Side.LONG) in trader._unknown_overlay


def test_stale_unknown_clears_only_when_exchange_confirmed_flat():
    from combo_bot.types import Side

    with tempfile.TemporaryDirectory() as tmp:
        trader = _trader(Path(tmp))
        trader._unknown_overlay[("BTC/USDT:USDT", Side.LONG)] = 1.0
        # Exchange shows no positions at all.
        trader.exchange.positions = []
        asyncio.run(trader._resolve_unknowns())
        assert ("BTC/USDT:USDT", Side.LONG) not in trader._unknown_overlay


# ────────────────────────────────────────────────────────────────────
# P1 #1: same-ms stuck pagination must hold watermark
# ────────────────────────────────────────────────────────────────────


def test_fill_polling_holds_watermark_on_same_ms_stuck_burst():
    from combo_bot.fill_events_manager import (
        FillEventManager,
        FillEventManagerConfig,
    )

    burst_ts = 1_000

    class _BurstEx:
        def __init__(self):
            self.calls = 0
            self.served_ids: set[str] = set()

        async def fetch_my_trades(self, symbol, since=None, limit=None, params=None):
            self.calls += 1
            from_id = (params or {}).get("fromId") if params else None
            # Binance semantics: fromId is the last trade YOU saw, so
            # the next batch starts AFTER it.
            if from_id is None:
                ids = ["burst-1", "burst-2"]  # full page
            elif from_id == "burst-2":
                ids = ["burst-3"]  # short page → drain ends
            else:
                ids = []
            return [
                {
                    "id": tid,
                    "timestamp": burst_ts,
                    "side": "buy",
                    "price": 50.0,
                    "amount": 0.001,
                    "fee": {"cost": 0.0},
                    "order": "ord-1",
                    "info": {},
                }
                for tid in ids
            ]

    ex = _BurstEx()
    mgr = FillEventManager(
        ex,
        FillEventManagerConfig(
            poll_interval_ms=0,
            page_size=2,
        ),
    )
    captured = []
    asyncio.run(mgr.poll("BTC/USDT:USDT", now_ms=2_000, sink=captured.extend))
    # All three trades drained via the fromId pagination.
    captured_ids = {f.trade_id for f in captured}
    assert captured_ids == {"burst-1", "burst-2", "burst-3"}


# ────────────────────────────────────────────────────────────────────
# P1 #2: custom_exit/custom_stoploss passes through confirm_trade_exit
# ────────────────────────────────────────────────────────────────────


def test_custom_exit_can_be_vetoed_by_confirm_trade_exit():
    from combo_bot.backtest import BacktestConfig, Backtester
    from combo_bot.strategy import IStrategy
    from combo_bot.types import Candle

    class _StratWithExitAndVeto(IStrategy):
        def populate_indicators(self, df, m):
            return df

        def populate_entry_trend(self, df, m):
            return df

        def populate_exit_trend(self, df, m):
            return df

        def custom_exit(self, ctx, profit_pct):
            return "wants_to_exit" if ctx.position.is_open else None

        def confirm_trade_exit(self, ctx, qty, price, reason):
            # Always veto.
            return False

    bt = Backtester(
        BacktestConfig(starting_balance=10_000), strategy=_StratWithExitAndVeto()
    )
    candles = [Candle(i * 60_000, 100, 101, 99, 100, 1) for i in range(100)]
    result = bt.run({"BTC": candles})
    # Without the round-16 fix, custom_exit would fire and bypass
    # the veto. With the fix, the strategy's veto holds.
    # We don't assert on PnL, just that no RISK-source close fills
    # snuck through.
    risk_closes = [
        f for f in result.fills if f.source.value == "risk" and f.reduce_only
    ]
    assert risk_closes == [], (
        f"custom_exit market closes must be vetoable via "
        f"confirm_trade_exit; got {len(risk_closes)} that snuck through"
    )


# ────────────────────────────────────────────────────────────────────
# P2: market exit skips custom_exit_price
# ────────────────────────────────────────────────────────────────────


def test_market_exit_skips_custom_exit_price():
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

    custom_called = {"count": 0}

    class _S(IStrategy):
        def populate_indicators(self, df, m):
            return df

        def populate_entry_trend(self, df, m):
            return df

        def populate_exit_trend(self, df, m):
            return df

        def custom_exit_price(self, ctx, proposed, reason):
            custom_called["count"] += 1
            return proposed + 999.0  # would dramatically shift if called

        def confirm_trade_exit(self, ctx, qty, price, reason):
            return True

    runner = StrategyRunner(_S())
    pos = Position(size=0.01, entry_price=50_000.0)
    ss = SymbolState(symbol="BTC", position_long=pos)
    acc = AccountState(balance=10_000)
    acc.symbols["BTC"] = ss
    ctx = TradeContext(
        symbol="BTC",
        side=Side.LONG,
        position=pos,
        account=acc,
        candle=Candle(0, 50_000, 50_000, 50_000, 50_000, 0),
        signal=TrendSignal(direction=0, strength=0, regime=TrendRegime.NEUTRAL),
        current_time_ms=0,
        exchange_params=ExchangeParams(),
    )
    market_exit = Order(
        symbol="BTC",
        side=Side.LONG,
        price=50_000.0,
        qty=0.01,
        source=OrderSource.GRID,
        reduce_only=True,
        is_market=True,
    )
    out = runner.filter_exits([market_exit], ctx)
    assert custom_called["count"] == 0, (
        "custom_exit_price must NOT be invoked for market exits — "
        "the price the strategy sees would be a lie (live uses None)"
    )
    # Price was NOT adjusted by +999.
    assert out[0].price == pytest.approx(50_000.0)
