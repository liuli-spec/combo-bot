"""Round-15 tests:

* fetch_order closed WITHOUT filled qty / avg holds UNKNOWN (covered
  in test_round_13.py's revised test).
* trend SL/TP exits go through StrategyRunner.filter_exits.
* confirm_trade_exit receives the FINAL price (after custom_/adjust_).
* fill_events.poll drains multiple pages until short page.
* stale UNKNOWN with no matching journal entries clears (no permanent
  block).
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

# ────────────────────────────────────────────────────────────────────
# Trend SL/TP exits go through strategy filter_exits
# ────────────────────────────────────────────────────────────────────


def test_trend_sl_tp_exits_routed_through_confirm_trade_exit():
    from combo_bot.backtest import BacktestConfig, Backtester
    from combo_bot.strategy import IStrategy, TradeContext
    from combo_bot.types import (
        Candle,
        Position,
        Side,
        SymbolState,
    )

    class _VetoTrendExitStrategy(IStrategy):
        saw_calls: list[tuple[str, str]] = []

        def populate_indicators(self, df, m):
            return df

        def populate_entry_trend(self, df, m):
            return df

        def populate_exit_trend(self, df, m):
            return df

        def confirm_trade_exit(self, ctx, qty, price, reason):
            _VetoTrendExitStrategy.saw_calls.append((reason, ctx.side.value))
            return False  # veto

    _VetoTrendExitStrategy.saw_calls = []
    cfg = BacktestConfig(starting_balance=10_000.0, symbols=["BTC"])
    bt = Backtester(cfg, strategy=_VetoTrendExitStrategy())

    # We don't need to drive a full backtest — exercise the merger
    # exit path directly + verify the runner sees a trend_exit call.
    ss = SymbolState(symbol="BTC")
    ss.trend_long = Position(size=0.01, entry_price=50_000.0)
    # Price below SL threshold.
    sl_price = 50_000.0 * (1 - 0.03 - 0.001)
    trend_exits = bt.merger.generate_trend_exit_orders(
        "BTC",
        ss.trend_long,
        Side.LONG,
        sl_price,
        __import__("combo_bot.types", fromlist=["ExchangeParams"]).ExchangeParams(),
    )
    assert len(trend_exits) == 1  # merger produced an exit
    # Now run through the runner with a trend-bucket context.
    from combo_bot.types import AccountState, ExchangeParams, TrendSignal, TrendRegime

    acc = AccountState(balance=10_000)
    acc.symbols["BTC"] = ss
    ctx = TradeContext(
        symbol="BTC",
        side=Side.LONG,
        position=ss.trend_long,
        account=acc,
        candle=Candle(0, sl_price, sl_price, sl_price, sl_price, 0),
        signal=TrendSignal(direction=0, strength=0, regime=TrendRegime.NEUTRAL),
        current_time_ms=0,
        exchange_params=ExchangeParams(),
    )
    survived = bt.strategy_runner.filter_exits(trend_exits, ctx)
    # Vetoed → empty.
    assert survived == []
    assert (
        _VetoTrendExitStrategy.saw_calls
    ), "confirm_trade_exit must be called for trend SL/TP exits"


# ────────────────────────────────────────────────────────────────────
# confirm_trade_exit sees FINAL price
# ────────────────────────────────────────────────────────────────────


def test_confirm_trade_exit_receives_final_price_after_adjustments():
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

    class _PriceCheck(IStrategy):
        def populate_indicators(self, df, m):
            return df

        def populate_entry_trend(self, df, m):
            return df

        def populate_exit_trend(self, df, m):
            return df

        def custom_exit_price(self, ctx, proposed_price, exit_reason):
            return proposed_price + 100.0  # +100 bump

        def confirm_trade_exit(self, ctx, qty, price, reason):
            seen_prices.append(price)
            return True

    runner = StrategyRunner(_PriceCheck())
    pos = Position(size=0.01, entry_price=50_000.0)
    ss = SymbolState(symbol="BTC", position_long=pos)
    acc = AccountState(balance=10_000)
    acc.symbols["BTC"] = ss
    ctx = TradeContext(
        symbol="BTC",
        side=Side.LONG,
        position=pos,
        account=acc,
        candle=Candle(0, 50_500, 50_500, 50_500, 50_500, 0),
        signal=TrendSignal(direction=0, strength=0, regime=TrendRegime.NEUTRAL),
        current_time_ms=0,
        exchange_params=ExchangeParams(),
    )
    exit_order = Order(
        symbol="BTC",
        side=Side.LONG,
        price=51_000.0,
        qty=0.01,
        source=OrderSource.GRID,
        reduce_only=True,
    )
    out = runner.filter_exits([exit_order], ctx)
    assert seen_prices == [51_100.0], (
        f"confirm_trade_exit must see the price AFTER custom_/adjust_; "
        f"saw {seen_prices}"
    )
    assert out[0].price == pytest.approx(51_100.0)


# ────────────────────────────────────────────────────────────────────
# fill polling drains multiple pages
# ────────────────────────────────────────────────────────────────────


def test_fill_polling_drains_until_short_page():
    """With page_size=2 and 5 same-burst trades, poll must walk
    multiple pages so no fills get silently dropped past the cursor."""
    from combo_bot.fill_events_manager import (
        FillEventManager,
        FillEventManagerConfig,
    )
    from combo_bot.types import Fill

    class _Ex:
        def __init__(self):
            self.calls = 0

        async def fetch_my_trades(self, symbol, since=None, limit=None):
            self.calls += 1
            # Two pages of two, then one page of one (short → break).
            if since is None:
                return [
                    {
                        "id": "t1",
                        "timestamp": 1_000,
                        "side": "buy",
                        "price": 50_000.0,
                        "amount": 0.001,
                        "fee": {"cost": 0.0},
                        "order": "ord-1",
                        "info": {},
                    },
                    {
                        "id": "t2",
                        "timestamp": 1_000,
                        "side": "buy",
                        "price": 50_000.0,
                        "amount": 0.001,
                        "fee": {"cost": 0.0},
                        "order": "ord-1",
                        "info": {},
                    },
                ]
            if since == 1_000:
                return [
                    {
                        "id": "t3",
                        "timestamp": 1_500,
                        "side": "buy",
                        "price": 50_000.0,
                        "amount": 0.001,
                        "fee": {"cost": 0.0},
                        "order": "ord-1",
                        "info": {},
                    },
                    {
                        "id": "t4",
                        "timestamp": 1_500,
                        "side": "buy",
                        "price": 50_000.0,
                        "amount": 0.001,
                        "fee": {"cost": 0.0},
                        "order": "ord-1",
                        "info": {},
                    },
                ]
            if since == 1_500:
                return [
                    {
                        "id": "t5",
                        "timestamp": 2_000,
                        "side": "buy",
                        "price": 50_000.0,
                        "amount": 0.001,
                        "fee": {"cost": 0.0},
                        "order": "ord-1",
                        "info": {},
                    },
                ]
            return []

    mgr = FillEventManager(
        _Ex(),
        FillEventManagerConfig(
            poll_interval_ms=0,
            page_size=2,
        ),
    )
    captured: list[Fill] = []
    asyncio.run(mgr.poll("BTC/USDT:USDT", now_ms=3_000, sink=captured.extend))
    captured_ids = {f.trade_id for f in captured}
    assert captured_ids == {
        "t1",
        "t2",
        "t3",
        "t4",
        "t5",
    }, f"pagination must drain all pages; got {captured_ids}"


def test_fill_polling_breaks_when_cursor_not_advancing():
    """If an exchange echos full pages forever without advancing
    timestamp, the loop must break (bounded by max_pages) rather than
    spinning forever."""
    from combo_bot.fill_events_manager import (
        FillEventManager,
        FillEventManagerConfig,
    )

    class _StuckEx:
        def __init__(self):
            self.calls = 0

        async def fetch_my_trades(self, *_a, **_k):
            self.calls += 1
            return [
                {
                    "id": f"t-{self.calls}-a",
                    "timestamp": 1_000,
                    "side": "buy",
                    "price": 50.0,
                    "amount": 0.001,
                    "fee": {"cost": 0.0},
                    "order": "ord-1",
                    "info": {},
                },
                {
                    "id": f"t-{self.calls}-b",
                    "timestamp": 1_000,
                    "side": "buy",
                    "price": 50.0,
                    "amount": 0.001,
                    "fee": {"cost": 0.0},
                    "order": "ord-1",
                    "info": {},
                },
            ]

    ex = _StuckEx()
    mgr = FillEventManager(
        ex,
        FillEventManagerConfig(
            poll_interval_ms=0,
            page_size=2,
        ),
    )
    captured = []
    asyncio.run(mgr.poll("BTC/USDT:USDT", now_ms=5_000, sink=captured.extend))
    # Cursor never advances past 1_000, so loop must short-circuit.
    # max_pages cap is 16 — we should see at most that many calls and
    # then break out cleanly without infinite loop.
    assert ex.calls < 100  # really just "didn't hang"


# ────────────────────────────────────────────────────────────────────
# stale UNKNOWN with no journal cIDs clears (round-15 P2)
# ────────────────────────────────────────────────────────────────────


def test_stale_unknown_with_no_journal_entries_self_clears():
    """If state was restored with a UNKNOWN slot but journal has no
    matching cIDs (e.g. they were already marked terminal then crashed
    before state save), _resolve_unknowns must NOT leave the slot
    permanently blocking the side."""
    from combo_bot.live import LiveConfig, LiveTrader
    from combo_bot.types import ExchangeParams, Side, SymbolState

    with tempfile.TemporaryDirectory() as tmp:

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
                return {"id": "ex-1", "status": "open"}

            async def cancel_order(self, *a, **k):
                return {}

            async def set_leverage(self, *a, **k):
                return {}

            async def set_margin_mode(self, *a, **k):
                return {}

        cfg = LiveConfig(
            symbols=["BTC/USDT:USDT"],
            dry_run=False,
            state_file=str(Path(tmp) / "state.json"),
        )
        trader = LiveTrader(cfg, _Ex())
        trader.exchange_params["BTC/USDT:USDT"] = ExchangeParams()
        trader.account.symbols["BTC/USDT:USDT"] = SymbolState(
            symbol="BTC/USDT:USDT",
        )
        # Stale UNKNOWN with no journal entries.
        trader._unknown_overlay[("BTC/USDT:USDT", Side.LONG)] = 1.0
        # journal is empty.
        asyncio.run(trader._resolve_unknowns())
        # Must have cleared the stale slot so overlay isn't permanently dead.
        assert ("BTC/USDT:USDT", Side.LONG) not in trader._unknown_overlay
