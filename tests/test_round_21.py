"""Round-21 tests:

* fill sink failure rolls back watermark + seen + sets last_poll_failed
* populate_* signal columns can update after first creation
* market entry skips custom_entry_price
* custom_stake_amount honors c_mult
"""

from __future__ import annotations

import asyncio

import pytest

# ────────────────────────────────────────────────────────────────────
# P0: sink failure rolls back
# ────────────────────────────────────────────────────────────────────


def test_sink_failure_rolls_back_watermark_and_seen_and_marks_failed():
    """If sink raises during a poll, the trades are NOT marked seen
    and the watermark does NOT advance. last_poll_failed is set so
    live blocks new entries this tick. The NEXT poll re-fetches the
    same trades, dedup catches any that already made it through, and
    sink gets another chance."""
    from combo_bot.fill_events_manager import (
        FillEventManager,
        FillEventManagerConfig,
    )

    class _Ex:
        async def fetch_my_trades(self, *a, **k):
            return [
                {
                    "id": "t-1",
                    "timestamp": 1_000,
                    "side": "buy",
                    "price": 50.0,
                    "amount": 0.001,
                    "fee": {"cost": 0.0},
                    "order": "o",
                    "info": {},
                }
            ]

    mgr = FillEventManager(
        _Ex(),
        FillEventManagerConfig(poll_interval_ms=0),
    )
    sym = "BTC/USDT:USDT"

    def boom_sink(_fills):
        raise RuntimeError("downstream blew up")

    asyncio.run(mgr.poll(sym, now_ms=0, sink=boom_sink))

    # Watermark must NOT have advanced — next poll re-fetches.
    assert sym not in mgr._last_ts_ms, (
        "sink failure must NOT commit the watermark; got "
        f"_last_ts_ms[{sym}]={mgr._last_ts_ms.get(sym)}"
    )
    # Trade id must NOT be marked seen.
    assert "t-1" not in mgr._seen_ids.get(sym, []), (
        "sink failure must NOT commit seen ids; got " f"{mgr._seen_ids.get(sym)}"
    )
    # last_poll_failed flags this tick fail-closed.
    assert mgr.last_poll_failed(sym) is True


def test_sink_failure_recovery_on_next_poll_does_not_double_send():
    """After a rolled-back poll, the next poll re-fetches the same
    trades. If sink succeeds this time, those trades land in the
    ledger exactly once."""
    from combo_bot.fill_events_manager import (
        FillEventManager,
        FillEventManagerConfig,
    )

    class _Ex:
        async def fetch_my_trades(self, *a, **k):
            return [
                {
                    "id": "t-1",
                    "timestamp": 1_000,
                    "side": "buy",
                    "price": 50.0,
                    "amount": 0.001,
                    "fee": {"cost": 0.0},
                    "order": "o",
                    "info": {},
                }
            ]

    mgr = FillEventManager(
        _Ex(),
        FillEventManagerConfig(poll_interval_ms=0),
    )
    sym = "BTC/USDT:USDT"
    asyncio.run(
        mgr.poll(sym, now_ms=0, sink=lambda _: (_ for _ in ()).throw(RuntimeError("x")))
    )
    # Recovery: clean sink. Should see t-1 exactly once.
    captured = []
    asyncio.run(mgr.poll(sym, now_ms=1, sink=captured.extend))
    assert len(captured) == 1
    assert captured[0].trade_id == "t-1"
    # last_poll_failed clears.
    assert mgr.last_poll_failed(sym) is False


# ────────────────────────────────────────────────────────────────────
# P1: populate_* signal column refresh
# ────────────────────────────────────────────────────────────────────


def test_populate_signal_columns_refresh_when_returned_dataframe_changes_values():
    """Strategy returns a NEW dataframe; signal column values must
    REPLACE the cached values, not be silently dropped because the
    column already exists in the cache."""
    pytest.importorskip("pandas")
    from combo_bot.backtest import BacktestConfig, Backtester
    from combo_bot.data_provider import DataProvider
    from combo_bot.strategy import IStrategy
    from combo_bot.types import Candle

    class _ShiftingSignals(IStrategy):
        tick = 0

        def populate_indicators(self, df, m):
            return df

        def populate_entry_trend(self, df, m):
            # Returns a NEW dataframe; signal flips every tick.
            new = df.copy()
            type(self).tick += 1
            new["enter_long"] = type(self).tick % 2  # 1, 0, 1, 0...
            return new

        def populate_exit_trend(self, df, m):
            return df

    cfg = BacktestConfig(starting_balance=10_000.0)
    bt = Backtester(cfg, strategy=_ShiftingSignals())
    bt.data_provider = DataProvider()
    sym = "BTC/USDT:USDT"
    bt.data_provider.append(
        sym,
        Candle(timestamp=1, open=100, high=101, low=99, close=100, volume=1),
    )
    bt._apply_strategy_populates(sym)
    df1 = bt.data_provider.get_dataframe(sym)
    first = int(df1["enter_long"].iloc[-1])
    bt.data_provider.append(
        sym,
        Candle(timestamp=2, open=100, high=101, low=99, close=100, volume=1),
    )
    bt._apply_strategy_populates(sym)
    df2 = bt.data_provider.get_dataframe(sym)
    second = int(df2["enter_long"].iloc[-1])
    # Tick 1 returned 1; tick 2 returned 0. With the fix the cache
    # reflects the most recent populate's value on the latest row.
    assert (first, second) == (
        1,
        0,
    ), f"signal column must refresh; saw (first={first}, second={second})"


# ────────────────────────────────────────────────────────────────────
# P1: market entry skips custom_entry_price
# ────────────────────────────────────────────────────────────────────


def test_market_entry_skips_custom_entry_price():
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
        TrendRegime,
        TrendSignal,
    )

    called = {"price": 0}

    class _S(IStrategy):
        def populate_indicators(self, df, m):
            return df

        def populate_entry_trend(self, df, m):
            return df

        def populate_exit_trend(self, df, m):
            return df

        def custom_entry_price(self, ctx, proposed):
            called["price"] += 1
            return proposed + 999.0

    runner = StrategyRunner(_S())
    pos = Position(size=0.0, entry_price=0.0)
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
    market_entry = Order(
        symbol="BTC",
        side=Side.LONG,
        price=49_000.0,  # would-be ref price, will be ignored on market
        qty=0.01,
        source=OrderSource.TREND,
        is_market=True,
    )
    out = runner.filter_entries([market_entry], ctx)
    assert (
        called["price"] == 0
    ), "custom_entry_price must NOT be invoked for market entries"
    # The order's stated price stays put (not bumped by +999).
    assert out[0].price == pytest.approx(49_000.0)


# ────────────────────────────────────────────────────────────────────
# P1: custom_stake_amount honors c_mult
# ────────────────────────────────────────────────────────────────────


def test_custom_stake_amount_passes_notional_with_c_mult():
    """For a c_mult=10 contract, proposed_stake passed to
    custom_stake_amount must be qty * price * 10, not qty * price."""
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
        TrendRegime,
        TrendSignal,
    )

    seen_stake: list[float] = []

    class _S(IStrategy):
        def populate_indicators(self, df, m):
            return df

        def populate_entry_trend(self, df, m):
            return df

        def populate_exit_trend(self, df, m):
            return df

        def custom_stake_amount(self, ctx, proposed, min_stake, max_stake):
            seen_stake.append(proposed)
            return proposed  # no change

    runner = StrategyRunner(_S())
    pos = Position()
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
        exchange_params=ExchangeParams(c_mult=10.0),
    )
    entry = Order(
        symbol="BTC",
        side=Side.LONG,
        price=50_000.0,
        qty=0.01,
        source=OrderSource.GRID,
    )
    out = runner.filter_entries([entry], ctx)
    # qty * price * c_mult = 0.01 * 50_000 * 10 = 5_000
    assert seen_stake == [pytest.approx(5_000.0)]
    # qty round-trips: stake / (price * c_mult) = 5000 / 500_000 = 0.01
    assert out[0].qty == pytest.approx(0.01)
