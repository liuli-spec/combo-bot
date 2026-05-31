"""Round-27 tests:

* Per-entry strategy.leverage() is called with a real TradeContext
  and the result hits exchange.set_leverage only when it changes
  from the cached value.
* BacktestConfig.use_rust_grid routes grid math through the Rust
  adapter when available; falls back transparently otherwise.
* DataProvider.register_informative + append_informative +
  get_informative round-trip works.
* LiveTrader._refresh_informative_candles polls registered streams
  and writes them into the DataProvider.
* Backtester registers strategy.informative_pairs() so
  data_provider.get_informative(pair, tf) is safe to call from a
  strategy callback.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

# ────────────────────────────────────────────────────────────────────
# Per-entry leverage
# ────────────────────────────────────────────────────────────────────


def test_per_entry_leverage_hits_set_leverage_on_change_only():
    """Strategy.leverage() returning different values across two
    entries must trigger TWO set_leverage calls. Returning the SAME
    value across two entries must trigger only ONE (cached)."""
    from combo_bot.live import LiveConfig, LiveTrader
    from combo_bot.strategy import IStrategy
    from combo_bot.types import (
        ExchangeParams,
        Order,
        OrderSource,
        Side,
        SymbolState,
    )

    leverage_seq = [3.0, 3.0, 5.0]  # tick1=3, tick2=3 (cached), tick3=5
    captured: list[float] = []

    class _Strat(IStrategy):
        idx = 0

        def populate_indicators(self, df, m):
            return df

        def populate_entry_trend(self, df, m):
            return df

        def populate_exit_trend(self, df, m):
            return df

        def leverage(self, ctx, proposed_leverage, max_leverage):
            v = leverage_seq[min(type(self).idx, len(leverage_seq) - 1)]
            type(self).idx += 1
            return v

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

        async def set_leverage(self, lev, sym):
            captured.append(float(lev))

        async def set_margin_mode(self, *a, **k):
            return {}

        async def create_order(self, *a, **k):
            return {"id": "ex-1", "status": "open"}

    with tempfile.TemporaryDirectory() as tmp:
        cfg = LiveConfig(
            symbols=["BTC/USDT:USDT"],
            dry_run=False,
            leverage=10,
            state_file=str(Path(tmp) / "state.json"),
        )
        trader = LiveTrader(cfg, _Ex(), strategy=_Strat())
        asyncio.run(trader._init_exchange())
        # _init_exchange consumed idx=0 → leverage 3.0 cached.
        assert captured == [3.0], f"init_exchange leverage; got {captured}"

        # Tick 1: same value (3.0) → cached, no set_leverage call.
        o1 = Order(
            symbol="BTC/USDT:USDT",
            side=Side.LONG,
            price=50_000.0,
            qty=0.01,
            source=OrderSource.GRID,
        )
        asyncio.run(trader._ensure_leverage_for_entry(o1))
        assert captured == [
            3.0
        ], f"cached leverage must NOT re-call set_leverage; got {captured}"

        # Tick 2: idx now at 2 → returns 5.0 → must re-call.
        # Need to also seed the symbol state for context building.
        trader.exchange_params["BTC/USDT:USDT"] = ExchangeParams()
        trader.account.symbols["BTC/USDT:USDT"] = SymbolState(
            symbol="BTC/USDT:USDT", last_price=50_000.0
        )
        o2 = Order(
            symbol="BTC/USDT:USDT",
            side=Side.LONG,
            price=50_000.0,
            qty=0.01,
            source=OrderSource.GRID,
        )
        asyncio.run(trader._ensure_leverage_for_entry(o2))
        assert captured == [
            3.0,
            5.0,
        ], f"new leverage value must re-call set_leverage; got {captured}"


def test_per_entry_leverage_skips_for_reduce_only_orders():
    """Reduce-only exits must NOT change leverage — the existing
    position is being closed under whatever leverage it opened at."""
    from combo_bot.live import LiveConfig, LiveTrader
    from combo_bot.strategy import IStrategy
    from combo_bot.types import (
        ExchangeParams,
        Order,
        OrderSource,
        Side,
        SymbolState,
    )

    captured: list[float] = []

    class _Strat(IStrategy):
        def populate_indicators(self, df, m):
            return df

        def populate_entry_trend(self, df, m):
            return df

        def populate_exit_trend(self, df, m):
            return df

        def leverage(self, ctx, proposed_leverage, max_leverage):
            captured.append(99.0)  # would crash test if called for exit
            return 5.0

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

        async def set_leverage(self, *a, **k):
            return {}

        async def set_margin_mode(self, *a, **k):
            return {}

    with tempfile.TemporaryDirectory() as tmp:
        cfg = LiveConfig(
            symbols=["BTC/USDT:USDT"],
            dry_run=False,
            state_file=str(Path(tmp) / "state.json"),
        )
        trader = LiveTrader(cfg, _Ex(), strategy=_Strat())
        asyncio.run(trader._init_exchange())  # consumes one call

        trader.exchange_params["BTC/USDT:USDT"] = ExchangeParams()
        trader.account.symbols["BTC/USDT:USDT"] = SymbolState(
            symbol="BTC/USDT:USDT", last_price=50_000.0
        )
        baseline_calls = len(captured)
        exit_order = Order(
            symbol="BTC/USDT:USDT",
            side=Side.LONG,
            price=50_000.0,
            qty=0.01,
            source=OrderSource.GRID,
            reduce_only=True,
        )
        asyncio.run(trader._ensure_leverage_for_entry(exit_order))
        assert len(captured) == baseline_calls, (
            f"reduce-only must NOT invoke leverage hook; "
            f"call count went {baseline_calls} -> {len(captured)}"
        )


# ────────────────────────────────────────────────────────────────────
# Hybrid Rust grid acceleration
# ────────────────────────────────────────────────────────────────────


def test_use_rust_grid_falls_back_gracefully_when_extension_missing():
    """``use_rust_grid=True`` with no Rust extension installed must
    silently fall back to the Python grid path (Backtester just sets
    the flag to False internally) — the run must still complete."""
    from combo_bot.backtest import BacktestConfig, Backtester
    from combo_bot.types import Candle, ExchangeParams

    cfg = BacktestConfig(
        starting_balance=10_000.0,
        symbols=["BTC/USDT:USDT"],
        bar_interval_minutes=60.0,
        use_rust_grid=True,
    )
    bt = Backtester(cfg)
    # The flag must be either honored (Rust present) or downgraded
    # (Rust missing). In either case the run completes without
    # raising — that's the contract.
    candles = {
        "BTC/USDT:USDT": [
            Candle(i * 3_600_000, 100.0, 101.0, 99.0, 100.0, 1.0) for i in range(10)
        ]
    }
    ep = {"BTC/USDT:USDT": ExchangeParams(qty_step=0.001, min_qty=0.001, min_cost=5.0)}
    result = bt.run(candles, exchange_params=ep)
    assert result is not None
    # _rust_grid_active reflects whether the Rust path actually engaged.
    # It's False on test infrastructure without the wheel; True if the
    # wheel is installed. Either way the run completes.
    assert isinstance(bt._rust_grid_active, bool)


def test_use_rust_grid_false_is_default_python_path():
    """Without the flag, Backtester never touches the Rust path."""
    from combo_bot.backtest import BacktestConfig, Backtester

    cfg = BacktestConfig()
    bt = Backtester(cfg)
    assert bt._rust_grid_active is False


# ────────────────────────────────────────────────────────────────────
# DataProvider informative API
# ────────────────────────────────────────────────────────────────────


def test_data_provider_informative_round_trip():
    """register_informative + append_informative + get_informative
    must produce a non-empty DataFrame in canonical OHLCV column order."""
    pytest.importorskip("pandas")
    from combo_bot.data_provider import DataProvider
    from combo_bot.types import Candle

    dp = DataProvider()
    dp.register_informative("BTC/USDT:USDT", "1h")
    assert ("BTC/USDT:USDT", "1h") in dp.informative_pairs()

    for i in range(3):
        dp.append_informative(
            "BTC/USDT:USDT",
            "1h",
            Candle(
                timestamp=i * 3_600_000,
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=1.0,
            ),
        )
    df = dp.get_informative("BTC/USDT:USDT", "1h")
    assert len(df) == 3
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    # Primary buffer is independent — must still be empty.
    assert dp.buffer_len("BTC/USDT:USDT") == 0


def test_data_provider_get_informative_returns_empty_for_unregistered():
    pytest.importorskip("pandas")
    from combo_bot.data_provider import DataProvider

    dp = DataProvider()
    df = dp.get_informative("BTC/USDT:USDT", "1d")
    assert len(df) == 0
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_data_provider_append_informative_auto_registers():
    """Appending a candle without an explicit register call still
    registers the stream so informative_pairs() returns it."""
    pytest.importorskip("pandas")
    from combo_bot.data_provider import DataProvider
    from combo_bot.types import Candle

    dp = DataProvider()
    dp.append_informative(
        "ETH/USDT:USDT",
        "4h",
        Candle(timestamp=0, open=2_000, high=2_010, low=1_990, close=2_000, volume=1.0),
    )
    assert ("ETH/USDT:USDT", "4h") in dp.informative_pairs()


# ────────────────────────────────────────────────────────────────────
# Live informative polling
# ────────────────────────────────────────────────────────────────────


def test_live_refresh_informative_pulls_registered_stream():
    """LiveTrader._refresh_informative_candles must call
    fetch_ohlcv with the registered pair+timeframe and append the
    returned bars to the DataProvider's informative buffer."""
    pytest.importorskip("pandas")
    from combo_bot.live import LiveConfig, LiveTrader
    from combo_bot.strategy import IStrategy

    fetch_calls: list[tuple] = []

    class _Strat(IStrategy):
        def populate_indicators(self, df, m):
            return df

        def populate_entry_trend(self, df, m):
            return df

        def populate_exit_trend(self, df, m):
            return df

        def informative_pairs(self):
            return [("BTC/USDT:USDT", "4h")]

    class _Ex:
        async def fetch_ohlcv(self, pair, timeframe, limit=100):
            fetch_calls.append((pair, timeframe))
            return [
                [3_600_000, 100.0, 101.0, 99.0, 100.5, 1.0],
                [7_200_000, 100.5, 102.0, 100.0, 101.5, 2.0],
            ]

    with tempfile.TemporaryDirectory() as tmp:
        cfg = LiveConfig(
            symbols=["BTC/USDT:USDT"],
            dry_run=True,
            state_file=str(Path(tmp) / "state.json"),
        )
        trader = LiveTrader(cfg, _Ex(), strategy=_Strat())
        # Register the stream manually (start() does it via
        # informative_pairs hook; here we test the refresh in isolation).
        trader.data_provider.register_informative("BTC/USDT:USDT", "4h")
        asyncio.run(trader._refresh_informative_candles())

    assert ("BTC/USDT:USDT", "4h") in fetch_calls, (
        f"_refresh_informative must call fetch_ohlcv for the registered "
        f"stream; got {fetch_calls}"
    )
    df = trader.data_provider.get_informative("BTC/USDT:USDT", "4h")
    assert len(df) == 2, f"two bars appended; got {len(df)}"
    assert float(df["close"].iloc[-1]) == 101.5


def test_live_refresh_informative_dedupes_across_ticks():
    """A second poll that returns the same bars must NOT double-append.
    Per-(pair, timeframe) watermark dedupes by timestamp."""
    pytest.importorskip("pandas")
    from combo_bot.live import LiveConfig, LiveTrader
    from combo_bot.strategy import IStrategy

    class _Strat(IStrategy):
        def populate_indicators(self, df, m):
            return df

        def populate_entry_trend(self, df, m):
            return df

        def populate_exit_trend(self, df, m):
            return df

    class _Ex:
        async def fetch_ohlcv(self, pair, timeframe, limit=100):
            return [
                [3_600_000, 100.0, 101.0, 99.0, 100.5, 1.0],
                [7_200_000, 100.5, 102.0, 100.0, 101.5, 2.0],
            ]

    with tempfile.TemporaryDirectory() as tmp:
        cfg = LiveConfig(
            symbols=["BTC/USDT:USDT"],
            dry_run=True,
            state_file=str(Path(tmp) / "state.json"),
        )
        trader = LiveTrader(cfg, _Ex(), strategy=_Strat())
        trader.data_provider.register_informative("BTC/USDT:USDT", "4h")
        asyncio.run(trader._refresh_informative_candles())
        asyncio.run(trader._refresh_informative_candles())

    df = trader.data_provider.get_informative("BTC/USDT:USDT", "4h")
    assert len(df) == 2, (
        f"dedupe must prevent double-append on repeated identical pages; "
        f"got {len(df)}"
    )


# ────────────────────────────────────────────────────────────────────
# Backtester registers informative pairs
# ────────────────────────────────────────────────────────────────────


def test_backtester_registers_strategy_informative_pairs():
    """When a strategy declares informative_pairs() at bot_start,
    the Backtester must register them in the DataProvider so that
    get_informative(pair, tf) returns the expected (empty) frame
    rather than raising."""
    pytest.importorskip("pandas")
    from combo_bot.backtest import BacktestConfig, Backtester
    from combo_bot.strategy import IStrategy
    from combo_bot.types import Candle

    class _Strat(IStrategy):
        def populate_indicators(self, df, m):
            return df

        def populate_entry_trend(self, df, m):
            return df

        def populate_exit_trend(self, df, m):
            return df

        def informative_pairs(self):
            return [("BTC/USDT:USDT", "1d"), ("ETH/USDT:USDT", "1h")]

    cfg = BacktestConfig(
        starting_balance=10_000.0,
        symbols=["BTC/USDT:USDT"],
        bar_interval_minutes=60.0,
    )
    bt = Backtester(cfg, strategy=_Strat())
    candles = {
        "BTC/USDT:USDT": [
            Candle(i * 3_600_000, 100.0, 101.0, 99.0, 100.0, 1.0) for i in range(3)
        ]
    }
    bt.run(candles)
    registered = bt.data_provider.informative_pairs()
    assert (
        "BTC/USDT:USDT",
        "1d",
    ) in registered, f"BTC informative stream must register; got {registered}"
    assert (
        "ETH/USDT:USDT",
        "1h",
    ) in registered, f"ETH informative stream must register; got {registered}"
