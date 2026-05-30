"""Round-24 tests:

* Per-side active gating: an ``enter_long`` signal must NOT activate
  the same symbol's short grid entries (P0).
* ``compute_active_sides`` honors n_positions per side independently.
* Open positions are always active on their side, even when forced
  exceeds n_positions.
* Strategy entry candidates are preferred over Forager picks for
  remaining budget slots.
* ``bot_loop_start`` fires AFTER data refresh / equity update so the
  strategy sees the current bar (backtest + live parity with
  Freqtrade).
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

# ────────────────────────────────────────────────────────────────────
# P0: per-side active gating
# ────────────────────────────────────────────────────────────────────


def test_enter_long_does_not_activate_short_grid_on_same_symbol():
    """The crucial Round-24 P0 fix: a Freqtrade ``enter_long`` signal
    forces the LONG side of SOL active, but the SHORT side must stay
    inactive. The pre-fix code used a single symbol-level active_set
    that let `grid_short` entries slip through too.
    """
    from combo_bot.grid_engine import (
        ForagerWeights,
        compute_active_sides,
    )
    from combo_bot.types import (
        AccountState,
        Candle,
        SymbolState,
        TrendRegime,
        TrendSignal,
    )

    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
    account = AccountState(balance=10_000.0)
    for s in symbols:
        account.symbols[s] = SymbolState(symbol=s)
    candles = {
        s: Candle(
            timestamp=0, open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0
        )
        for s in symbols
    }
    # Give BTC/ETH huge volume so they win on Forager ranking.
    candles["BTC/USDT:USDT"] = Candle(
        timestamp=0, open=100.0, high=100.0, low=100.0, close=100.0, volume=10_000.0
    )
    candles["ETH/USDT:USDT"] = Candle(
        timestamp=0, open=100.0, high=100.0, low=100.0, close=100.0, volume=9_000.0
    )
    signals = {
        s: TrendSignal(direction=0.0, strength=0.0, regime=TrendRegime.NEUTRAL)
        for s in symbols
    }
    # Strategy fires enter_long ONLY on SOL. enter_short is False.
    strategy = {
        "BTC/USDT:USDT": (False, False, False, False),
        "ETH/USDT:USDT": (False, False, False, False),
        "SOL/USDT:USDT": (True, False, False, False),
    }
    long_set, short_set = compute_active_sides(
        symbols=symbols,
        account=account,
        candles=candles,
        signals=signals,
        strategy_signals=strategy,
        n_positions=2,
        weights=ForagerWeights(),
    )
    assert (
        "SOL/USDT:USDT" in long_set
    ), f"enter_long must put SOL in active_long_set; got {long_set}"
    assert "SOL/USDT:USDT" not in short_set, (
        f"enter_long must NOT activate the short side of SOL; "
        f"got short_set={short_set}"
    )


# ────────────────────────────────────────────────────────────────────
# n_positions is per-side independent
# ────────────────────────────────────────────────────────────────────


def test_n_positions_independent_per_side():
    """LONG and SHORT each get their own n_positions budget — 7 longs
    + 7 shorts is fine even with n_positions=7 because the budgets
    are independent."""
    from combo_bot.grid_engine import ForagerWeights, compute_active_sides
    from combo_bot.types import (
        AccountState,
        Candle,
        Position,
        SymbolState,
        TrendRegime,
        TrendSignal,
    )

    symbols = [f"PAIR{i}/USDT:USDT" for i in range(10)]
    account = AccountState(balance=100_000.0)
    for s in symbols:
        account.symbols[s] = SymbolState(symbol=s)
    # PAIR0..PAIR6 hold open longs.
    for i in range(7):
        account.symbols[symbols[i]].position_long = Position(
            size=0.1, entry_price=100.0
        )
    # PAIR0..PAIR6 ALSO hold open shorts (hedge).
    for i in range(7):
        account.symbols[symbols[i]].position_short = Position(
            size=0.1, entry_price=100.0
        )
    candles = {
        s: Candle(
            timestamp=0, open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0
        )
        for s in symbols
    }
    signals = {
        s: TrendSignal(direction=0.0, strength=0.0, regime=TrendRegime.NEUTRAL)
        for s in symbols
    }
    strategy = {s: (False, False, False, False) for s in symbols}
    long_set, short_set = compute_active_sides(
        symbols=symbols,
        account=account,
        candles=candles,
        signals=signals,
        strategy_signals=strategy,
        n_positions=7,
        weights=ForagerWeights(),
    )
    assert len(long_set) == 7, (
        f"long side has 7 open positions, should fill the long budget; "
        f"got {len(long_set)}"
    )
    assert len(short_set) == 7, (
        f"short side has 7 open positions, should fill the short budget; "
        f"got {len(short_set)}"
    )
    # No new symbols added since both sides are at the cap.
    assert long_set == set(symbols[:7])
    assert short_set == set(symbols[:7])


def test_open_positions_always_active_even_if_above_n_positions():
    """If forced opens exceed n_positions on one side, all stay
    active on that side (existing risk must be managed). The OTHER
    side allocates from its own independent budget normally — Forager
    fills its slots with the top-ranked symbols."""
    from combo_bot.grid_engine import ForagerWeights, compute_active_sides
    from combo_bot.types import (
        AccountState,
        Candle,
        Position,
        SymbolState,
        TrendRegime,
        TrendSignal,
    )

    symbols = [f"PAIR{i}/USDT:USDT" for i in range(8)]
    account = AccountState(balance=100_000.0)
    for s in symbols:
        account.symbols[s] = SymbolState(symbol=s)
    # 8 open longs but n_positions=5 → all 8 must stay active on the
    # long side. Strategy enters NONE; Forager would otherwise pick
    # the top 5 by score but the long budget is already 8/5 (over) so
    # nothing further joins long_set.
    for i in range(8):
        account.symbols[symbols[i]].position_long = Position(
            size=0.1, entry_price=100.0
        )
    candles = {
        s: Candle(
            timestamp=0, open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0
        )
        for s in symbols
    }
    signals = {
        s: TrendSignal(direction=0.0, strength=0.0, regime=TrendRegime.NEUTRAL)
        for s in symbols
    }
    strategy = {s: (False, False, False, False) for s in symbols}
    long_set, short_set = compute_active_sides(
        symbols=symbols,
        account=account,
        candles=candles,
        signals=signals,
        strategy_signals=strategy,
        n_positions=5,
        weights=ForagerWeights(),
    )
    assert long_set == set(
        symbols
    ), f"all 8 open longs must stay active on the long side; got {long_set}"
    # Short side has no opens → Forager fills the full short budget
    # of 5 picks. This is Passivbot's "both-side grid" default — every
    # active symbol grids both sides unless a side-specific signal
    # restricts it. The crucial invariant under test is that long-side
    # over-fill (8) doesn't shrink short-side allocation (which is
    # entirely independent and capped at n_positions=5).
    assert len(short_set) == 5, (
        f"short side budget is independent at n_positions=5; got "
        f"{len(short_set)} (long over-fill must not steal short slots)"
    )


def test_strategy_entries_preferred_over_forager_for_remaining_budget():
    """When there's open-position slack, strategy enter signals
    consume the remaining budget BEFORE the Forager fills it."""
    from combo_bot.grid_engine import ForagerWeights, compute_active_sides
    from combo_bot.types import (
        AccountState,
        Candle,
        Position,
        SymbolState,
        TrendRegime,
        TrendSignal,
    )

    symbols = [f"PAIR{i}/USDT:USDT" for i in range(5)]
    account = AccountState(balance=100_000.0)
    for s in symbols:
        account.symbols[s] = SymbolState(symbol=s)
    # PAIR0 holds an open long.
    account.symbols[symbols[0]].position_long = Position(size=0.1, entry_price=100.0)
    # PAIR4 has huge volume → Forager would pick it first.
    candles = {
        s: Candle(
            timestamp=0, open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0
        )
        for s in symbols
    }
    candles[symbols[4]] = Candle(
        timestamp=0, open=100.0, high=100.0, low=100.0, close=100.0, volume=10_000.0
    )
    signals = {
        s: TrendSignal(direction=0.0, strength=0.0, regime=TrendRegime.NEUTRAL)
        for s in symbols
    }
    # Strategy fires enter_long on PAIR2.
    strategy = {s: (False, False, False, False) for s in symbols}
    strategy[symbols[2]] = (True, False, False, False)
    long_set, _short = compute_active_sides(
        symbols=symbols,
        account=account,
        candles=candles,
        signals=signals,
        strategy_signals=strategy,
        n_positions=2,
        weights=ForagerWeights(),
    )
    # n_positions=2, open=PAIR0, strategy=PAIR2 → budget exhausted,
    # PAIR4 (Forager pick) must NOT make it in.
    assert long_set == {symbols[0], symbols[2]}, (
        f"strategy enter must beat Forager for the remaining slot; " f"got {long_set}"
    )


# ────────────────────────────────────────────────────────────────────
# bot_loop_start ordering
# ────────────────────────────────────────────────────────────────────


def test_bot_loop_start_in_backtest_sees_current_bar():
    """Strategy reading DataProvider in bot_loop_start must see the
    bar that just arrived, not the previous one — matches Freqtrade
    refresh→bot_loop_start→analyze ordering."""
    pd = pytest.importorskip("pandas")
    from combo_bot.backtest import BacktestConfig, Backtester
    from combo_bot.strategy import IStrategy
    from combo_bot.types import Candle

    seen: list[int] = []

    class _PeekingStrat(IStrategy):
        def __init__(self, ref):
            self._ref = ref

        def populate_indicators(self, df, m):
            return df

        def populate_entry_trend(self, df, m):
            return df

        def populate_exit_trend(self, df, m):
            return df

        def bot_loop_start(self, current_time, **kwargs):
            df = self._ref.data_provider.get_dataframe("BTC/USDT:USDT")
            if len(df) > 0:
                seen.append(int(df["timestamp"].iloc[-1]))

    cfg = BacktestConfig(
        starting_balance=10_000.0,
        symbols=["BTC/USDT:USDT"],
        bar_interval_minutes=60.0,
    )
    bt = Backtester(cfg, strategy=_PeekingStrat.__new__(_PeekingStrat))
    bt.strategy = _PeekingStrat(bt)
    candles = [
        Candle(
            timestamp=(i + 1) * 3_600_000,
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=1.0,
        )
        for i in range(3)
    ]
    bt.run({"BTC/USDT:USDT": candles})
    # Three bars, three hook calls. Each call must see the CURRENT
    # bar (the one just appended), not the previous one. The pre-fix
    # ordering ran bot_loop_start BEFORE data_provider.append on tick
    # N, so the strategy would see N-1 (or nothing on tick 0).
    assert seen == [
        3_600_000,
        7_200_000,
        10_800_000,
    ], f"bot_loop_start must see the latest bar timestamp; got {seen}"
    _ = pd


def test_bot_loop_start_in_live_runs_after_data_refresh():
    """In live, bot_loop_start must run AFTER _refresh_candles so the
    strategy sees the freshest data, not the previous tick's."""
    from combo_bot.live import LiveConfig, LiveTrader
    from combo_bot.strategy import IStrategy

    order: list[str] = []

    class _SequenceStrat(IStrategy):
        def populate_indicators(self, df, m):
            return df

        def populate_entry_trend(self, df, m):
            return df

        def populate_exit_trend(self, df, m):
            return df

        def bot_loop_start(self, current_time, **kwargs):
            order.append("bot_loop_start")

    class _Stub:
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
            order.append("fetch_positions")
            return []

        async def fetch_funding_rate(self, _):
            return {"fundingRate": 0.0}

        async def fetch_ohlcv(self, *a, **k):
            order.append("fetch_ohlcv")
            return []

        async def fetch_my_trades(self, *a, **k):
            return []

        async def fetch_open_orders(self, _):
            return []

        async def set_leverage(self, *a, **k):
            return {}

        async def set_margin_mode(self, *a, **k):
            return {}

    with tempfile.TemporaryDirectory() as tmp:
        cfg = LiveConfig(
            symbols=["BTC/USDT:USDT"],
            dry_run=True,
            state_file=str(Path(tmp) / "state.json"),
        )
        trader = LiveTrader(cfg, _Stub(), strategy=_SequenceStrat())

        async def drive():
            await trader._init_exchange()
            await trader._load_state()
            await trader._tick()

        asyncio.run(drive())

    # bot_loop_start MUST appear after at least one data-refresh call
    # (fetch_positions or fetch_ohlcv). Pre-fix ordering had
    # bot_loop_start BEFORE both.
    assert "bot_loop_start" in order, "hook must have fired"
    bls_idx = order.index("bot_loop_start")
    # Find any data-refresh marker before it.
    data_refresh_before = any(
        marker in order[:bls_idx] for marker in ("fetch_positions", "fetch_ohlcv")
    )
    assert data_refresh_before, (
        f"bot_loop_start must run AFTER at least one data refresh; "
        f"observed order={order}"
    )
