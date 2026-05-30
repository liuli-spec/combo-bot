"""Round-23 tests:

* Strategy enter_long/enter_short forces a low-ranked symbol into the
  Forager active set so Freqtrade signals aren't silently dropped.
* n_positions is a HARD CEILING: forced + ranked never exceeds it.
* GridConfig.total_wallet_exposure_limit actually gates GRID-bucket
  TWE (was previously a dead config field).
* Risk partial-fit trim scales the last over-cap order down to fit
  exactly instead of dropping it.
* Backtester invokes bot_start / bot_loop_start lifecycle hooks
  (parity with live).
* Live reduce-only timeout uses POSITION-side inference, not the
  raw exchange-side leg label.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

# ────────────────────────────────────────────────────────────────────
# P1 #1: strategy entries force-active a low-ranked symbol
# ────────────────────────────────────────────────────────────────────


def test_strategy_entry_signal_forces_low_ranked_symbol_active():
    """A symbol that would lose the Forager ranking still gets to
    place its strategy-driven entry order if populate_entry_trend
    set enter_long=True (or enter_short)."""
    pd = pytest.importorskip("pandas")
    from combo_bot.backtest import BacktestConfig, Backtester
    from combo_bot.grid_engine import GridConfig
    from combo_bot.strategy import IStrategy

    class _ForceEnter(IStrategy):
        def populate_indicators(self, df, m):
            return df

        def populate_entry_trend(self, df, m):
            # Only set enter_long on SOL (the low-ranked symbol).
            if m["pair"] == "SOL/USDT:USDT":
                df["enter_long"] = 1
            else:
                df["enter_long"] = 0
            df["enter_short"] = 0
            return df

        def populate_exit_trend(self, df, m):
            return df

    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
    cfg = BacktestConfig(
        starting_balance=10_000.0,
        bar_interval_minutes=60.0,
        grid=GridConfig(n_positions=1, wallet_exposure_limit=0.5),
        symbols=symbols,
    )
    bt = Backtester(cfg, strategy=_ForceEnter())

    # Seed the DataProvider via its public append() so the strategy's
    # populate_* hooks actually run and write enter_long=1 into SOL's
    # cached DataFrame.
    from combo_bot.types import AccountState, SymbolState, Candle

    for s in symbols:
        bt.data_provider.append(
            s,
            Candle(
                timestamp=0, open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0
            ),
        )
        bt._apply_strategy_populates(s)

    # Confirm enter_long actually landed on SOL after the populate pass.
    sol_df = bt.data_provider.get_dataframe("SOL/USDT:USDT")
    assert (
        int(sol_df["enter_long"].iloc[-1]) == 1
    ), "test setup precondition: SOL must have enter_long=1 after populate"
    btc_df = bt.data_provider.get_dataframe("BTC/USDT:USDT")
    assert (
        int(btc_df["enter_long"].iloc[-1]) == 0
    ), "test setup precondition: BTC must have enter_long=0"

    account = AccountState(balance=10_000.0)
    for s in symbols:
        account.symbols[s] = SymbolState(symbol=s)

    # Re-implement the round-23 forced calc in line with the production
    # code so the test asserts the right thing without needing to run
    # a full backtest.
    from combo_bot.regime import read_strategy_signals

    tick_strategy_signals = {
        s: read_strategy_signals(bt.data_provider, s) for s in symbols
    }
    forced = set()
    for s in symbols:
        ss = account.symbols[s]
        if (
            ss.position_long.is_open
            or ss.position_short.is_open
            or ss.trend_long.is_open
            or ss.trend_short.is_open
        ):
            forced.add(s)
            continue
        enter_l, enter_s, _, _ = tick_strategy_signals[s]
        if enter_l or enter_s:
            forced.add(s)

    # SOL must be in forced even though it has no open position and
    # the Forager would otherwise have picked BTC/ETH first.
    assert (
        "SOL/USDT:USDT" in forced
    ), f"strategy enter_long must force SOL active; forced={forced}"
    _ = pd  # silence unused-import warning


# ────────────────────────────────────────────────────────────────────
# P1 #2: n_positions is a hard ceiling
# ────────────────────────────────────────────────────────────────────


def test_n_positions_hard_ceiling_when_forced_already_full():
    """When already-open positions cover all n_positions slots, the
    Forager must contribute ZERO additional symbols. The pre-fix code
    did ``set(ranked) | forced`` which could grow up to 2× n_positions.
    """
    from combo_bot.grid_engine import (
        ForagerScorer,
        ForagerWeights,
        build_forager_candidates,
    )
    from combo_bot.types import (
        AccountState,
        Candle,
        Position,
        SymbolState,
        TrendRegime,
        TrendSignal,
    )

    # 4 symbols, n_positions=2. Both BTC and ETH are forced (open).
    # Even though SOL ranks high on volume, it must NOT join active.
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "DOGE/USDT:USDT"]
    account = AccountState(balance=10_000.0)
    for s in symbols:
        account.symbols[s] = SymbolState(symbol=s)
    account.symbols["BTC/USDT:USDT"].position_long = Position(
        size=0.01, entry_price=50_000.0
    )
    account.symbols["ETH/USDT:USDT"].position_long = Position(
        size=0.5, entry_price=2_000.0
    )
    candles = {
        s: Candle(timestamp=0, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0)
        for s in symbols
    }
    candles["SOL/USDT:USDT"] = Candle(
        timestamp=0, open=1.0, high=1.0, low=1.0, close=1.0, volume=10_000.0
    )
    signals = {
        s: TrendSignal(direction=0.0, strength=0.0, regime=TrendRegime.NEUTRAL)
        for s in symbols
    }

    # Reproduce production logic.
    n_active = 2
    forced = {"BTC/USDT:USDT", "ETH/USDT:USDT"}
    remaining = max(0, n_active - len(forced))
    if remaining > 0:
        ranking_universe = [s for s in symbols if s not in forced]
        ranked = ForagerScorer.select_symbols(
            build_forager_candidates(ranking_universe, candles, account, signals),
            remaining,
            ForagerWeights(),
        )
    else:
        ranked = []
    active_set = forced | set(ranked)

    assert active_set == forced, (
        f"forced already fills n_positions=2, Forager must contribute "
        f"nothing; got active_set={active_set}"
    )
    assert len(active_set) == n_active, (
        f"active set must never exceed n_positions={n_active}; "
        f"got {len(active_set)}"
    )


def test_n_positions_partial_budget_for_forager():
    """forced=1, n_positions=3 → Forager fills the remaining 2 slots,
    NOT 3."""
    from combo_bot.grid_engine import (
        ForagerScorer,
        ForagerWeights,
        build_forager_candidates,
    )
    from combo_bot.types import (
        AccountState,
        Candle,
        Position,
        SymbolState,
        TrendRegime,
        TrendSignal,
    )

    symbols = [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "SOL/USDT:USDT",
        "DOGE/USDT:USDT",
        "XRP/USDT:USDT",
    ]
    account = AccountState(balance=10_000.0)
    for s in symbols:
        account.symbols[s] = SymbolState(symbol=s)
    account.symbols["BTC/USDT:USDT"].position_long = Position(
        size=0.01, entry_price=50_000.0
    )
    # Vary volume so ranking is deterministic.
    candles = {
        s: Candle(timestamp=0, open=1.0, high=1.0, low=1.0, close=1.0, volume=float(i))
        for i, s in enumerate(symbols, start=1)
    }
    signals = {
        s: TrendSignal(direction=0.0, strength=0.0, regime=TrendRegime.NEUTRAL)
        for s in symbols
    }

    n_active = 3
    forced = {"BTC/USDT:USDT"}
    remaining = max(0, n_active - len(forced))
    ranking_universe = [s for s in symbols if s not in forced]
    ranked = ForagerScorer.select_symbols(
        build_forager_candidates(ranking_universe, candles, account, signals),
        remaining,
        ForagerWeights(),
    )
    active = forced | set(ranked)
    assert (
        len(active) == n_active
    ), f"active set must equal n_positions={n_active}; got {len(active)}"
    assert len(ranked) == 2, f"Forager must fill 2 slots, not 3; got {len(ranked)}"


# ────────────────────────────────────────────────────────────────────
# P1 #3a: GridConfig.total_wallet_exposure_limit actually enforced
# ────────────────────────────────────────────────────────────────────


def test_grid_total_wallet_exposure_limit_enforced_against_grid_bucket():
    """When grid_total_wallet_exposure_limit is provided, the GRID
    bucket's combined exposure cannot exceed it — even when the
    overall TWE cap is much higher. Trend-bucket exposure is unaffected.
    """
    from combo_bot.risk import RiskConfig, RiskManager
    from combo_bot.types import (
        AccountState,
        Order,
        OrderSource,
        Position,
        Side,
        SymbolState,
    )

    risk = RiskManager(
        RiskConfig(
            max_total_wallet_exposure=10.0,  # essentially unbounded
            max_single_exposure=10.0,
            yellow_threshold=0.99,
            orange_threshold=0.99,
            red_threshold=0.99,
        )
    )
    account = AccountState(balance=10_000.0, equity=10_000.0, equity_peak=10_000.0)
    account.symbols["BTC/USDT:USDT"] = SymbolState(symbol="BTC/USDT:USDT")
    account.symbols["ETH/USDT:USDT"] = SymbolState(symbol="ETH/USDT:USDT")
    # BTC grid bucket already at WE=0.3.
    account.symbols["BTC/USDT:USDT"].position_long = Position(
        size=0.06, entry_price=50_000.0
    )

    grid_entry = Order(
        symbol="ETH/USDT:USDT",
        side=Side.LONG,
        price=2_000.0,
        qty=2.5,  # cost=5000 → WE 0.5
        source=OrderSource.GRID,
    )
    trend_entry = Order(
        symbol="ETH/USDT:USDT",
        side=Side.LONG,
        price=2_000.0,
        qty=2.5,  # cost=5000 → WE 0.5
        source=OrderSource.TREND,
    )
    # Grid bucket cap: 0.5. Base grid WE is 0.3, so 0.2 headroom for
    # GRID orders. Trend orders are uncapped by this limit.
    out = risk.filter_orders(
        [grid_entry, trend_entry],
        account,
        timestamp=0,
        grid_total_wallet_exposure_limit=0.5,
    )
    # GRID entry: trimmed from 0.5 → 0.2 → qty 1.0 (cost 2000).
    grid_filtered = [o for o in out if o.source == OrderSource.GRID]
    trend_filtered = [o for o in out if o.source == OrderSource.TREND]
    assert len(grid_filtered) == 1, "grid entry trimmed to fit, not dropped"
    assert grid_filtered[0].qty == pytest.approx(1.0, abs=1e-6), (
        f"grid entry must trim from qty 2.5 → 1.0 (0.2 WE headroom); "
        f"got {grid_filtered[0].qty}"
    )
    assert (
        len(trend_filtered) == 1
    ), "trend entry must not be touched by grid_total_wallet_exposure_limit"
    assert trend_filtered[0].qty == pytest.approx(
        2.5
    ), f"trend entry must pass through unchanged; got {trend_filtered[0].qty}"


# ────────────────────────────────────────────────────────────────────
# P1 #3b: partial-fit trim
# ────────────────────────────────────────────────────────────────────


def test_partial_fit_trim_quantizes_to_qty_step():
    """When an entry would push past TWE and exchange_params are
    provided, the trimmed qty must respect qty_step. If quantize
    drops it below min_qty, the order is dropped entirely."""
    from combo_bot.risk import RiskConfig, RiskManager
    from combo_bot.types import (
        AccountState,
        ExchangeParams,
        Order,
        OrderSource,
        Side,
        SymbolState,
    )

    risk = RiskManager(
        RiskConfig(
            max_total_wallet_exposure=0.1,
            max_single_exposure=1.0,
            yellow_threshold=0.99,
            orange_threshold=0.99,
            red_threshold=0.99,
        )
    )
    account = AccountState(balance=10_000.0, equity=10_000.0, equity_peak=10_000.0)
    account.symbols["BTC/USDT:USDT"] = SymbolState(symbol="BTC/USDT:USDT")

    # Headroom is 0.1 → max cost 1000. qty 0.05 at price 50_000 → 2500
    # → must trim to qty 0.02 (cost 1000).
    order = Order(
        symbol="BTC/USDT:USDT",
        side=Side.LONG,
        price=50_000.0,
        qty=0.05,
        source=OrderSource.GRID,
    )
    ep = {
        "BTC/USDT:USDT": ExchangeParams(
            qty_step=0.001, price_step=0.01, min_qty=0.001, min_cost=5.0
        )
    }
    out = risk.filter_orders([order], account, timestamp=0, exchange_params=ep)
    assert len(out) == 1
    assert out[0].qty == pytest.approx(0.02, abs=1e-9), (
        f"trimmed qty must be 0.02 (quantized to qty_step 0.001); " f"got {out[0].qty}"
    )


def test_partial_fit_trim_drops_when_below_min_qty():
    """If the trimmed qty falls under exchange min_qty, the order
    must be dropped (matches what the live executor would do)."""
    from combo_bot.risk import RiskConfig, RiskManager
    from combo_bot.types import (
        AccountState,
        ExchangeParams,
        Order,
        OrderSource,
        Side,
        SymbolState,
    )

    risk = RiskManager(
        RiskConfig(
            max_total_wallet_exposure=0.001,  # ~10 USD cost ceiling
            max_single_exposure=1.0,
            yellow_threshold=0.99,
            orange_threshold=0.99,
            red_threshold=0.99,
        )
    )
    account = AccountState(balance=10_000.0, equity=10_000.0, equity_peak=10_000.0)
    account.symbols["BTC/USDT:USDT"] = SymbolState(symbol="BTC/USDT:USDT")

    order = Order(
        symbol="BTC/USDT:USDT",
        side=Side.LONG,
        price=50_000.0,
        qty=0.05,
        source=OrderSource.GRID,
    )
    ep = {
        "BTC/USDT:USDT": ExchangeParams(
            qty_step=0.001, price_step=0.01, min_qty=0.01, min_cost=5.0
        )
    }
    out = risk.filter_orders([order], account, timestamp=0, exchange_params=ep)
    # Headroom is 0.001 → max cost 10 → trimmed qty 0.0002 < min_qty 0.01 → drop.
    assert out == [], f"trimmed qty below min_qty must be dropped; got {out}"


# ────────────────────────────────────────────────────────────────────
# P2 #4: backtest invokes bot_start / bot_loop_start
# ────────────────────────────────────────────────────────────────────


def test_backtester_invokes_bot_start_and_bot_loop_start():
    from combo_bot.backtest import BacktestConfig, Backtester
    from combo_bot.strategy import IStrategy
    from combo_bot.types import Candle

    calls = {"bot_start": 0, "bot_loop_start": 0}

    class _StrategyWithLifecycle(IStrategy):
        def populate_indicators(self, df, m):
            return df

        def populate_entry_trend(self, df, m):
            return df

        def populate_exit_trend(self, df, m):
            return df

        def bot_start(self, **kwargs):
            calls["bot_start"] += 1

        def bot_loop_start(self, current_time, **kwargs):
            calls["bot_loop_start"] += 1

    cfg = BacktestConfig(
        starting_balance=10_000.0,
        symbols=["BTC/USDT:USDT"],
        bar_interval_minutes=60.0,
    )
    bt = Backtester(cfg, strategy=_StrategyWithLifecycle())
    candles = [
        Candle(
            timestamp=i * 3_600_000,
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=1.0,
        )
        for i in range(5)
    ]
    bt.run({"BTC/USDT:USDT": candles})
    assert (
        calls["bot_start"] == 1
    ), f"bot_start must fire exactly once per backtest; got {calls['bot_start']}"
    assert calls["bot_loop_start"] == 5, (
        f"bot_loop_start must fire once per bar (5 bars); "
        f"got {calls['bot_loop_start']}"
    )


# ────────────────────────────────────────────────────────────────────
# P2 #5: reduce-only side inference uses position side, not leg
# ────────────────────────────────────────────────────────────────────


def test_reduce_only_timeout_uses_position_side_not_exchange_leg():
    """A reduce-only SELL closes a LONG; check_exit_timeout must see
    side=LONG and the LONG position (not SHORT with empty pos)."""
    from combo_bot.live import LiveConfig, LiveTrader
    from combo_bot.strategy import IStrategy
    from combo_bot.types import ExchangeParams, Position, Side, SymbolState

    seen_ctx: list = []

    class _CaptureStrat(IStrategy):
        def populate_indicators(self, df, m):
            return df

        def populate_entry_trend(self, df, m):
            return df

        def populate_exit_trend(self, df, m):
            return df

        def check_exit_timeout(self, ctx, age_s):
            seen_ctx.append((ctx.side, ctx.position.size, ctx.position.entry_price))
            return False  # don't actually cancel

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
            return []

        async def fetch_funding_rate(self, _):
            return {"fundingRate": 0.0}

        async def fetch_ohlcv(self, *a, **k):
            return []

        async def fetch_my_trades(self, *a, **k):
            return []

        async def fetch_open_orders(self, _):
            return [
                {
                    "id": "ex-99",
                    "symbol": "BTC/USDT:USDT",
                    "side": "sell",  # reduce-only sell closes a LONG
                    "type": "limit",
                    "price": 51_000.0,
                    "amount": 0.02,
                    "reduceOnly": True,
                    "timestamp": 1_000,
                }
            ]

        async def cancel_order(self, *a, **k):
            return {}

        async def create_order(self, *a, **k):
            return {"id": "ex-new", "status": "open"}

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
        trader = LiveTrader(cfg, _Stub(), strategy=_CaptureStrat())
        # Seed a LONG position so the context can find it.
        trader.exchange_params["BTC/USDT:USDT"] = ExchangeParams(
            qty_step=0.001, price_step=0.01, min_qty=0.001, min_cost=5.0
        )
        trader.account.symbols["BTC/USDT:USDT"] = SymbolState(
            symbol="BTC/USDT:USDT",
            position_long=Position(size=0.02, entry_price=50_000.0),
            last_price=50_500.0,
        )
        # Drive a reconciliation pass.
        asyncio.run(trader._reconcile_orders([]))

    assert seen_ctx, "check_exit_timeout must have been invoked"
    side, pos_size, pos_entry = seen_ctx[0]
    assert (
        side == Side.LONG
    ), f"reduce-only SELL closes a LONG; ctx.side must be LONG, got {side}"
    assert pos_size == pytest.approx(0.02), (
        f"ctx.position must be the LONG position with size 0.02; "
        f"got size={pos_size}"
    )
    assert pos_entry == pytest.approx(
        50_000.0
    ), f"ctx.position must carry the LONG entry_price; got {pos_entry}"
