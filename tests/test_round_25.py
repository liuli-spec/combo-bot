"""Round-25 tests:

* bot_loop_start fires BEFORE populate_* in both backtest and live
  so any external state the hook sets is visible to populate
  (Freqtrade refresh→bot_loop_start→analyze ordering).
* ``n_positions=0`` truly disables new entries on both sides;
  open positions remain active for management.
* Risk TWEL sort is by distance-from-mark (closer-to-market entries
  fill first under tight budget) — matches Passivbot risk.rs.
* ``confirm_trade_entry`` runs as the FINAL gate after correlation /
  vol-target / risk partial-fit so the strategy sees post-scale qty.
"""

from __future__ import annotations


import pytest

# ────────────────────────────────────────────────────────────────────
# bot_loop_start ordering: hook → populate
# ────────────────────────────────────────────────────────────────────


def test_bot_loop_start_state_visible_to_populate_in_backtest():
    """A strategy that stamps ``self.tag`` in bot_loop_start must have
    populate_entry_trend observe that same tag on the SAME tick.
    Pre-round-25 ordering (populate then bot_loop_start) would let the
    populate use the PREVIOUS tick's tag — exactly the lag ChatGPT
    flagged in round-25 P1 #1."""
    pd = pytest.importorskip("pandas")
    from combo_bot.backtest import BacktestConfig, Backtester
    from combo_bot.strategy import IStrategy
    from combo_bot.types import Candle

    observations: list[tuple[int, str]] = []

    class _SequencedStrat(IStrategy):
        tag = "init"
        tick = 0

        def populate_indicators(self, df, m):
            return df

        def populate_entry_trend(self, df, m):
            # Capture the tag VISIBLE to populate. If round-25 ordering
            # holds, populate sees "tick-N" set by bot_loop_start at
            # tick N. The pre-fix ordering would still show "tick-(N-1)"
            # because bot_loop_start ran AFTER populate.
            observations.append((type(self).tick, type(self).tag))
            return df

        def populate_exit_trend(self, df, m):
            return df

        def bot_loop_start(self, current_time, **kwargs):
            type(self).tick += 1
            type(self).tag = f"tick-{type(self).tick}"

    cfg = BacktestConfig(
        starting_balance=10_000.0,
        symbols=["BTC/USDT:USDT"],
        bar_interval_minutes=60.0,
    )
    bt = Backtester(cfg, strategy=_SequencedStrat())
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
    # Each populate call's observed tag must match its tick (set by
    # bot_loop_start AT THE START OF THAT TICK).
    assert observations == [
        (1, "tick-1"),
        (2, "tick-2"),
        (3, "tick-3"),
    ], (
        f"populate must observe the tag set by THIS tick's "
        f"bot_loop_start; got {observations}"
    )
    _ = pd


# ────────────────────────────────────────────────────────────────────
# n_positions=0 disables side
# ────────────────────────────────────────────────────────────────────


def test_n_positions_zero_blocks_new_entries_on_empty_account():
    """With no open positions and n_positions=0, both active sets are
    empty — no new entries can enter the active gate."""
    from combo_bot.grid_engine import ForagerWeights, compute_active_sides
    from combo_bot.types import (
        AccountState,
        Candle,
        SymbolState,
        TrendRegime,
        TrendSignal,
    )

    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    account = AccountState(balance=10_000.0)
    for s in symbols:
        account.symbols[s] = SymbolState(symbol=s)
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
    strategy = {
        s: (True, True, False, False) for s in symbols  # both sides try to enter
    }
    long_set, short_set = compute_active_sides(
        symbols=symbols,
        account=account,
        candles=candles,
        signals=signals,
        strategy_signals=strategy,
        n_positions=0,
        weights=ForagerWeights(),
    )
    assert (
        long_set == set()
    ), f"n_positions=0 must disable new long entries; got {long_set}"
    assert (
        short_set == set()
    ), f"n_positions=0 must disable new short entries; got {short_set}"


def test_n_positions_zero_still_manages_open_positions():
    """n_positions=0 disables NEW entries; open positions stay
    active so the engine can close / unstuck / trail them."""
    from combo_bot.grid_engine import ForagerWeights, compute_active_sides
    from combo_bot.types import (
        AccountState,
        Candle,
        Position,
        SymbolState,
        TrendRegime,
        TrendSignal,
    )

    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    account = AccountState(balance=10_000.0)
    for s in symbols:
        account.symbols[s] = SymbolState(symbol=s)
    account.symbols["BTC/USDT:USDT"].position_long = Position(
        size=0.01, entry_price=50_000.0
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
        n_positions=0,
        weights=ForagerWeights(),
    )
    # Open BTC long must STAY active so reduce-only / unstuck orders
    # can still flow.
    assert long_set == {
        "BTC/USDT:USDT"
    }, f"open positions must stay active under n_positions=0; got {long_set}"
    assert short_set == set()


# ────────────────────────────────────────────────────────────────────
# TWEL distance-based sorting
# ────────────────────────────────────────────────────────────────────


def test_twel_prefers_entries_closer_to_market_under_tight_budget():
    """Two same-size entries on different symbols competing for a
    single-entry budget. Round-25 sort is by distance-from-mark, so
    the entry priced closer to its symbol's mark must win — even if
    listed second in input order. Pre-round-25 sort was "by bucket
    base WE ascending" which had different priorities entirely."""
    from combo_bot.risk import RiskConfig, RiskManager
    from combo_bot.types import (
        AccountState,
        Order,
        OrderSource,
        Side,
        SymbolState,
    )

    account = AccountState(balance=10_000.0, equity=10_000.0, equity_peak=10_000.0)
    account.symbols["FAR/USDT:USDT"] = SymbolState(
        symbol="FAR/USDT:USDT", last_price=100.0
    )
    account.symbols["NEAR/USDT:USDT"] = SymbolState(
        symbol="NEAR/USDT:USDT", last_price=100.0
    )
    # FAR entry: priced 20% below mark → distance 0.20.
    # NEAR entry: priced 1% below mark → distance 0.01.
    far = Order(
        symbol="FAR/USDT:USDT",
        side=Side.LONG,
        price=80.0,
        qty=5.0,  # cost=400 → WE 0.04
        source=OrderSource.GRID,
    )
    near = Order(
        symbol="NEAR/USDT:USDT",
        side=Side.LONG,
        price=99.0,
        qty=4.04,  # cost≈400 → WE≈0.04
        source=OrderSource.GRID,
    )
    risk = RiskManager(
        RiskConfig(
            max_total_wallet_exposure=0.05,  # fits exactly ONE 0.04 entry
            max_single_exposure=1.0,
            yellow_threshold=0.99,
            orange_threshold=0.99,
            red_threshold=0.99,
        )
    )
    # FAR is first in input order; without distance-based sort it
    # would win by virtue of position. With it, NEAR wins.
    out = risk.filter_orders([far, near], account, timestamp=0)
    assert len(out) >= 1
    # NEAR (the closer-to-market entry) should be the one accepted at
    # full size; the FAR entry should be dropped or trimmed.
    near_kept = [o for o in out if o.symbol == "NEAR/USDT:USDT"]
    assert near_kept, (
        f"NEAR (distance 0.01) must beat FAR (distance 0.20) when "
        f"budget fits only one; got out={[o.symbol for o in out]}"
    )
    assert near_kept[0].qty == pytest.approx(
        4.04, abs=1e-9
    ), f"NEAR must enter at full size; got qty={near_kept[0].qty}"


# ────────────────────────────────────────────────────────────────────
# Final confirm sees post-scale qty
# ────────────────────────────────────────────────────────────────────


def test_final_confirm_sees_post_risk_trim_qty():
    """End-to-end: filter_entries + risk partial-fit trim + final
    confirm. The strategy's confirm_trade_entry must observe the
    TRIMMED qty, not the pre-trim qty."""
    from combo_bot.strategy import IStrategy, StrategyRunner, TradeContext
    from combo_bot.risk import RiskConfig, RiskManager
    from combo_bot.types import (
        AccountState,
        Candle,
        ExchangeParams,
        Order,
        OrderSource,
        Side,
        SymbolState,
        TrendRegime,
        TrendSignal,
    )

    seen_confirm: list[tuple[float, float]] = []

    class _S(IStrategy):
        def populate_indicators(self, df, m):
            return df

        def populate_entry_trend(self, df, m):
            return df

        def populate_exit_trend(self, df, m):
            return df

        def confirm_trade_entry(self, ctx, qty, price):
            seen_confirm.append((qty, price))
            return True

    runner = StrategyRunner(_S())
    account = AccountState(balance=10_000.0, equity=10_000.0, equity_peak=10_000.0)
    account.symbols["BTC/USDT:USDT"] = SymbolState(
        symbol="BTC/USDT:USDT", last_price=50_000.0
    )
    ctx = TradeContext(
        symbol="BTC/USDT:USDT",
        side=Side.LONG,
        position=account.symbols["BTC/USDT:USDT"].position_long,
        account=account,
        candle=Candle(0, 50_000, 50_000, 50_000, 50_000, 0),
        signal=TrendSignal(direction=0, strength=0, regime=TrendRegime.NEUTRAL),
        current_time_ms=0,
        exchange_params=ExchangeParams(
            qty_step=0.001, price_step=0.01, min_qty=0.001, min_cost=5.0
        ),
    )

    # Tight TWE cap of 0.05 → headroom 500. A 0.05 BTC order at
    # 50_000 costs 2500 (WE 0.25). Partial-fit trims to qty 0.01
    # (cost 500, WE 0.05).
    order = Order(
        symbol="BTC/USDT:USDT",
        side=Side.LONG,
        price=50_000.0,
        qty=0.05,
        source=OrderSource.GRID,
    )
    after_strategy = runner.filter_entries([order], ctx)
    assert after_strategy[0].qty == pytest.approx(
        0.05
    ), "filter_entries does NOT trim — only risk does"

    risk = RiskManager(
        RiskConfig(
            max_total_wallet_exposure=0.05,
            max_single_exposure=1.0,
            yellow_threshold=0.99,
            orange_threshold=0.99,
            red_threshold=0.99,
        )
    )
    after_risk = risk.filter_orders(
        after_strategy,
        account,
        timestamp=0,
        exchange_params={"BTC/USDT:USDT": ctx.exchange_params},
    )
    assert after_risk[0].qty == pytest.approx(0.01, abs=1e-9), (
        f"risk must trim qty to 0.01 (WE 0.05 headroom); " f"got {after_risk[0].qty}"
    )

    final = runner.final_confirm(after_risk, lambda _o: ctx)
    assert seen_confirm == [(pytest.approx(0.01, abs=1e-9), 50_000.0)], (
        f"confirm_trade_entry must see the TRIMMED qty=0.01, NOT the "
        f"pre-trim qty=0.05; got {seen_confirm}"
    )
    assert len(final) == 1


def test_final_confirm_veto_drops_order_after_risk_pass():
    """Strategy returning False from confirm_trade_entry drops the
    entry even when risk would have accepted it."""
    from combo_bot.strategy import IStrategy, StrategyRunner, TradeContext
    from combo_bot.types import (
        AccountState,
        Candle,
        ExchangeParams,
        Order,
        OrderSource,
        Side,
        SymbolState,
        TrendRegime,
        TrendSignal,
    )

    class _Veto(IStrategy):
        def populate_indicators(self, df, m):
            return df

        def populate_entry_trend(self, df, m):
            return df

        def populate_exit_trend(self, df, m):
            return df

        def confirm_trade_entry(self, ctx, qty, price):
            return False

    runner = StrategyRunner(_Veto())
    account = AccountState(balance=10_000.0)
    account.symbols["BTC/USDT:USDT"] = SymbolState(
        symbol="BTC/USDT:USDT", last_price=50_000.0
    )
    ctx = TradeContext(
        symbol="BTC/USDT:USDT",
        side=Side.LONG,
        position=account.symbols["BTC/USDT:USDT"].position_long,
        account=account,
        candle=Candle(0, 50_000, 50_000, 50_000, 50_000, 0),
        signal=TrendSignal(direction=0, strength=0, regime=TrendRegime.NEUTRAL),
        current_time_ms=0,
        exchange_params=ExchangeParams(),
    )
    order = Order(
        symbol="BTC/USDT:USDT",
        side=Side.LONG,
        price=50_000.0,
        qty=0.01,
        source=OrderSource.GRID,
    )
    out = runner.final_confirm([order], lambda _o: ctx)
    assert out == [], "vetoed entry must be dropped by final_confirm"
