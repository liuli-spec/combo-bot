from __future__ import annotations
import pytest
from combo_bot.strategy import (
    IStrategy, StrategyRunner, DefaultStrategy, ExampleTrendStrategy, TradeContext,
)
from combo_bot.types import (
    AccountState, Candle, ExchangeParams, Order, OrderSource, Position,
    Side, SymbolState, TradingMode, TrendRegime, TrendSignal,
)


@pytest.fixture
def default_account():
    acc = AccountState(balance=10000, equity=10000, equity_peak=10000)
    acc.symbols["BTC"] = SymbolState("BTC", last_price=50000)
    return acc


@pytest.fixture
def default_candle():
    return Candle(timestamp=1700000000000, open=50000, high=50100, low=49900,
                  close=50000, volume=100)


@pytest.fixture
def default_ctx(default_account, default_candle):
    return TradeContext(
        symbol="BTC", side=Side.LONG, position=Position(),
        account=default_account, candle=default_candle,
        signal=TrendSignal(direction=0.5, strength=0.5, regime=TrendRegime.BULL),
        current_time_ms=1700000000000,
        exchange_params=ExchangeParams(),
    )


class TestDefaultStrategy:
    def test_default_strategy_instantiates(self):
        s = DefaultStrategy()
        assert isinstance(s, IStrategy)

    def test_default_callbacks_no_op(self, default_ctx):
        s = DefaultStrategy()
        # confirm_trade_entry default should return True
        assert s.confirm_trade_entry(default_ctx, 0.01, 50000) is True
        # custom_stake_amount default returns proposed stake
        assert s.custom_stake_amount(default_ctx, 100, 5, 1000) == 100


class TestExampleStrategy:
    def test_example_instantiates(self):
        s = ExampleTrendStrategy()
        assert isinstance(s, IStrategy)

    def test_example_vetoes_weak_signal(self, default_account, default_candle):
        s = ExampleTrendStrategy()
        ctx = TradeContext(
            symbol="BTC", side=Side.LONG, position=Position(),
            account=default_account, candle=default_candle,
            signal=TrendSignal(direction=0.2, strength=0.2, regime=TrendRegime.NEUTRAL),
            current_time_ms=1700000000000,
            exchange_params=ExchangeParams(),
        )
        # Strategy should veto entries when signal strength is below threshold
        assert s.confirm_trade_entry(ctx, 0.01, 50000) is False


class TestStrategyRunner:
    def test_filter_entries_passes_through_for_default(self, default_ctx):
        runner = StrategyRunner(DefaultStrategy())
        orders = [
            Order("BTC", Side.LONG, 49000, 0.01, OrderSource.GRID),
            Order("BTC", Side.LONG, 48500, 0.015, OrderSource.GRID),
        ]
        filtered = runner.filter_entries(orders, default_ctx)
        assert len(filtered) == 2

    def test_filter_entries_vetoes_on_callback_false(self, default_ctx):
        class VetoStrategy(DefaultStrategy):
            def confirm_trade_entry(self, ctx, qty, price):
                return False

        runner = StrategyRunner(VetoStrategy())
        orders = [Order("BTC", Side.LONG, 49000, 0.01, OrderSource.GRID)]
        filtered = runner.filter_entries(orders, default_ctx)
        assert len(filtered) == 0

    def test_filter_exits_passes_reduce_only(self, default_ctx):
        runner = StrategyRunner(DefaultStrategy())
        orders = [
            Order("BTC", Side.LONG, 51000, 0.01, OrderSource.GRID, reduce_only=True),
        ]
        filtered = runner.filter_exits(orders, default_ctx)
        assert len(filtered) == 1


class TestCustomExit:
    def test_custom_exit_callback_triggers_close(self, default_account):
        class StopStrategy(DefaultStrategy):
            def custom_exit(self, ctx, current_profit_pct):
                if current_profit_pct < -0.02:
                    return "stop_loss"
                return None

        pos = Position(size=0.01, entry_price=51000)
        default_account.symbols["BTC"].position_long = pos
        bad_candle = Candle(
            timestamp=1700000000000, open=49500, high=49600,
            low=49400, close=49000, volume=100,
        )
        ctx = TradeContext(
            symbol="BTC", side=Side.LONG, position=pos,
            account=default_account, candle=bad_candle,
            signal=None, current_time_ms=1700000000000,
            exchange_params=ExchangeParams(),
        )

        runner = StrategyRunner(StopStrategy())
        order = runner.check_custom_exit(ctx)
        assert order is not None
        assert order.reduce_only
