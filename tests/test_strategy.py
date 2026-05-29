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

    def test_confirm_trade_entry_sees_final_price_and_qty(self, default_ctx):
        seen: list[tuple[float, float]] = []

        class FinalOrderStrategy(DefaultStrategy):
            def custom_entry_price(self, ctx, proposed_price):
                return 100.0

            def adjust_entry_price(self, ctx, current_order_price):
                return 125.0

            def custom_stake_amount(self, ctx, proposed_stake, min_stake, max_stake):
                return 250.0

            def confirm_trade_entry(self, ctx, qty, price):
                seen.append((qty, price))
                return price == 125.0 and qty == 2.0

        runner = StrategyRunner(FinalOrderStrategy())
        orders = [Order("BTC", Side.LONG, 49000, 0.01, OrderSource.GRID)]
        filtered = runner.filter_entries(orders, default_ctx)

        assert len(filtered) == 1
        assert seen == [(2.0, 125.0)]
        assert filtered[0].price == pytest.approx(125.0)
        assert filtered[0].qty == pytest.approx(2.0)


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

    def test_trend_context_custom_exit_routes_to_trend_bucket(self, default_account):
        class StopStrategy(DefaultStrategy):
            def custom_exit(self, ctx, current_profit_pct):
                return "trend_stop"

        pos = Position(size=0.02, entry_price=51000)
        ss = default_account.symbols["BTC"]
        ss.trend_long = pos
        candle = Candle(
            timestamp=1700000000000, open=49500, high=49600,
            low=49400, close=49000, volume=100,
        )
        ctx = TradeContext(
            symbol="BTC", side=Side.LONG, position=pos,
            account=default_account, candle=candle,
            signal=None, current_time_ms=1700000000000,
            exchange_params=ExchangeParams(),
            source=OrderSource.TREND,
        )

        runner = StrategyRunner(StopStrategy())
        order = runner.check_custom_exit(ctx)

        assert order is not None
        assert order.source == OrderSource.TREND
        assert ss.bucket(order.source, order.side) is ss.trend_long


class TestPositionAdjustment:
    def test_trend_context_adjustment_routes_to_trend_bucket(self, default_account):
        class TrimStrategy(DefaultStrategy):
            def adjust_trade_position(self, ctx, current_profit_pct):
                return -0.01

        pos = Position(size=0.04, entry_price=51000)
        ss = default_account.symbols["BTC"]
        ss.trend_long = pos
        candle = Candle(
            timestamp=1700000000000, open=49500, high=49600,
            low=49400, close=49000, volume=100,
        )
        ctx = TradeContext(
            symbol="BTC", side=Side.LONG, position=pos,
            account=default_account, candle=candle,
            signal=None, current_time_ms=1700000000000,
            exchange_params=ExchangeParams(),
            source=OrderSource.TREND,
        )

        runner = StrategyRunner(TrimStrategy())
        order = runner.check_position_adjustment(ctx)

        assert order is not None
        assert order.reduce_only
        assert order.source == OrderSource.TREND
        assert ss.bucket(order.source, order.side) is ss.trend_long
