from __future__ import annotations

import pytest

from combo_bot.backtest import BacktestConfig, Backtester
from combo_bot.types import (
    AccountState,
    Order,
    OrderSource,
    Position,
    Side,
    SymbolState,
)


class TestShortAccounting:
    def test_positive_size_short_unrealized_pnl_increases_equity_when_price_falls(self):
        account = AccountState(balance=10000, equity=10000, equity_peak=10000)
        account.symbols["BTC"] = SymbolState("BTC", last_price=49000)
        account.symbols["BTC"].position_short = Position(size=0.1, entry_price=50000)

        account.update_equity()

        assert account.equity == pytest.approx(10100)

    def test_positive_funding_rate_credits_short_position(self):
        account = AccountState(balance=10000, equity=10000, equity_peak=10000)
        account.symbols["BTC"] = SymbolState("BTC", last_price=50000)
        account.symbols["BTC"].position_short = Position(size=0.1, entry_price=50000)
        backtester = Backtester(BacktestConfig(funding_rate_default=0.001))

        funding_cost = backtester._apply_funding(account, None, 0, ["BTC"])

        assert funding_cost == pytest.approx(-5.0)
        assert account.balance == pytest.approx(10005.0)

    def test_funding_notional_uses_contract_multiplier(self):
        account = AccountState(balance=10000, equity=10000, equity_peak=10000)
        account.symbols["BTC"] = SymbolState("BTC", last_price=50000, c_mult=100.0)
        account.symbols["BTC"].position_short = Position(size=0.1, entry_price=50000)
        backtester = Backtester(BacktestConfig(funding_rate_default=0.001))

        funding_cost = backtester._apply_funding(account, None, 0, ["BTC"])

        assert funding_cost == pytest.approx(-500.0)
        assert account.balance == pytest.approx(10500.0)


class TestOrderSideSemantics:
    @pytest.mark.parametrize(
        ("position_side", "reduce_only", "expected_exchange_side"),
        [
            (Side.LONG, False, "buy"),
            (Side.LONG, True, "sell"),
            (Side.SHORT, False, "sell"),
            (Side.SHORT, True, "buy"),
        ],
    )
    def test_exchange_side_is_derived_from_position_side_and_reduce_only(
        self,
        position_side,
        reduce_only,
        expected_exchange_side,
    ):
        order = Order(
            symbol="BTC",
            side=position_side,
            price=50000,
            qty=0.1,
            source=OrderSource.GRID,
            reduce_only=reduce_only,
        )

        assert order.exchange_side == expected_exchange_side
