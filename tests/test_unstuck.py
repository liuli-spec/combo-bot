"""Stage 5 unstuck-mechanism tests.

Mirrors passivbot's ``calc_unstucking_action`` semantics adapted to our
source-isolated buckets:

  * fires only when a bucket's wallet exposure crosses
    ``unstuck_threshold * WEL``;
  * only when current price has reverted across the EMA band by
    ``unstuck_ema_dist`` (sell into the bounce, not into the bleed);
  * emits small reduce-only **limit** orders, not market;
  * throttled by a 24h rolling realized-loss budget;
  * each bucket evaluated independently.
"""

from __future__ import annotations

import pytest

from combo_bot.risk import RiskConfig, RiskManager
from combo_bot.types import (
    AccountState,
    EMAState,
    OrderSource,
    Position,
    Side,
    SymbolState,
)


# ---------------------------------------------------------------------------
# 24h rolling loss log on AccountState
# ---------------------------------------------------------------------------


class TestAccountLossLog:
    def test_negative_pnl_appended_to_log(self):
        acc = AccountState(balance=10_000.0)
        acc.add_realized_pnl(OrderSource.GRID, -50.0, 1_000)
        acc.add_realized_pnl(OrderSource.TREND, -25.0, 2_000)

        assert list(acc.grid_loss_log) == [(1_000, -50.0)]
        assert list(acc.trend_loss_log) == [(2_000, -25.0)]

    def test_positive_pnl_does_not_pollute_loss_log(self):
        acc = AccountState(balance=10_000.0)
        acc.add_realized_pnl(OrderSource.GRID, 100.0, 1_000)
        assert list(acc.grid_loss_log) == []

    def test_loss_24h_sums_recent_entries(self):
        acc = AccountState(balance=10_000.0)
        acc.add_realized_pnl(OrderSource.GRID, -20.0, 1_000)
        acc.add_realized_pnl(OrderSource.GRID, -30.0, 2_000)
        # 1 hour later → both still in the 24h window.
        loss = acc.loss_24h(OrderSource.GRID, 1_000 + 60 * 60 * 1_000)
        assert loss == pytest.approx(-50.0)

    def test_loss_24h_prunes_old_entries(self):
        acc = AccountState(balance=10_000.0)
        acc.add_realized_pnl(OrderSource.GRID, -100.0, 1_000)
        # 25 hours later → outside the window.
        loss = acc.loss_24h(OrderSource.GRID, 1_000 + 25 * 60 * 60 * 1_000)
        assert loss == 0.0
        # And the deque was pruned in place.
        assert len(acc.grid_loss_log) == 0


# ---------------------------------------------------------------------------
# RiskManager.compute_unstuck_orders
# ---------------------------------------------------------------------------


def _ema_state(lower=49_000.0, upper=49_500.0) -> EMAState:
    ema = EMAState(
        spans=[100.0, 200.0],
        values=[lower, upper],
        alphas=[0.02, 0.01],
        initialized=True,
    )
    return ema


def _risk(**overrides) -> RiskManager:
    defaults = dict(
        unstuck_threshold=0.90,
        unstuck_close_pct=0.05,   # 5% per fire — easier to inspect in tests
        unstuck_ema_dist=0.01,
        daily_loss_allowance_pct=0.05,
        trend_wallet_exposure_limit=0.15,
    )
    defaults.update(overrides)
    return RiskManager(RiskConfig(**defaults))


def _account_with_grid_long(*, balance, qty, entry, last_price, ema) -> AccountState:
    acc = AccountState(balance=balance)
    ss = SymbolState("BTC", last_price=last_price)
    ss.ema = ema
    ss.position_long = Position(qty, entry)
    acc.symbols["BTC"] = ss
    return acc


class TestUnstuckTrigger:
    def test_fires_when_stuck_and_bounce_above_ema_band(self):
        risk = _risk()
        # Grid bucket: 10 BTC @ 50k = 500_000 notional / 1_000 balance = WE 500.
        # WEL 600 → we/wel = 0.833... hmm, below 0.9. Let me size it past.
        # 5.5 BTC @ 50k = 275_000 / 1_000 = WE 275. WEL 300 → ratio 0.916.
        acc = _account_with_grid_long(
            balance=1_000.0, qty=5.5, entry=50_000.0,
            last_price=50_300.0,  # > upper(49_500) * 1.01 = 49_995
            ema=_ema_state(lower=49_000, upper=49_500),
        )

        orders = risk.compute_unstuck_orders(
            acc, grid_wallet_exposure_limit=300.0, now_ms=0,
        )

        assert len(orders) == 1
        o = orders[0]
        assert o.reduce_only is True
        assert o.is_market is False
        assert o.source == OrderSource.GRID
        assert o.side == Side.LONG
        # Price = ema.upper * (1 + dist)
        assert o.price == pytest.approx(49_500 * 1.01)
        # qty = position size * close_pct
        assert o.qty == pytest.approx(5.5 * 0.05)

    def test_does_not_fire_when_we_below_threshold(self):
        risk = _risk()
        # 4 BTC @ 50k = 200_000 / 1_000 = WE 200. WEL 300 → ratio 0.66.
        acc = _account_with_grid_long(
            balance=1_000.0, qty=4.0, entry=50_000.0, last_price=50_300.0,
            ema=_ema_state(),
        )
        orders = risk.compute_unstuck_orders(
            acc, grid_wallet_exposure_limit=300.0, now_ms=0,
        )
        assert orders == []

    def test_does_not_fire_when_price_below_ema_band_target(self):
        """passivbot only sells the bounce — if we're still bleeding,
        keep holding."""
        risk = _risk(unstuck_ema_dist=0.02)
        # Stuck (we/wel 0.916) but price still below ema.upper * 1.02.
        acc = _account_with_grid_long(
            balance=1_000.0, qty=5.5, entry=50_000.0,
            last_price=49_500.0,  # below 49_500 * 1.02 = 50_490
            ema=_ema_state(upper=49_500),
        )
        orders = risk.compute_unstuck_orders(
            acc, grid_wallet_exposure_limit=300.0, now_ms=0,
        )
        assert orders == []

    def test_short_bucket_uses_lower_band(self):
        risk = _risk()
        acc = AccountState(balance=1_000.0)
        ss = SymbolState("BTC", last_price=48_500.0)
        ss.ema = _ema_state(lower=49_000.0, upper=49_500.0)
        ss.position_short = Position(5.5, 50_000.0)
        acc.symbols["BTC"] = ss

        orders = risk.compute_unstuck_orders(
            acc, grid_wallet_exposure_limit=300.0, now_ms=0,
        )
        assert len(orders) == 1
        o = orders[0]
        assert o.side == Side.SHORT
        # price = lower * (1 - dist)
        assert o.price == pytest.approx(49_000 * 0.99)


# ---------------------------------------------------------------------------
# Per-bucket independence
# ---------------------------------------------------------------------------


class TestBucketIndependence:
    def test_trend_bucket_uses_its_own_wel(self):
        risk = _risk(trend_wallet_exposure_limit=0.20)
        acc = AccountState(balance=10_000.0)
        ss = SymbolState("BTC", last_price=50_300.0)
        ss.ema = _ema_state()
        # Trend WE: 0.05 * 50_000 = 2_500 / 10_000 = 0.25. Trend WEL 0.20.
        # ratio = 1.25 > 0.9 → stuck.
        ss.trend_long = Position(0.05, 50_000.0)
        acc.symbols["BTC"] = ss

        orders = risk.compute_unstuck_orders(
            acc, grid_wallet_exposure_limit=300.0, now_ms=0,
        )
        assert len(orders) == 1
        assert orders[0].source == OrderSource.TREND

    def test_grid_stuck_does_not_trigger_trend_unstuck(self):
        risk = _risk()
        acc = AccountState(balance=1_000.0)
        ss = SymbolState("BTC", last_price=50_300.0)
        ss.ema = _ema_state()
        # Grid stuck.
        ss.position_long = Position(5.5, 50_000.0)
        # Trend tiny — well under its WEL.
        ss.trend_long = Position(0.001, 50_000.0)
        acc.symbols["BTC"] = ss

        orders = risk.compute_unstuck_orders(
            acc, grid_wallet_exposure_limit=300.0, now_ms=0,
        )
        sources = {o.source for o in orders}
        assert OrderSource.GRID in sources
        assert OrderSource.TREND not in sources


# ---------------------------------------------------------------------------
# 24h loss-allowance throttle
# ---------------------------------------------------------------------------


class TestAllowanceThrottle:
    def test_allowance_exhausted_blocks_unstuck(self):
        risk = _risk(daily_loss_allowance_pct=0.01)  # $10 of $1000 balance
        acc = _account_with_grid_long(
            balance=1_000.0, qty=5.5, entry=50_000.0, last_price=50_300.0,
            ema=_ema_state(),
        )
        # Spent the entire grid loss budget already.
        acc.add_realized_pnl(OrderSource.GRID, -20.0, 0)

        orders = risk.compute_unstuck_orders(
            acc, grid_wallet_exposure_limit=300.0, now_ms=1_000,
        )
        assert orders == []

    def test_allowance_does_not_block_other_bucket(self):
        """A trend-bucket loss should not consume the grid bucket's allowance."""
        risk = _risk(daily_loss_allowance_pct=0.01)
        acc = AccountState(balance=1_000.0)
        ss = SymbolState("BTC", last_price=50_300.0)
        ss.ema = _ema_state()
        ss.position_long = Position(5.5, 50_000.0)  # grid stuck
        acc.symbols["BTC"] = ss
        # Trend losses don't deplete the grid budget.
        acc.add_realized_pnl(OrderSource.TREND, -50.0, 0)

        orders = risk.compute_unstuck_orders(
            acc, grid_wallet_exposure_limit=300.0, now_ms=1_000,
        )
        # Grid unstuck still fires.
        assert any(o.source == OrderSource.GRID for o in orders)

    def test_allowance_window_rolls_off(self):
        risk = _risk(daily_loss_allowance_pct=0.01)
        acc = _account_with_grid_long(
            balance=1_000.0, qty=5.5, entry=50_000.0, last_price=50_300.0,
            ema=_ema_state(),
        )
        # Loss recorded 25h ago — outside the window.
        acc.add_realized_pnl(OrderSource.GRID, -50.0, 0)

        orders = risk.compute_unstuck_orders(
            acc, grid_wallet_exposure_limit=300.0,
            now_ms=25 * 60 * 60 * 1000,
        )
        assert len(orders) == 1


# ---------------------------------------------------------------------------
# Edge guards
# ---------------------------------------------------------------------------


class TestEdgeGuards:
    def test_no_position_no_orders(self):
        risk = _risk()
        acc = AccountState(balance=1_000.0)
        acc.symbols["BTC"] = SymbolState("BTC", last_price=50_000.0)
        orders = risk.compute_unstuck_orders(
            acc, grid_wallet_exposure_limit=300.0, now_ms=0,
        )
        assert orders == []

    def test_uninitialized_ema_skipped(self):
        risk = _risk()
        acc = _account_with_grid_long(
            balance=1_000.0, qty=5.5, entry=50_000.0, last_price=50_300.0,
            ema=EMAState(),  # uninitialized
        )
        orders = risk.compute_unstuck_orders(
            acc, grid_wallet_exposure_limit=300.0, now_ms=0,
        )
        assert orders == []

    def test_zero_balance_returns_empty(self):
        risk = _risk()
        acc = _account_with_grid_long(
            balance=0.0, qty=5.5, entry=50_000.0, last_price=50_300.0,
            ema=_ema_state(),
        )
        orders = risk.compute_unstuck_orders(
            acc, grid_wallet_exposure_limit=300.0, now_ms=0,
        )
        assert orders == []

    def test_negative_threshold_disables_mechanism(self):
        risk = _risk(unstuck_threshold=-1.0)
        acc = _account_with_grid_long(
            balance=1_000.0, qty=5.5, entry=50_000.0, last_price=50_300.0,
            ema=_ema_state(),
        )
        orders = risk.compute_unstuck_orders(
            acc, grid_wallet_exposure_limit=300.0, now_ms=0,
        )
        assert orders == []
