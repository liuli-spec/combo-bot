"""Freqtrade-inspired strategy interface.

This module provides a pluggable strategy layer that lets users customize
the bot's behavior via callback methods, independent of the underlying
Rust engine. Users subclass :class:`IStrategy` and override only the
hooks they need; :class:`StrategyRunner` applies those callbacks to the
order stream produced by the engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any

from combo_bot.types import (
    AccountState,
    Candle,
    ExchangeParams,
    Order,
    OrderSource,
    Position,
    Side,
    SymbolState,
    TradingMode,
    TrendRegime,
    TrendSignal,
)

if TYPE_CHECKING:
    # Imported lazily inside methods to avoid hard pandas dependency at import time.
    from pandas import DataFrame


# ---------------------------------------------------------------------------
# Context passed to every strategy callback
# ---------------------------------------------------------------------------


@dataclass
class TradeContext:
    """Snapshot of state passed to strategy callbacks.

    All fields are read-only references to the bot's current view of the
    world. Strategies must not mutate these values; return new values from
    callbacks instead.
    """

    symbol: str
    side: Side
    position: Position
    account: AccountState
    candle: Candle
    signal: TrendSignal | None
    current_time_ms: int
    exchange_params: ExchangeParams

    @property
    def current_price(self) -> float:
        return self.candle.close

    @property
    def is_in_position(self) -> bool:
        return self.position.is_open


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------


class IStrategy(ABC):
    """Abstract base for user-defined strategies.

    The three ``populate_*`` methods are required. All other hooks have
    sensible no-op defaults so subclasses can override only what they
    need.
    """

    # --- metadata -----------------------------------------------------------

    timeframe: str = "1h"
    stoploss: float = -0.10
    trailing_stop: bool = False
    process_only_new_candles: bool = True
    startup_candle_count: int = 30

    # --- required overrides --------------------------------------------------

    @abstractmethod
    def populate_indicators(
        self, dataframe: "DataFrame", metadata: dict[str, Any]
    ) -> "DataFrame":
        """Compute indicator columns on the dataframe and return it."""

    @abstractmethod
    def populate_entry_trend(
        self, dataframe: "DataFrame", metadata: dict[str, Any]
    ) -> "DataFrame":
        """Mark entry signals via ``enter_long``, ``enter_short``, ``enter_tag``."""

    @abstractmethod
    def populate_exit_trend(
        self, dataframe: "DataFrame", metadata: dict[str, Any]
    ) -> "DataFrame":
        """Mark exit signals via ``exit_long``, ``exit_short``, ``exit_tag``."""

    # --- lifecycle hooks -----------------------------------------------------

    def bot_start(self, **kwargs: Any) -> None:
        """Called once when the bot starts. Default: no-op."""

    def bot_loop_start(self, current_time: datetime, **kwargs: Any) -> None:
        """Called at the start of each tick. Default: no-op."""

    def informative_pairs(self) -> list[tuple[str, str]]:
        """Additional (pair, timeframe) tuples to load. Default: none."""
        return []

    # --- entry / exit veto ---------------------------------------------------

    def confirm_trade_entry(
        self,
        ctx: TradeContext,
        proposed_qty: float,
        proposed_price: float,
    ) -> bool:
        """Return ``False`` to veto a proposed entry. Default: allow."""
        return True

    def confirm_trade_exit(
        self,
        ctx: TradeContext,
        proposed_qty: float,
        proposed_price: float,
        exit_reason: str,
    ) -> bool:
        """Return ``False`` to veto a proposed exit. Default: allow."""
        return True

    # --- price / size adjustment --------------------------------------------

    def custom_entry_price(
        self, ctx: TradeContext, proposed_price: float
    ) -> float:
        """Adjust the entry price. Default: passthrough."""
        return proposed_price

    def custom_exit_price(
        self,
        ctx: TradeContext,
        proposed_price: float,
        exit_reason: str,
    ) -> float:
        """Adjust the exit price. Default: passthrough."""
        return proposed_price

    def custom_stake_amount(
        self,
        ctx: TradeContext,
        proposed_stake: float,
        min_stake: float,
        max_stake: float,
    ) -> float:
        """Override position size in quote currency. Default: passthrough."""
        return proposed_stake

    # --- adaptive risk ------------------------------------------------------

    def custom_stoploss(
        self, ctx: TradeContext, current_profit_pct: float
    ) -> float | None:
        """Return an absolute stoploss price, or ``None`` to skip. Default: ``None``."""
        return None

    def custom_exit(
        self, ctx: TradeContext, current_profit_pct: float
    ) -> str | None:
        """Return an exit reason to trigger a close, or ``None`` to skip."""
        return None

    def adjust_trade_position(
        self, ctx: TradeContext, current_profit_pct: float
    ) -> float | None:
        """Return qty to add (positive) or trim (negative), or ``None``."""
        return None

    # --- pending order adjustment -------------------------------------------

    def adjust_entry_price(
        self, ctx: TradeContext, current_order_price: float
    ) -> float | None:
        """Modify a pending entry price. Default: keep as-is."""
        return None

    def adjust_exit_price(
        self,
        ctx: TradeContext,
        current_order_price: float,
        exit_reason: str,
    ) -> float | None:
        """Modify a pending exit price. Default: keep as-is."""
        return None

    # --- leverage / timeouts ------------------------------------------------

    def leverage(
        self,
        ctx: TradeContext,
        proposed_leverage: float,
        max_leverage: float,
    ) -> float:
        """Set leverage for a new position. Default: passthrough, capped."""
        return min(proposed_leverage, max_leverage)

    def check_entry_timeout(
        self, ctx: TradeContext, order_age_seconds: int
    ) -> bool:
        """Return ``True`` to cancel a stale pending entry. Default: never."""
        return False

    def check_exit_timeout(
        self, ctx: TradeContext, order_age_seconds: int
    ) -> bool:
        """Return ``True`` to cancel a stale pending exit. Default: never."""
        return False


# ---------------------------------------------------------------------------
# Strategy runner — bridges the callbacks to the engine's order stream
# ---------------------------------------------------------------------------


class StrategyRunner:
    """Applies a strategy's callbacks to the order list produced by the engine."""

    def __init__(self, strategy: IStrategy) -> None:
        self.strategy = strategy

    # --- entry filtering ----------------------------------------------------

    def filter_entries(
        self, orders: list[Order], context: TradeContext
    ) -> list[Order]:
        """Apply entry-side strategy hooks to a list of orders.

        Order processing:
          1. ``confirm_trade_entry`` — drop vetoed orders.
          2. ``custom_stake_amount`` — adjust qty for new entries.
          3. ``custom_entry_price`` / ``adjust_entry_price`` — adjust price.
        """
        result: list[Order] = []
        for order in orders:
            if order.reduce_only:
                # Reduce-only orders are exits, not entries — pass through.
                result.append(order)
                continue

            ctx = self._ctx_for_order(context, order)
            if not self.strategy.confirm_trade_entry(ctx, order.qty, order.price):
                continue

            new_price = self.strategy.custom_entry_price(ctx, order.price)
            adjusted = self.strategy.adjust_entry_price(ctx, new_price)
            final_price = adjusted if adjusted is not None else new_price

            proposed_stake = order.qty * final_price
            min_stake = context.exchange_params.min_cost
            max_stake = max(context.account.balance, proposed_stake)
            new_stake = self.strategy.custom_stake_amount(
                ctx, proposed_stake, min_stake, max_stake
            )
            new_qty = new_stake / final_price if final_price > 0 else order.qty

            result.append(replace(order, price=final_price, qty=new_qty))

        return result

    # --- exit filtering -----------------------------------------------------

    def filter_exits(
        self, orders: list[Order], context: TradeContext
    ) -> list[Order]:
        """Apply exit-side strategy hooks to a list of orders."""
        result: list[Order] = []
        for order in orders:
            if not order.reduce_only:
                # Non-reduce-only orders are entries — pass through.
                result.append(order)
                continue

            ctx = self._ctx_for_order(context, order)
            exit_reason = self._exit_reason_for(order)

            if not self.strategy.confirm_trade_exit(
                ctx, order.qty, order.price, exit_reason
            ):
                continue

            new_price = self.strategy.custom_exit_price(
                ctx, order.price, exit_reason
            )
            adjusted = self.strategy.adjust_exit_price(
                ctx, new_price, exit_reason
            )
            final_price = adjusted if adjusted is not None else new_price

            result.append(replace(order, price=final_price))

        return result

    # --- adaptive exit checks -----------------------------------------------

    def check_custom_exit(self, context: TradeContext) -> Order | None:
        """Run ``custom_exit`` then ``custom_stoploss``. Returns a market close order if triggered."""
        if not context.position.is_open:
            return None

        current_profit_pct = self._profit_pct(context)

        reason = self.strategy.custom_exit(context, current_profit_pct)
        if reason is not None:
            return self._make_close_order(context, context.current_price)

        sl_price = self.strategy.custom_stoploss(context, current_profit_pct)
        if sl_price is not None and self._stoploss_hit(context, sl_price):
            return self._make_close_order(context, sl_price)

        return None

    def check_position_adjustment(
        self, context: TradeContext
    ) -> Order | None:
        """Call ``adjust_trade_position`` and turn its result into an order."""
        if not context.position.is_open:
            return None

        current_profit_pct = self._profit_pct(context)
        delta_qty = self.strategy.adjust_trade_position(
            context, current_profit_pct
        )
        if delta_qty is None or abs(delta_qty) < 1e-12:
            return None

        # Positive delta = add to position; negative = trim.
        reduce_only = (
            (context.side == Side.LONG and delta_qty < 0)
            or (context.side == Side.SHORT and delta_qty > 0)
        )
        return Order(
            symbol=context.symbol,
            side=context.side,
            price=context.current_price,
            qty=abs(delta_qty),
            source=OrderSource.TREND,
            reduce_only=reduce_only,
        )

    # --- internal helpers ---------------------------------------------------

    def _ctx_for_order(
        self, base: TradeContext, order: Order
    ) -> TradeContext:
        """Build a per-order context (side may differ from the base context)."""
        if order.side == base.side and order.symbol == base.symbol:
            return base
        symbol_state = base.account.symbols.get(order.symbol)
        position = self._position_for(symbol_state, order.side)
        return TradeContext(
            symbol=order.symbol,
            side=order.side,
            position=position,
            account=base.account,
            candle=base.candle,
            signal=base.signal,
            current_time_ms=base.current_time_ms,
            exchange_params=base.exchange_params,
        )

    @staticmethod
    def _position_for(
        symbol_state: SymbolState | None, side: Side
    ) -> Position:
        if symbol_state is None:
            return Position()
        return (
            symbol_state.position_long
            if side == Side.LONG
            else symbol_state.position_short
        )

    @staticmethod
    def _profit_pct(ctx: TradeContext) -> float:
        pos = ctx.position
        if not pos.is_open or pos.entry_price <= 0:
            return 0.0
        if ctx.side == Side.LONG:
            return (ctx.current_price - pos.entry_price) / pos.entry_price
        return (pos.entry_price - ctx.current_price) / pos.entry_price

    @staticmethod
    def _stoploss_hit(ctx: TradeContext, sl_price: float) -> bool:
        if ctx.side == Side.LONG:
            return ctx.current_price <= sl_price
        return ctx.current_price >= sl_price

    @staticmethod
    def _exit_reason_for(order: Order) -> str:
        if order.source == OrderSource.RISK:
            return "risk_exit"
        if order.source == OrderSource.TREND:
            return "trend_exit"
        return "grid_exit"

    @staticmethod
    def _make_close_order(ctx: TradeContext, price: float) -> Order:
        return Order(
            symbol=ctx.symbol,
            side=ctx.side,
            price=price,
            qty=abs(ctx.position.size),
            source=OrderSource.RISK,
            reduce_only=True,
        )


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------


class DefaultStrategy(IStrategy):
    """No-op strategy used when the user supplies no override.

    All callbacks fall back to the :class:`IStrategy` defaults; the
    required populate methods return the dataframe unchanged so no
    trend-driven entries or exits are emitted.
    """

    def populate_indicators(
        self, dataframe: "DataFrame", metadata: dict[str, Any]
    ) -> "DataFrame":
        return dataframe

    def populate_entry_trend(
        self, dataframe: "DataFrame", metadata: dict[str, Any]
    ) -> "DataFrame":
        return dataframe

    def populate_exit_trend(
        self, dataframe: "DataFrame", metadata: dict[str, Any]
    ) -> "DataFrame":
        return dataframe


class ExampleTrendStrategy(IStrategy):
    """Minimal but realistic example of the strategy pattern.

    - Indicators: RSI(14), EMA(12) fast, EMA(26) slow.
    - Long entries when RSI < 30 and the fast EMA is above the slow EMA.
    - Long exits when RSI > 70.
    - Trailing 5% stoploss via :meth:`custom_stoploss`.
    - Vetoes entries when trend signal strength is weak.
    """

    timeframe = "1h"
    stoploss = -0.10
    startup_candle_count = 30

    # --- thresholds ---------------------------------------------------------

    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    ema_fast_period: int = 12
    ema_slow_period: int = 26
    trailing_stop_pct: float = 0.05
    min_signal_strength: float = 0.5

    # --- required overrides --------------------------------------------------

    def populate_indicators(
        self, dataframe: "DataFrame", metadata: dict[str, Any]
    ) -> "DataFrame":
        import pandas as pd  # local import: keep pandas optional at module load

        close = dataframe["close"]

        # RSI via Wilder's smoothing.
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, pd.NA)
        dataframe["rsi"] = 100.0 - (100.0 / (1.0 + rs))

        dataframe["ema_fast"] = close.ewm(
            span=self.ema_fast_period, adjust=False
        ).mean()
        dataframe["ema_slow"] = close.ewm(
            span=self.ema_slow_period, adjust=False
        ).mean()

        return dataframe

    def populate_entry_trend(
        self, dataframe: "DataFrame", metadata: dict[str, Any]
    ) -> "DataFrame":
        long_cond = (dataframe["rsi"] < self.rsi_oversold) & (
            dataframe["ema_fast"] > dataframe["ema_slow"]
        )
        dataframe.loc[long_cond, "enter_long"] = 1
        dataframe.loc[long_cond, "enter_tag"] = "rsi_oversold_ema_up"
        return dataframe

    def populate_exit_trend(
        self, dataframe: "DataFrame", metadata: dict[str, Any]
    ) -> "DataFrame":
        exit_cond = dataframe["rsi"] > self.rsi_overbought
        dataframe.loc[exit_cond, "exit_long"] = 1
        dataframe.loc[exit_cond, "exit_tag"] = "rsi_overbought"
        return dataframe

    # --- optional overrides --------------------------------------------------

    def custom_stoploss(
        self, ctx: TradeContext, current_profit_pct: float
    ) -> float | None:
        """5% trailing stop, only active once the trade is in profit."""
        if current_profit_pct <= 0:
            return None

        pos = ctx.position
        if not pos.is_open or pos.entry_price <= 0:
            return None

        if ctx.side == Side.LONG:
            return ctx.current_price * (1.0 - self.trailing_stop_pct)
        return ctx.current_price * (1.0 + self.trailing_stop_pct)

    def confirm_trade_entry(
        self,
        ctx: TradeContext,
        proposed_qty: float,
        proposed_price: float,
    ) -> bool:
        if ctx.signal is None:
            return True
        if ctx.signal.strength < self.min_signal_strength:
            return False
        # In strong counter-trend regimes, avoid adding to the dominant side.
        if (
            ctx.side == Side.LONG
            and ctx.signal.regime == TrendRegime.STRONG_BEAR
        ):
            return False
        if (
            ctx.side == Side.SHORT
            and ctx.signal.regime == TrendRegime.STRONG_BULL
        ):
            return False
        return True


__all__ = [
    "TradeContext",
    "IStrategy",
    "StrategyRunner",
    "DefaultStrategy",
    "ExampleTrendStrategy",
]
