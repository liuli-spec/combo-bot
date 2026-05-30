from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# A day in milliseconds — used for the rolling-loss allowance window
# that throttles the unstuck mechanism. Defined here (not in risk.py) so
# AccountState can prune entries without importing the risk layer.
_DAY_MS = 24 * 60 * 60 * 1000


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


class TradingMode(str, Enum):
    NORMAL = "normal"
    # Favorable regime: grid stacks faster (larger DDF) and closes sooner
    # (compressed markup). Used for the side aligned with a strong trend.
    AGGRESSIVE = "aggressive"
    TP_ONLY = "tp_only"
    GRACEFUL_STOP = "graceful_stop"
    PANIC = "panic"


class OrderSource(str, Enum):
    GRID = "grid"
    TREND = "trend"
    RISK = "risk"


class TrendRegime(str, Enum):
    STRONG_BULL = "strong_bull"
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"


@dataclass
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Position:
    """Position bucket for one side of one symbol.

    ``size`` is always stored as a *non-negative* scalar (|contracts|) for
    both long and short sides.  The direction is determined by the enclosing
    :class:`SymbolState` bucket (``position_long`` vs ``position_short``).
    Callers MUST pass the correct :class:`Side` to :meth:`unrealized_pnl`
    and :meth:`update_best_price`; a missing ``side`` argument raises because
    the unsigned size alone cannot distinguish long from short P&L.
    """

    size: float = 0.0
    entry_price: float = 0.0
    # Most-favorable price seen since the position opened.
    # Long: running max of mark price. Short: running min of mark price.
    # Zero means uninitialized (treat as entry_price).
    best_price: float = 0.0

    @property
    def is_open(self) -> bool:
        return abs(self.size) > 1e-12

    def unrealized_pnl(self, price: float, side: Side | None = None) -> float:
        if not self.is_open:
            return 0.0
        if side == Side.LONG:
            return abs(self.size) * (price - self.entry_price)
        if side == Side.SHORT:
            return abs(self.size) * (self.entry_price - price)
        # ``size`` is stored as a non-negative scalar regardless of side
        # (see the class docstring), so a no-side fallback that does
        # ``self.size * (price - entry)`` silently returns the WRONG SIGN
        # for short positions. Force callers to specify side instead of
        # quietly mis-reporting P&L.
        raise ValueError(
            "Position.unrealized_pnl requires an explicit side — size is "
            "stored unsigned so the fallback would mis-report short P&L"
        )

    def update_best_price(self, mark_price: float, side: Side) -> None:
        if not self.is_open or mark_price <= 0:
            return
        if side == Side.LONG:
            self.best_price = max(self.best_price or self.entry_price, mark_price)
        else:
            current = self.best_price if self.best_price > 0 else self.entry_price
            self.best_price = min(current, mark_price)


@dataclass
class Order:
    symbol: str
    side: Side
    price: float
    qty: float
    source: OrderSource
    reduce_only: bool = False
    # When True, the fill simulator treats this as a guaranteed-fill at the
    # current candle close (taker fee). Live executor sends it as a market
    # order. Use for forced exits (panic close, trend SL/TP hit).
    is_market: bool = False
    # Optional client-side identifier the live executor uses to match
    # exchange-returned orders without falling back to fuzzy (price,
    # qty) matching. Empty string means "no cOID set yet" — the live
    # trader generates a UUID-derived value before sending.
    client_order_id: str = ""

    @property
    def exchange_side(self) -> str:
        if (self.side == Side.LONG and not self.reduce_only) or (
            self.side == Side.SHORT and self.reduce_only
        ):
            return "buy"
        return "sell"


@dataclass
class Fill:
    timestamp: int
    symbol: str
    side: Side
    price: float
    qty: float
    fee: float
    realized_pnl: float
    source: OrderSource
    reduce_only: bool = False
    # Identity of the originating order. ``exchange_order_id`` and
    # ``client_order_id`` are best-effort and may be empty when the
    # exchange (or test stub) doesn't echo them. Without these the
    # intent journal can only match fills to journal entries by
    # (source, symbol, side) — too coarse when several grid orders
    # share those keys, leading to wrong cIDs being marked terminal
    # and the wrong rows being compacted out. ``trade_id`` is the
    # exchange's per-trade id — useful for downstream dedup beyond
    # what FillEventManager already does internally.
    exchange_order_id: str = ""
    client_order_id: str = ""
    trade_id: str = ""


@dataclass
class TrendSignal:
    direction: float
    strength: float
    regime: TrendRegime


@dataclass(frozen=True)
class RegimeView:
    """Synthesized regime decision consumed by Backtester / LiveTrader.

    Produced by :class:`combo_bot.regime.RegimeArbiter` from a TrendSignal,
    an optional strategy signal (latest-row enter_long/enter_short), and the
    current funding rate. Drives per-side grid mode, trend-overlay activation,
    and close-markup compression. Frozen so it cannot be mutated mid-tick.
    """

    primary: TrendRegime
    conviction: float
    long_mode: TradingMode
    short_mode: TradingMode
    # When True, the merger may emit grid entries on the matching side.
    # When False, only reduce-only orders pass through (TP_ONLY semantics
    # are already encoded via long_mode/short_mode == TP_ONLY; this flag is
    # an extra safety brake the strategy or risk layer can wire to).
    allow_grid_long: bool = True
    allow_grid_short: bool = True
    # Which side, if any, the trend overlay should pyramid into.
    trend_overlay: Side | None = None
    # 0..N — multiplier on the base trend entry sizing.
    trend_qty_scale: float = 0.0
    # 1.0 = engine defaults. <1.0 = tighter close markup (close sooner).
    close_aggressiveness: float = 1.0
    veto_reasons: tuple[str, ...] = ()


@dataclass
class EMAState:
    spans: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    alphas: list[float] = field(default_factory=list)
    initialized: bool = False

    def init(self, spans: list[float], price: float):
        self.spans = list(spans)
        self.alphas = [2.0 / (s + 1.0) for s in spans]
        self.values = [price] * len(spans)
        self.initialized = True

    def update(self, price: float):
        for i in range(len(self.values)):
            self.values[i] = (
                self.alphas[i] * price + (1.0 - self.alphas[i]) * self.values[i]
            )

    @property
    def lower(self) -> float:
        return min(self.values) if self.values else 0.0

    @property
    def upper(self) -> float:
        return max(self.values) if self.values else 0.0


@dataclass
class TrailingState:
    """Per-side trailing price tracker for passivbot-style re-entries.

    Tracks two values across the lifetime of an open position:

    * ``extreme``  — for LONG: the running minimum since position open;
                     for SHORT: the running maximum.
    * ``recovery`` — for LONG: the running maximum *after* the most-recent
                     ``extreme`` was set; for SHORT: the running minimum
                     after the most-recent extreme.

    The two-stage trigger (passivbot ``calc_trailing_entry_long``):
      1. extreme moved at least ``threshold_pct`` against the entry price;
      2. recovery rebounded at least ``retracement_pct`` from extreme.

    Both must fire before a re-entry order is placed — we want to DCA
    *after* a bottom is in, not while still bleeding.
    """

    extreme: float = 0.0
    recovery: float = 0.0
    initialized: bool = False

    def reset(self, price: float) -> None:
        self.extreme = price
        self.recovery = price
        self.initialized = True

    def update_long(self, high: float, low: float) -> None:
        if not self.initialized:
            return
        if low < self.extreme:
            self.extreme = low
            # New low → recovery anchor resets to the same point so the
            # rebound is measured from this new bottom.
            self.recovery = low
        elif high > self.recovery:
            self.recovery = high

    def update_short(self, high: float, low: float) -> None:
        if not self.initialized:
            return
        if high > self.extreme:
            self.extreme = high
            self.recovery = high
        elif low < self.recovery:
            self.recovery = low


@dataclass
class VolatilityState:
    ema_span_hours: float = 1000.0
    value: float = 0.0
    alpha: float = 0.0
    initialized: bool = False

    def init(
        self,
        span_hours: float,
        initial_range: float,
        bar_interval_minutes: float = 1.0,
    ):
        """Seed the EMA. ``bar_interval_minutes`` lets callers tell us the
        cadence at which ``update`` will be called — alpha = 2/(N+1)
        where N is the effective number of bars in ``span_hours``.

        Default of 1.0 preserves legacy behaviour (one-minute bars). For
        hourly bars, pass 60.0 so the EMA actually spans ``span_hours``
        and not ``span_hours * 60`` hours.
        """
        self.ema_span_hours = span_hours
        bars_per_span = max(1.0, span_hours * 60.0 / max(bar_interval_minutes, 1e-9))
        self.alpha = 2.0 / (bars_per_span + 1.0)
        self.value = initial_range
        self.initialized = True

    def update(self, log_range: float):
        self.value = self.alpha * log_range + (1.0 - self.alpha) * self.value


@dataclass
class SymbolState:
    """Per-symbol state with source-isolated position buckets.

    ``position_long`` / ``position_short`` hold the GRID-engine bucket (and
    strategy-callback / risk-driven closes route here too — historical
    "combined" bucket). ``trend_long`` / ``trend_short`` hold the
    trend-overlay bucket so its PnL, drawdown, and SL/TP can be reasoned
    about in isolation from the grid. Fills are routed to the correct
    bucket via :attr:`Order.source` by the fill simulator.
    """

    symbol: str
    position_long: Position = field(default_factory=Position)
    position_short: Position = field(default_factory=Position)
    trend_long: Position = field(default_factory=Position)
    trend_short: Position = field(default_factory=Position)
    ema: EMAState = field(default_factory=EMAState)
    volatility: VolatilityState = field(default_factory=VolatilityState)
    mode_long: TradingMode = TradingMode.NORMAL
    mode_short: TradingMode = TradingMode.NORMAL
    # Stage 8 trailing re-entry state. Reset on each side's position
    # open (size 0 -> non-zero), updated every tick from candle.high/low.
    trailing_long: TrailingState = field(default_factory=TrailingState)
    trailing_short: TrailingState = field(default_factory=TrailingState)
    last_price: float = 0.0
    # Contract multiplier for WE / equity calculations.
    # Default 1.0 for USDT-margined linear contracts (the common case);
    # must match the exchange's contractSize for accurate exposure math.
    c_mult: float = 1.0

    def bucket(self, source: "OrderSource", side: Side) -> Position:
        """Return the position bucket targeted by an order's (source, side).

        RISK is routed to the grid bucket for backward compatibility with
        the strategy ``custom_exit`` path, which emits RISK-tagged orders
        against the (combined → now grid) position.
        """
        if source == OrderSource.TREND:
            return self.trend_long if side == Side.LONG else self.trend_short
        return self.position_long if side == Side.LONG else self.position_short


@dataclass
class AccountState:
    balance: float = 0.0
    equity: float = 0.0
    equity_peak: float = 0.0
    pnl_cumsum: float = 0.0
    funding_cumsum: float = 0.0
    # Stage 4: per-source bookkeeping so the risk layer can pause one
    # source (e.g. trend overlay) without throttling the other. Each
    # bucket tracks realized P&L, current marked-to-market equity, and
    # its running peak — drawdowns are measured against the bucket's
    # own peak, normalized to the wallet balance.
    grid_realized_pnl: float = 0.0
    trend_realized_pnl: float = 0.0
    grid_equity: float = 0.0
    trend_equity: float = 0.0
    grid_equity_peak: float = 0.0
    trend_equity_peak: float = 0.0
    # Stage 5: rolling-24h realized losses per source, fed by every fill
    # via add_realized_pnl. Used by the unstuck mechanism to cap the rate
    # at which controlled exposure shedding can realize losses. Only
    # negative deltas are recorded (we don't want gains offsetting the
    # budget — that's a different decision than P&L bookkeeping).
    grid_loss_log: deque[tuple[int, float]] = field(default_factory=deque)
    trend_loss_log: deque[tuple[int, float]] = field(default_factory=deque)
    symbols: dict[str, SymbolState] = field(default_factory=dict)

    def total_wallet_exposure(self, side: Side) -> float:
        twe = 0.0
        denom = max(self.balance, 1e-12)
        for ss in self.symbols.values():
            for pos in self._buckets_for_side(ss, side):
                if pos.is_open:
                    twe += abs(pos.size) * pos.entry_price * ss.c_mult / denom
        return twe

    def update_equity(self):
        grid_upnl = 0.0
        trend_upnl = 0.0
        for ss in self.symbols.values():
            price = ss.last_price
            cm = ss.c_mult
            grid_upnl += ss.position_long.unrealized_pnl(price, Side.LONG) * cm
            grid_upnl += ss.position_short.unrealized_pnl(price, Side.SHORT) * cm
            trend_upnl += ss.trend_long.unrealized_pnl(price, Side.LONG) * cm
            trend_upnl += ss.trend_short.unrealized_pnl(price, Side.SHORT) * cm
        self.equity = self.balance + grid_upnl + trend_upnl
        self.equity_peak = max(self.equity_peak, self.equity)
        self.grid_equity = self.grid_realized_pnl + grid_upnl
        self.trend_equity = self.trend_realized_pnl + trend_upnl
        self.grid_equity_peak = max(self.grid_equity_peak, self.grid_equity)
        self.trend_equity_peak = max(self.trend_equity_peak, self.trend_equity)

    def add_realized_pnl(
        self, source: "OrderSource", amount: float, timestamp_ms: int = 0
    ) -> None:
        """Route a realized P&L delta to the bucket the fill belongs to.

        RISK source closes the grid bucket (strategy ``custom_exit`` compat),
        so it accrues to grid_realized_pnl too — matching the fill-routing
        rule in :meth:`SymbolState.bucket`.

        Negative deltas are also appended to the bucket's rolling-24h
        loss log so the unstuck mechanism can see how much loss budget
        has already been spent.
        """
        if source == OrderSource.TREND:
            self.trend_realized_pnl += amount
            if amount < 0:
                self.trend_loss_log.append((timestamp_ms, amount))
        else:
            self.grid_realized_pnl += amount
            if amount < 0:
                self.grid_loss_log.append((timestamp_ms, amount))

    def loss_24h(self, source: "OrderSource", now_ms: int) -> float:
        """Sum of negative realized P&L for ``source`` in the last 24h.

        Returns a non-positive number (zero if no losses, or losses are
        outside the window). Prunes the bucket's log of entries older
        than 24h as a side effect — keeps the deque bounded.
        """
        log = self.trend_loss_log if source == OrderSource.TREND else self.grid_loss_log
        cutoff = now_ms - _DAY_MS
        while log and log[0][0] < cutoff:
            log.popleft()
        return sum(amount for _, amount in log)

    def source_drawdown_pct(self, source: "OrderSource") -> float:
        """Bucket drawdown as a fraction of the wallet balance.

        Normalized to ``balance`` (not to the bucket's own peak) so a
        tiny bucket peak doesn't produce a misleadingly large fractional
        drawdown. Returns 0 when peak is non-positive (nothing to draw
        down from yet).
        """
        if source == OrderSource.TREND:
            peak, eq = self.trend_equity_peak, self.trend_equity
        else:
            peak, eq = self.grid_equity_peak, self.grid_equity
        denom = max(self.balance, 1e-12)
        return max(0.0, (peak - eq) / denom)

    @staticmethod
    def _buckets_for_side(ss: "SymbolState", side: Side) -> tuple[Position, Position]:
        if side == Side.LONG:
            return (ss.position_long, ss.trend_long)
        return (ss.position_short, ss.trend_short)

    @property
    def drawdown(self) -> float:
        if self.equity_peak <= 0:
            return 0.0
        return 1.0 - self.equity / self.equity_peak


@dataclass
class ExchangeParams:
    qty_step: float = 0.001
    price_step: float = 0.01
    min_qty: float = 0.001
    min_cost: float = 5.0
    c_mult: float = 1.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005


@dataclass
class BacktestResult:
    fills: list[Fill]
    equity_curve: np.ndarray
    final_balance: float
    total_pnl: float
    total_fees: float
    total_funding: float
    n_trades: int
    win_rate: float
    adg: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    grid_pnl: float
    trend_pnl: float
    duration_days: float
