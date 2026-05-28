from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


class TradingMode(str, Enum):
    NORMAL = "normal"
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
    size: float = 0.0
    entry_price: float = 0.0

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
        return self.size * (price - self.entry_price)


@dataclass
class Order:
    symbol: str
    side: Side
    price: float
    qty: float
    source: OrderSource
    reduce_only: bool = False

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


@dataclass
class TrendSignal:
    direction: float
    strength: float
    regime: TrendRegime


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
            self.values[i] = self.alphas[i] * price + (1.0 - self.alphas[i]) * self.values[i]

    @property
    def lower(self) -> float:
        return min(self.values) if self.values else 0.0

    @property
    def upper(self) -> float:
        return max(self.values) if self.values else 0.0


@dataclass
class VolatilityState:
    ema_span_hours: float = 1000.0
    value: float = 0.0
    alpha: float = 0.0
    initialized: bool = False

    def init(self, span_hours: float, initial_range: float):
        self.ema_span_hours = span_hours
        self.alpha = 2.0 / (span_hours * 60.0 + 1.0)
        self.value = initial_range
        self.initialized = True

    def update(self, log_range: float):
        self.value = self.alpha * log_range + (1.0 - self.alpha) * self.value


@dataclass
class SymbolState:
    symbol: str
    position_long: Position = field(default_factory=Position)
    position_short: Position = field(default_factory=Position)
    ema: EMAState = field(default_factory=EMAState)
    volatility: VolatilityState = field(default_factory=VolatilityState)
    mode_long: TradingMode = TradingMode.NORMAL
    mode_short: TradingMode = TradingMode.NORMAL
    trailing_min_since_open: float = 0.0
    trailing_max_since_open: float = 0.0
    last_price: float = 0.0


@dataclass
class AccountState:
    balance: float = 0.0
    equity: float = 0.0
    equity_peak: float = 0.0
    pnl_cumsum: float = 0.0
    funding_cumsum: float = 0.0
    symbols: dict[str, SymbolState] = field(default_factory=dict)

    def total_wallet_exposure(self, side: Side) -> float:
        twe = 0.0
        for ss in self.symbols.values():
            pos = ss.position_long if side == Side.LONG else ss.position_short
            if pos.is_open:
                twe += abs(pos.size) * pos.entry_price / max(self.balance, 1e-12)
        return twe

    def update_equity(self):
        upnl = 0.0
        for ss in self.symbols.values():
            price = ss.last_price
            upnl += ss.position_long.unrealized_pnl(price, Side.LONG)
            upnl += ss.position_short.unrealized_pnl(price, Side.SHORT)
        self.equity = self.balance + upnl
        self.equity_peak = max(self.equity_peak, self.equity)

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
