from __future__ import annotations
import logging
from dataclasses import asdict, dataclass, field
from typing import Any
import numpy as np

from combo_bot.grid_engine import GridConfig
from combo_bot.types import Candle, ExchangeParams

logger = logging.getLogger(__name__)

try:
    import combo_futures_core as _rust
    RUST_AVAILABLE = True
except ImportError:
    _rust = None
    RUST_AVAILABLE = False


@dataclass
class RustBacktestConfig:
    starting_balance: float = 10000.0
    funding_rate: float = 0.0001
    # Funding fires every N bars. Default assumes 1-minute bars
    # (480 = 8h funding). For 1h bars set to 8, etc. Must match the
    # bar cadence of the candles passed to ``run_rust_backtest``.
    funding_interval_bars: int = 480
    liquidation_threshold_pct: float = 0.05
    max_grid_levels: int = 5
    # Bar cadence for Sharpe/Sortino annualization. 1m → 525600 periods
    # per year; 1h → 8760; daily → 365. Defaults to 1m to preserve
    # legacy behaviour. Mis-setting this only mis-scales the reported
    # ratios — the actual P&L is unaffected.
    bar_interval_minutes: float = 1.0


@dataclass
class RustBacktestResult:
    final_balance: float
    final_equity: float
    max_drawdown: float
    n_trades: int
    liquidated: bool
    liquidation_bar: int | None
    equity_curve: np.ndarray
    fills: list[dict]
    # Periods per year used for Sharpe / Sortino annualization. Default
    # 525600 = minutes per year (1m bars); 8760 = hourly; 365 = daily.
    # Populated by ``run_rust_backtest`` from ``RustBacktestConfig
    # .bar_interval_minutes``.
    periods_per_year: float = 525_600.0

    @property
    def total_return(self) -> float:
        if self.equity_curve.size == 0:
            return 0.0
        return float(self.equity_curve[-1] / self.equity_curve[0] - 1.0)

    @property
    def sharpe_ratio(self) -> float:
        if self.equity_curve.size < 2:
            return 0.0
        returns = np.diff(self.equity_curve) / np.maximum(self.equity_curve[:-1], 1e-12)
        std = float(np.std(returns))
        if std < 1e-12:
            return 0.0
        return float(np.mean(returns) / std * np.sqrt(self.periods_per_year))

    @property
    def sortino_ratio(self) -> float:
        if self.equity_curve.size < 2:
            return 0.0
        returns = np.diff(self.equity_curve) / np.maximum(self.equity_curve[:-1], 1e-12)
        downside = returns[returns < 0]
        if downside.size == 0:
            return 0.0
        ds_std = float(np.std(downside))
        if ds_std < 1e-12:
            return 0.0
        return float(np.mean(returns) / ds_std * np.sqrt(self.periods_per_year))

    @property
    def calmar_ratio(self) -> float:
        if self.max_drawdown < 1e-12:
            return 0.0
        return self.total_return / self.max_drawdown


def candles_to_array(candles: list[Candle]) -> np.ndarray:
    """Convert list[Candle] to the [N, 5] float64 numpy array Rust expects."""
    n = len(candles)
    arr = np.empty((n, 5), dtype=np.float64)
    for i, c in enumerate(candles):
        arr[i, 0] = c.open
        arr[i, 1] = c.high
        arr[i, 2] = c.low
        arr[i, 3] = c.close
        arr[i, 4] = c.volume
    return arr


def grid_config_to_dict(cfg: GridConfig) -> dict[str, Any]:
    """Convert GridConfig to the dict Rust expects, with all required fields."""
    d = asdict(cfg)
    d.setdefault("entry_trailing_threshold_pct", 0.01)
    d.setdefault("entry_trailing_retracement_pct", 0.005)
    d.setdefault("entry_trailing_grid_ratio", -1.0)
    d.setdefault("close_trailing_threshold_pct", 0.01)
    d.setdefault("close_trailing_retracement_pct", 0.004)
    d.setdefault("close_trailing_grid_ratio", -1.0)
    d.setdefault("close_trailing_qty_pct", 0.5)
    d.setdefault("unstuck_threshold", 0.8)
    d.setdefault("unstuck_close_pct", 0.05)
    d.setdefault("unstuck_ema_dist", 0.01)
    d.setdefault("unstuck_loss_allowance_pct", 0.001)
    d.setdefault("risk_we_excess_allowance_pct", 0.0)
    d.setdefault("risk_wel_enforcer_threshold", 0.98)
    d.setdefault("risk_twel_enforcer_threshold", 0.95)
    return d


def exchange_params_to_dict(ep: ExchangeParams) -> dict[str, Any]:
    return asdict(ep)


def run_rust_backtest(
    candles: list[Candle] | np.ndarray,
    grid_config: GridConfig,
    exchange_params: ExchangeParams | None = None,
    bt_config: RustBacktestConfig | None = None,
) -> RustBacktestResult:
    """Run a Rust backtest and return a structured result.

    `candles` may be a list of Candle dataclasses or a pre-built [N, 5] numpy
    array (open, high, low, close, volume).
    """
    if not RUST_AVAILABLE:
        raise RuntimeError(
            "combo_futures_core (Rust extension) is not installed. "
            "Build it with: cd rust && maturin build --release && pip install target/wheels/*.whl"
        )

    if isinstance(candles, np.ndarray):
        arr = candles.astype(np.float64, copy=False)
        if arr.shape[1] != 5:
            raise ValueError(f"candles array must have 5 columns, got {arr.shape[1]}")
    else:
        arr = candles_to_array(candles)

    bp_dict = grid_config_to_dict(grid_config)
    ep_dict = exchange_params_to_dict(exchange_params or ExchangeParams())
    cfg = bt_config or RustBacktestConfig()

    raw = _rust.run_backtest(
        arr,
        bp_dict,
        ep_dict,
        starting_balance=cfg.starting_balance,
        funding_rate=cfg.funding_rate,
        funding_interval_bars=cfg.funding_interval_bars,
        liquidation_threshold_pct=cfg.liquidation_threshold_pct,
        max_grid_levels=cfg.max_grid_levels,
    )

    # Convert bar cadence to "periods per year" for Sharpe/Sortino
    # annualization. minutes_per_year / bar_minutes = bars_per_year.
    periods_per_year = max(1.0, 525_600.0 / max(cfg.bar_interval_minutes, 1e-9))
    return RustBacktestResult(
        final_balance=float(raw["final_balance"]),
        final_equity=float(raw["final_equity"]),
        max_drawdown=float(raw["max_drawdown"]),
        n_trades=int(raw["n_trades"]),
        liquidated=bool(raw["liquidated"]),
        liquidation_bar=raw["liquidation_bar"],
        equity_curve=np.asarray(raw["equity_curve"], dtype=np.float64),
        fills=list(raw["fills"]),
        periods_per_year=periods_per_year,
    )


def parallel_backtest_grid(
    candles: list[Candle] | np.ndarray,
    grid_configs: list[GridConfig],
    exchange_params: ExchangeParams | None = None,
    bt_config: RustBacktestConfig | None = None,
    n_workers: int | None = None,
) -> list[RustBacktestResult]:
    """Run many backtests in parallel — one per GridConfig.

    Uses a process pool because the Rust extension releases the GIL but
    Python-side serialization dominates for small candle counts.
    """
    if not RUST_AVAILABLE:
        raise RuntimeError("Rust extension not available")
    if not grid_configs:
        return []

    if isinstance(candles, list):
        arr = candles_to_array(candles)
    else:
        arr = candles

    if n_workers is None or n_workers <= 1:
        return [run_rust_backtest(arr, cfg, exchange_params, bt_config) for cfg in grid_configs]

    from concurrent.futures import ProcessPoolExecutor
    ep = exchange_params or ExchangeParams()
    cfg = bt_config or RustBacktestConfig()
    args = [(arr, gcfg, ep, cfg) for gcfg in grid_configs]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(_run_one, args))


def _run_one(args: tuple) -> RustBacktestResult:
    arr, gcfg, ep, cfg = args
    return run_rust_backtest(arr, gcfg, ep, cfg)
