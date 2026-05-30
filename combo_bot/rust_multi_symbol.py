from __future__ import annotations
import logging
from dataclasses import dataclass
import numpy as np

from combo_bot.grid_engine import GridConfig
from combo_bot.rust_backtest import (
    RUST_AVAILABLE,
    candles_to_array,
    exchange_params_to_dict,
    grid_config_to_dict,
)
from combo_bot.types import Candle, ExchangeParams

logger = logging.getLogger(__name__)

if RUST_AVAILABLE:
    import combo_futures_core as _rust
else:
    _rust = None


@dataclass
class MultiSymbolConfig:
    starting_balance: float = 10000.0
    funding_rate: float = 0.0001
    funding_interval_bars: int = 480
    liquidation_threshold_pct: float = 0.05
    max_grid_levels: int = 5
    n_positions_max: int = 5
    forager_volume_weight: float = 0.23
    forager_volatility_weight: float = 0.71
    forager_ema_readiness_weight: float = 0.06


@dataclass
class MultiSymbolResult:
    final_balance: float
    final_equity: float
    max_drawdown: float
    n_trades: int
    liquidated: bool
    liquidation_bar: int | None
    equity_curve: np.ndarray
    fills: list[dict]
    final_positions: list[dict]
    symbols: list[str]

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
        return float(np.mean(returns) / std * np.sqrt(525600))

    @property
    def sortino_ratio(self) -> float:
        if self.equity_curve.size < 2:
            return 0.0
        returns = np.diff(self.equity_curve) / np.maximum(self.equity_curve[:-1], 1e-12)
        downside = returns[returns < 0]
        if downside.size == 0:
            return 0.0
        ds = float(np.std(downside))
        if ds < 1e-12:
            return 0.0
        return float(np.mean(returns) / ds * np.sqrt(525600))

    @property
    def calmar_ratio(self) -> float:
        if self.max_drawdown < 1e-12:
            return 0.0
        return self.total_return / self.max_drawdown

    def fills_for_symbol(self, idx: int) -> list[dict]:
        return [f for f in self.fills if f["symbol_idx"] == idx]


def run_multi_symbol_backtest(
    candle_data: dict[str, list[Candle] | np.ndarray],
    grid_configs: dict[str, GridConfig],
    exchange_params: dict[str, ExchangeParams] | None = None,
    bt_config: MultiSymbolConfig | None = None,
) -> MultiSymbolResult:
    """Run a multi-symbol backtest with shared balance and Forager selection.

    Args:
        candle_data: maps symbol -> OHLCV (list[Candle] or [N, 5] ndarray)
        grid_configs: per-symbol GridConfig (one per symbol)
        exchange_params: per-symbol ExchangeParams (defaults applied per symbol)
        bt_config: backtest-wide config (balance, funding, forager weights, ...)

    All symbols must have the same number of bars (use forward-fill if needed).
    """
    if not RUST_AVAILABLE:
        raise RuntimeError("Rust extension not installed")
    if not candle_data:
        raise ValueError("candle_data must not be empty")

    symbols = list(candle_data.keys())
    cfg = bt_config or MultiSymbolConfig()

    # Convert candles to numpy arrays
    candle_arrays: list[np.ndarray] = []
    for s in symbols:
        cd = candle_data[s]
        if isinstance(cd, np.ndarray):
            arr = cd.astype(np.float64, copy=False)
        else:
            arr = candles_to_array(cd)
        if arr.shape[1] != 5:
            raise ValueError(f"{s}: expected 5 columns, got {arr.shape[1]}")
        candle_arrays.append(arr)

    # All symbols must have same number of bars
    n_bars = candle_arrays[0].shape[0]
    for i, arr in enumerate(candle_arrays):
        if arr.shape[0] != n_bars:
            raise ValueError(
                f"{symbols[i]}: {arr.shape[0]} bars but first symbol has {n_bars}"
            )

    # Build per-symbol param dicts
    bp_dicts = [grid_config_to_dict(grid_configs[s]) for s in symbols]
    ep_default = ExchangeParams()
    ep_dicts = [
        exchange_params_to_dict((exchange_params or {}).get(s, ep_default))
        for s in symbols
    ]

    raw = _rust.run_multi_symbol_backtest(
        candle_arrays,
        bp_dicts,
        ep_dicts,
        starting_balance=cfg.starting_balance,
        funding_rate=cfg.funding_rate,
        funding_interval_bars=cfg.funding_interval_bars,
        liquidation_threshold_pct=cfg.liquidation_threshold_pct,
        max_grid_levels=cfg.max_grid_levels,
        n_positions_max=cfg.n_positions_max,
        forager_volume_weight=cfg.forager_volume_weight,
        forager_volatility_weight=cfg.forager_volatility_weight,
        forager_ema_readiness_weight=cfg.forager_ema_readiness_weight,
    )

    return MultiSymbolResult(
        final_balance=float(raw["final_balance"]),
        final_equity=float(raw["final_equity"]),
        max_drawdown=float(raw["max_drawdown"]),
        n_trades=int(raw["n_trades"]),
        liquidated=bool(raw["liquidated"]),
        liquidation_bar=raw["liquidation_bar"],
        equity_curve=np.asarray(raw["equity_curve"], dtype=np.float64),
        fills=list(raw["fills"]),
        final_positions=list(raw["final_positions"]),
        symbols=symbols,
    )
