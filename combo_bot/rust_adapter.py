from __future__ import annotations
import logging
from dataclasses import asdict
from combo_bot.types import (
    EMAState,
    ExchangeParams,
    Order,
    OrderSource,
    Position,
    Side,
    TradingMode,
    TrailingState,
    VolatilityState,
)
from combo_bot.grid_engine import GridConfig

logger = logging.getLogger(__name__)

try:
    import combo_futures_core as _rust

    RUST_AVAILABLE = True
except ImportError:
    _rust = None
    RUST_AVAILABLE = False
    logger.info("Rust core not available, falling back to pure Python")


_RUST_OT_TO_SOURCE = {
    "EntryInitialNormalLong": OrderSource.GRID,
    "EntryInitialNormalShort": OrderSource.GRID,
    "EntryGridNormalLong": OrderSource.GRID,
    "EntryGridNormalShort": OrderSource.GRID,
    "EntryTrailingNormalLong": OrderSource.GRID,
    "EntryTrailingNormalShort": OrderSource.GRID,
    "CloseGridLong": OrderSource.GRID,
    "CloseGridShort": OrderSource.GRID,
    "CloseTrailingLong": OrderSource.GRID,
    "CloseTrailingShort": OrderSource.GRID,
    "CloseUnstuckLong": OrderSource.RISK,
    "CloseUnstuckShort": OrderSource.RISK,
    "ClosePanicLong": OrderSource.RISK,
    "ClosePanicShort": OrderSource.RISK,
}


def _config_to_dict(cfg: GridConfig) -> dict:
    return asdict(cfg)


def _ep_to_dict(ep: ExchangeParams) -> dict:
    return asdict(ep)


def _position_to_dict(pos: Position, side: Side | None = None) -> dict:
    """Convert a Python ``Position`` to the dict shape the Rust core expects.

    Round-26 fix: Python guarantees ``Position.size >= 0`` (see types.py
    docstring), but the Rust core uses sign to encode direction —
    ``calc_closes_short`` early-returns when ``size >= 0`` because it
    interprets that as "no short position". Without an explicit sign
    flip here the adapter silently fed Rust an empty short, which then
    emitted no close orders. Caller MUST pass ``side`` so this layer
    can do the conversion; ``side=None`` preserves the legacy behaviour
    (long-only) for callers that don't yet pass it.
    """
    size = pos.size
    if side == Side.SHORT and size > 0:
        size = -size
    return {"size": size, "price": pos.entry_price}


def _state_to_dict(
    balance: float,
    bid: float,
    ask: float,
    ema_state: EMAState,
    volatility: VolatilityState,
) -> dict:
    return {
        "balance": balance,
        "order_book": {"bid": bid, "ask": ask},
        "ema_bands": {
            "upper": ema_state.upper,
            "lower": ema_state.lower,
        },
        "entry_volatility_logrange_ema_1h": volatility.value,
    }


def _trailing_to_dict(trailing: TrailingState, side: Side) -> dict:
    """Map Python TrailingState (extreme + recovery, 2 fields) to Rust
    TrailingPriceBundle (min_since_open / max_since_min / max_since_open /
    min_since_max, 4 fields).

    * LONG:  extreme = running minimum, recovery = running maximum after extreme.
    * SHORT: extreme = running maximum, recovery = running minimum after extreme.
    """
    import sys

    if not trailing.initialized:
        return {
            "min_since_open": sys.float_info.max,
            "max_since_min": sys.float_info.min,
            "max_since_open": sys.float_info.min,
            "min_since_max": sys.float_info.max,
        }
    if side == Side.LONG:
        return {
            "min_since_open": trailing.extreme,
            "max_since_min": trailing.recovery,
            "max_since_open": trailing.recovery,  # best-effort: no independent max_since_open
            "min_since_max": trailing.extreme,  # best-effort
        }
    else:
        return {
            "min_since_open": trailing.recovery,  # best-effort
            "max_since_min": trailing.extreme,  # best-effort
            "max_since_open": trailing.extreme,
            "min_since_max": trailing.recovery,
        }


def _rust_orders_to_python(
    rust_orders: list[dict], symbol: str, side: Side, reduce_only: bool
) -> list[Order]:
    result = []
    for o in rust_orders:
        source = _RUST_OT_TO_SOURCE.get(o["order_type"], OrderSource.GRID)
        result.append(
            Order(
                symbol=symbol,
                side=side,
                price=float(o["price"]),
                qty=float(o["qty"]),
                source=source,
                reduce_only=reduce_only,
            )
        )
    return result


def compute_grid_orders_rust(
    symbol: str,
    side: Side,
    position: Position,
    ema_state: EMAState,
    volatility: VolatilityState,
    balance: float,
    bid: float,
    ask: float,
    exchange_params: ExchangeParams,
    grid_config: GridConfig,
    mode: TradingMode,
    max_levels: int = 5,
    trailing: TrailingState | None = None,
) -> list[Order]:
    """Compute grid orders using the Rust core for performance.

    Returns the same list[Order] structure as the pure-Python GridEngine.
    """
    if not RUST_AVAILABLE:
        raise RuntimeError("Rust core not available; install combo_futures_core wheel")

    bp_dict = _config_to_dict(grid_config)
    ep_dict = _ep_to_dict(exchange_params)
    sp_dict = _state_to_dict(balance, bid, ask, ema_state, volatility)
    pos_dict = _position_to_dict(position, side=side)
    trail_dict = _trailing_to_dict(trailing or TrailingState(), side)

    orders: list[Order] = []

    if mode == TradingMode.PANIC:
        if position.is_open:
            orders.append(
                Order(
                    symbol=symbol,
                    side=side,
                    price=bid if side == Side.LONG else ask,
                    qty=abs(position.size),
                    source=OrderSource.RISK,
                    reduce_only=True,
                )
            )
        return orders

    if position.is_open and mode != TradingMode.PANIC:
        if side == Side.LONG:
            closes = _rust.calc_closes_long(
                bp_dict, ep_dict, sp_dict, pos_dict, trail_dict
            )
        else:
            closes = _rust.calc_closes_short(
                bp_dict, ep_dict, sp_dict, pos_dict, trail_dict
            )
        orders.extend(_rust_orders_to_python(closes, symbol, side, reduce_only=True))

    # NORMAL and AGGRESSIVE both emit entries; TP_ONLY / GRACEFUL_STOP
    # / PANIC do not. AGGRESSIVE falls through to NORMAL Rust sizing
    # for now — the larger-DDF / compressed-spacing tuning the Python
    # path applies for AGGRESSIVE isn't yet plumbed through to the Rust
    # bot_params, so passing it here would have no effect. Without this
    # mode-inclusive branch, AGGRESSIVE silently produced zero entries
    # on the Rust path — the opposite of intent.
    if mode in (TradingMode.NORMAL, TradingMode.AGGRESSIVE):
        wel_cap = grid_config.wallet_exposure_limit
        if side == Side.LONG:
            entries = _rust.calc_entries_long(
                bp_dict,
                ep_dict,
                sp_dict,
                pos_dict,
                trail_dict,
                wel_cap,
                max_levels,
            )
        else:
            entries = _rust.calc_entries_short(
                bp_dict,
                ep_dict,
                sp_dict,
                pos_dict,
                trail_dict,
                wel_cap,
                max_levels,
            )
        orders.extend(_rust_orders_to_python(entries, symbol, side, reduce_only=False))

    return orders


def benchmark_rust_vs_python(n_iterations: int = 10000) -> dict:
    """Compare Rust vs Python performance on a typical grid computation."""
    import time
    from combo_bot.grid_engine import GridEngine

    if not RUST_AVAILABLE:
        return {"error": "Rust core not available"}

    config = GridConfig(
        wallet_exposure_limit=1.0,
        entry_initial_ema_dist=0.005,
        entry_grid_spacing_pct=0.02,
        entry_grid_double_down_factor=1.5,
    )
    ep = ExchangeParams()
    ema = EMAState()
    ema.init([385.0, 620.0], 50000.0)
    for p in [49900, 50100, 49950, 50050]:
        ema.update(p)
    vol = VolatilityState()
    vol.init(1000.0, 0.001)
    pos = Position()

    py_engine = GridEngine(config)

    t0 = time.perf_counter()
    for _ in range(n_iterations):
        py_engine.compute_orders(
            "BTC",
            Side.LONG,
            pos,
            ema,
            vol,
            10000.0,
            0.0,
            ep,
            TradingMode.NORMAL,
        )
    py_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_iterations):
        compute_grid_orders_rust(
            "BTC",
            Side.LONG,
            pos,
            ema,
            vol,
            10000.0,
            49900.0,
            50000.0,
            ep,
            config,
            TradingMode.NORMAL,
            max_levels=5,
        )
    rust_time = time.perf_counter() - t0

    return {
        "iterations": n_iterations,
        "python_seconds": round(py_time, 3),
        "rust_seconds": round(rust_time, 3),
        "speedup": round(py_time / rust_time, 2) if rust_time > 0 else float("inf"),
    }
