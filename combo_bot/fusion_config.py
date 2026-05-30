"""Config-driven factory for the Stage 7-11 fusion layer components.

Reads a JSON config blob (the same dict ``main.py`` already passes to
BacktestConfig / LiveConfig) and produces the optional sizer /
protection objects that ``Backtester`` and ``LiveTrader`` accept via
their constructors. Previously these were ``None`` by default and the
CLI commands didn't read them at all — users could put
``correlation_gate`` / ``kelly_sizer`` / ``protections`` blocks in
their config and they'd be silently ignored.

The factory is intentionally tolerant: a missing block disables that
component, an ``enabled: false`` block also disables it, and unknown
keys inside a block are passed through to the dataclass so future
fields don't need a code change here.
"""

from __future__ import annotations

import logging
import importlib
from typing import Any

from combo_bot.correlation import CorrelationGate, CorrelationGateConfig
from combo_bot.protections import (
    CooldownPeriod,
    CooldownPeriodConfig,
    IProtection,
    StoplossGuard,
    StoplossGuardConfig,
)
from combo_bot.regime import RegimeArbiterConfig
from combo_bot.sizing import KellySizer, KellySizerConfig
from combo_bot.strategy import DefaultStrategy, ExampleTrendStrategy, IStrategy
from combo_bot.vol_target import VolTargetSizer, VolTargetSizerConfig

logger = logging.getLogger(__name__)


# Map from "type" string in JSON to (ProtectionClass, ConfigClass).
_PROTECTION_REGISTRY: dict[str, tuple[type[IProtection], type]] = {
    "stoploss_guard": (StoplossGuard, StoplossGuardConfig),
    "cooldown_period": (CooldownPeriod, CooldownPeriodConfig),
}

_STRATEGY_REGISTRY: dict[str, type[IStrategy]] = {
    "DefaultStrategy": DefaultStrategy,
    "ExampleTrendStrategy": ExampleTrendStrategy,
}


def _block_enabled(block: dict[str, Any] | None) -> bool:
    """A block is active when it exists and ``enabled`` isn't False."""
    if not block:
        return False
    return bool(block.get("enabled", True))


def _build_dataclass(klass: type, block: dict[str, Any]) -> Any:
    """Construct a dataclass, ignoring keys that aren't fields."""
    valid = {k for k in klass.__dataclass_fields__}
    kwargs = {k: v for k, v in block.items() if k in valid}
    return klass(**kwargs)


def build_regime_config(cfg: dict[str, Any]) -> RegimeArbiterConfig:
    """Build a RegimeArbiterConfig from the ``regime`` block of cfg."""
    return _build_dataclass(RegimeArbiterConfig, cfg.get("regime", {}) or {})


def build_kelly_sizer(cfg: dict[str, Any]) -> KellySizer | None:
    block = cfg.get("kelly_sizer")
    if not _block_enabled(block):
        return None
    return KellySizer(_build_dataclass(KellySizerConfig, block))


def build_correlation_gate(cfg: dict[str, Any]) -> CorrelationGate | None:
    block = cfg.get("correlation_gate")
    if not _block_enabled(block):
        return None
    return CorrelationGate(_build_dataclass(CorrelationGateConfig, block))


def build_vol_target_sizer(cfg: dict[str, Any]) -> VolTargetSizer | None:
    block = cfg.get("vol_target_sizer")
    if not _block_enabled(block):
        return None
    sizer_cfg = _build_dataclass(VolTargetSizerConfig, block)
    # When the user didn't pin periods_per_year explicitly, derive it
    # from bar_interval_minutes so Sharpe-style annualization matches the
    # candle cadence backtest/live actually consume. Without this an
    # hourly live deployment annualizes with the 1m default (525600) and
    # over-throttles size by ~60×.
    if "periods_per_year" not in (block or {}):
        bar_min = float(cfg.get("bar_interval_minutes") or 0.0)
        if bar_min <= 0:
            bar_min = _timeframe_to_minutes(str(cfg.get("timeframe", "1m")))
        if bar_min > 0:
            sizer_cfg.periods_per_year = int(round(525_600 / bar_min))
    return VolTargetSizer(sizer_cfg)


# Shared with main.py CLI; mirrored here so fusion_config doesn't need
# to reach back through it.
_TIMEFRAME_TO_MIN = {
    "1m": 1.0,
    "3m": 3.0,
    "5m": 5.0,
    "15m": 15.0,
    "30m": 30.0,
    "1h": 60.0,
    "2h": 120.0,
    "4h": 240.0,
    "6h": 360.0,
    "8h": 480.0,
    "12h": 720.0,
    "1d": 1440.0,
}


def _timeframe_to_minutes(tf: str) -> float:
    return _TIMEFRAME_TO_MIN.get(tf, 1.0)


def build_protections(cfg: dict[str, Any]) -> list[IProtection]:
    """Build a list of IProtection from the ``protections`` config array.

    Each entry must include ``type`` matching a registered protection.
    Unknown types are logged and skipped.
    """
    blocks = cfg.get("protections") or []
    if not isinstance(blocks, list):
        logger.warning(
            "[fusion_config] 'protections' must be a list, got %r — ignoring",
            type(blocks).__name__,
        )
        return []
    out: list[IProtection] = []
    for block in blocks:
        if not _block_enabled(block):
            continue
        ptype = block.get("type")
        entry = _PROTECTION_REGISTRY.get(ptype)
        if entry is None:
            logger.warning(
                "[fusion_config] unknown protection type %r — skipping",
                ptype,
            )
            continue
        prot_cls, prot_cfg_cls = entry
        out.append(prot_cls(_build_dataclass(prot_cfg_cls, block)))
    return out


def build_strategy(cfg: dict[str, Any]) -> IStrategy | None:
    """Instantiate a user strategy from config.

    Accepted forms:

    * ``"strategy": "ExampleTrendStrategy"`` — no-args.
    * ``"strategy": "package.module:ClassName"`` — no-args.
    * ``"strategy": {"class": "package.module:ClassName"}`` — no-args.
    * ``"strategy": {"class": "...", "params": {"rsi_period": 21}}``
      — kwargs from ``params`` are splatted into the constructor.

    If ``params`` is supplied but the strategy's ``__init__`` doesn't
    accept those kwargs, the resulting ``TypeError`` propagates so the
    config error is loud rather than silently dropped.
    """
    block = cfg.get("strategy")
    if not block:
        return None
    params: dict[str, Any] = {}
    if isinstance(block, dict):
        if not _block_enabled(block):
            return None
        spec = block.get("class") or block.get("path") or block.get("name")
        raw_params = block.get("params") or {}
        if isinstance(raw_params, dict):
            params = {k: v for k, v in raw_params.items() if not k.startswith("_")}
        else:
            logger.warning(
                "[fusion_config] strategy.params must be a dict, got %r — "
                "ignoring parameters",
                type(raw_params).__name__,
            )
    else:
        spec = block
    if not isinstance(spec, str) or not spec.strip():
        logger.warning("[fusion_config] invalid strategy spec %r — ignoring", spec)
        return None
    spec = spec.strip()
    klass = _STRATEGY_REGISTRY.get(spec)
    if klass is None:
        klass = _import_strategy_class(spec)
    if not issubclass(klass, IStrategy):
        raise TypeError(f"strategy {spec!r} is not an IStrategy subclass")
    if params:
        return klass(**params)
    return klass()


def _import_strategy_class(spec: str) -> type:
    if ":" in spec:
        module_name, class_name = spec.split(":", 1)
    else:
        module_name, class_name = spec.rsplit(".", 1)
    module = importlib.import_module(module_name)
    klass = getattr(module, class_name)
    if not isinstance(klass, type):
        raise TypeError(f"strategy {spec!r} did not resolve to a class")
    return klass


def build_fusion(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a dict ready to splat into Backtester/LiveTrader kwargs.

    Keys produced: ``protections``, ``kelly_sizer``, ``correlation_gate``,
    ``vol_target_sizer``. Each value is the constructed component or
    ``None`` (for sizers) / ``[]`` (for protections).
    """
    return {
        "strategy": build_strategy(cfg),
        "protections": build_protections(cfg),
        "kelly_sizer": build_kelly_sizer(cfg),
        "correlation_gate": build_correlation_gate(cfg),
        "vol_target_sizer": build_vol_target_sizer(cfg),
    }


__all__ = [
    "build_fusion",
    "build_regime_config",
    "build_strategy",
    "build_kelly_sizer",
    "build_correlation_gate",
    "build_vol_target_sizer",
    "build_protections",
]
