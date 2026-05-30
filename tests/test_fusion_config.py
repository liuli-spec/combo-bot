"""Verify the fusion_config factory wires Stage 7-11 components from JSON."""

from __future__ import annotations


from combo_bot.correlation import CorrelationGate
from combo_bot.fusion_config import (
    build_correlation_gate,
    build_fusion,
    build_kelly_sizer,
    build_protections,
    build_regime_config,
    build_vol_target_sizer,
    build_strategy,
)
from combo_bot.protections import CooldownPeriod, StoplossGuard
from combo_bot.sizing import KellySizer
from combo_bot.strategy import ExampleTrendStrategy
from combo_bot.vol_target import VolTargetSizer


def test_empty_config_disables_every_component():
    fusion = build_fusion({})
    assert fusion["protections"] == []
    assert fusion["kelly_sizer"] is None
    assert fusion["correlation_gate"] is None
    assert fusion["vol_target_sizer"] is None
    assert fusion["strategy"] is None


def test_enabled_false_block_disables_sizer():
    cfg = {"kelly_sizer": {"enabled": False, "window": 50}}
    assert build_kelly_sizer(cfg) is None


def test_kelly_sizer_built_from_block():
    cfg = {
        "kelly_sizer": {
            "enabled": True,
            "window": 75,
            "min_samples": 10,
            "fractional_kelly": 0.5,
            "max_fraction": 0.8,
        },
    }
    sizer = build_kelly_sizer(cfg)
    assert isinstance(sizer, KellySizer)
    assert sizer.config.window == 75
    assert sizer.config.fractional_kelly == 0.5
    assert sizer.config.max_fraction == 0.8


def test_correlation_gate_built_from_block():
    cfg = {
        "correlation_gate": {
            "enabled": True,
            "window": 90,
            "min_samples": 40,
            "soft_threshold": 0.5,
            "hard_threshold": 0.85,
        },
    }
    gate = build_correlation_gate(cfg)
    assert isinstance(gate, CorrelationGate)
    assert gate.config.window == 90
    assert gate.config.hard_threshold == 0.85


def test_vol_target_sizer_built_from_block():
    cfg = {
        "vol_target_sizer": {
            "enabled": True,
            "target_annual_vol": 0.5,
            "window": 720,
            "min_samples": 30,
        },
    }
    sizer = build_vol_target_sizer(cfg)
    assert isinstance(sizer, VolTargetSizer)
    assert sizer.config.target_annual_vol == 0.5
    assert sizer.config.window == 720


def test_protections_list_with_two_types():
    cfg = {
        "protections": [
            {"type": "stoploss_guard", "trade_limit": 5, "stop_duration_ms": 600_000},
            {"type": "cooldown_period", "stop_duration_ms": 120_000},
        ],
    }
    prots = build_protections(cfg)
    assert len(prots) == 2
    assert isinstance(prots[0], StoplossGuard)
    assert prots[0].config.trade_limit == 5
    assert isinstance(prots[1], CooldownPeriod)
    assert prots[1].config.stop_duration_ms == 120_000


def test_unknown_protection_type_is_skipped():
    cfg = {"protections": [{"type": "wat_no", "foo": 1}]}
    assert build_protections(cfg) == []


def test_unknown_keys_in_block_are_ignored():
    """Tolerant to future fields — unknown keys pass through silently."""
    cfg = {
        "kelly_sizer": {"window": 100, "future_field_not_yet_implemented": "x"},
    }
    sizer = build_kelly_sizer(cfg)
    assert sizer is not None
    assert sizer.config.window == 100


def test_regime_config_built_from_block():
    cfg = {
        "regime": {
            "aggressive_strength": 0.3,
            "overlay_min_conviction": 0.4,
            "overlay_conviction_curve": 2.0,
        },
    }
    rc = build_regime_config(cfg)
    assert rc.aggressive_strength == 0.3
    assert rc.overlay_min_conviction == 0.4
    assert rc.overlay_conviction_curve == 2.0


def test_full_fusion_payload_is_splattable_into_backtester_kwargs():
    """Smoke test: the returned dict's keys must all match Backtester
    constructor parameters so ``**fusion`` actually works."""
    import inspect

    from combo_bot.backtest import Backtester

    cfg = {
        "kelly_sizer": {"enabled": True},
        "correlation_gate": {"enabled": True},
        "vol_target_sizer": {"enabled": True},
        "protections": [{"type": "stoploss_guard"}],
    }
    fusion = build_fusion(cfg)
    sig = inspect.signature(Backtester.__init__)
    bt_params = set(sig.parameters.keys())
    for key in fusion:
        assert (
            key in bt_params
        ), f"fusion key {key!r} not in Backtester params {bt_params}"


def test_strategy_loaded_from_dotted_path_block():
    cfg = {"strategy": {"class": "combo_bot.strategy:ExampleTrendStrategy"}}
    strategy = build_strategy(cfg)
    assert isinstance(strategy, ExampleTrendStrategy)


def test_strategy_loaded_from_builtin_name():
    cfg = {"strategy": "ExampleTrendStrategy"}
    strategy = build_strategy(cfg)
    assert isinstance(strategy, ExampleTrendStrategy)
