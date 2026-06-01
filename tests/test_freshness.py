"""Unit tests for the epoch-based FreshnessLedger.

Covers the self-healing symbol-block lifecycle: a block fires when a
surface goes stale and clears automatically once the required surfaces
refresh at/after the block's min_epoch.
"""

from __future__ import annotations

from combo_bot.freshness import (
    FreshnessLedger,
    candle_surface,
)


def test_stamp_advances_surface_epoch():
    led = FreshnessLedger()
    led.begin_epoch()  # epoch 1
    changed = led.stamp("balance", signature=100.0, now_ms=1)
    assert changed is True
    assert led.surface_epoch("balance") == 1
    # Same signature next epoch → not "changed" but epoch advances.
    led.begin_epoch()  # epoch 2
    changed2 = led.stamp("balance", signature=100.0, now_ms=2)
    assert changed2 is False
    assert led.surface_epoch("balance") == 2


def test_symbol_block_self_heals_when_surface_refreshes():
    led = FreshnessLedger()
    sym = "BTC/USDT:USDT"
    surf = candle_surface(sym)

    # Epoch 1: candle fetch FAILED → block requiring the candle surface
    # to refresh at/after epoch 2.
    led.begin_epoch()  # epoch 1
    led.flag_symbol_block(
        sym,
        reason="candle_fetch_failed",
        required_surfaces={surf},
        min_epoch=led.epoch + 1,  # 2
        detected_ms=1,
    )
    assert led.is_blocked(sym) is True

    # Epoch 2: candle still not stamped → still blocked.
    led.begin_epoch()  # epoch 2
    assert led.is_blocked(sym) is True

    # Epoch 2: candle refreshes → block clears (epoch 2 >= min_epoch 2).
    led.stamp(surf, signature=("bar", 123), now_ms=2)
    assert led.is_blocked(sym) is False


def test_block_requires_all_surfaces():
    led = FreshnessLedger()
    sym = "ETH/USDT:USDT"
    required = {"balance", "positions"}
    led.begin_epoch()  # 1
    led.flag_symbol_block(
        sym,
        reason="account_stale",
        required_surfaces=required,
        min_epoch=led.epoch + 1,  # 2
        detected_ms=1,
    )
    led.begin_epoch()  # 2
    # Refresh all but one — still blocked.
    led.stamp("balance", now_ms=2)
    assert led.is_blocked(sym) is True
    # Refresh the last one — clears.
    led.stamp("positions", now_ms=2)
    assert led.is_blocked(sym) is False


def test_stale_refresh_before_min_epoch_does_not_clear():
    """A surface stamped in the SAME epoch the block was raised does not
    satisfy a min_epoch of epoch+1 — it must be a genuinely newer refresh."""
    led = FreshnessLedger()
    sym = "BTC/USDT:USDT"
    surf = candle_surface(sym)
    led.begin_epoch()  # 1
    # Stamp at epoch 1 first…
    led.stamp(surf, now_ms=1)
    # …then flag a block requiring refresh at/after epoch 2.
    led.flag_symbol_block(
        sym,
        reason="candle_fetch_failed",
        required_surfaces={surf},
        min_epoch=led.epoch + 1,  # 2
        detected_ms=1,
    )
    # The epoch-1 stamp must NOT satisfy the epoch-2 requirement.
    assert led.is_blocked(sym) is True


def test_manual_clear_symbol():
    led = FreshnessLedger()
    sym = "BTC/USDT:USDT"
    led.begin_epoch()
    led.flag_symbol_block(
        sym,
        reason="x",
        required_surfaces={candle_surface(sym)},
        min_epoch=99,
        detected_ms=1,
    )
    assert led.is_blocked(sym) is True
    led.clear_symbol(sym)
    assert led.is_blocked(sym) is False


def test_per_symbol_candle_isolation():
    """One symbol's stale candle must not block another symbol."""
    led = FreshnessLedger()
    a, b = "BTC/USDT:USDT", "ETH/USDT:USDT"
    led.begin_epoch()  # 1
    led.flag_symbol_block(
        a,
        reason="candle_fetch_failed",
        required_surfaces={candle_surface(a)},
        min_epoch=led.epoch + 1,
        detected_ms=1,
    )
    assert led.is_blocked(a) is True
    assert led.is_blocked(b) is False
