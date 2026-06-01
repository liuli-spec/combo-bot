"""Epoch-based data-surface freshness ledger.

Ported from passivbot's ``live/freshness.py`` and adapted to combo-bot's
LiveTrader. The idea: every tick the trader re-fetches a set of *data
surfaces* (balance, positions, open orders, fills, per-symbol candles).
Each successful refresh ``stamp``s the surface with the current epoch.
When the trader detects that a symbol's view might be stale (a candle
fetch failed, a self-order vanished, …) it ``flag_symbol_block``s that
symbol, requiring the named surfaces to refresh AFTER a given epoch
before the block clears. The block self-heals: as soon as the required
surfaces are stamped at or beyond ``min_epoch``, it is removed.

This generalises combo-bot's existing per-surface ad-hoc handling
(account refresh = skip whole tick; fills = STUCK / last_poll_failed)
into one uniform, self-healing, per-symbol gate that also covers the
previously-silent gap: a per-symbol candle fetch failure that left the
trader generating orders off a stale price / EMA.

The ledger is a pure in-memory runtime structure — it is NOT persisted.
A restart starts at epoch 0 and re-evaluates freshness from the first
tick's refresh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Account-wide surfaces combo-bot actually refreshes and stamps once per
# tick (see LiveTrader._refresh_account). Kept to exactly what is stamped:
# a surface listed here but never stamped would make any block requiring
# it un-healable. Fill freshness is handled separately by the fill-event
# STUCK / last_poll_failed signals; open-order refresh failure by the
# reconcile path — neither flows through this ledger, so they are not
# listed here. Per-symbol candles use candle_surface() keys, stamped
# on demand rather than pre-registered.
ACCOUNT_SURFACES = frozenset({"balance", "positions"})


def candle_surface(symbol: str) -> str:
    """Per-symbol candle surface key. Candles are fetched per symbol, so
    one symbol's stale feed must not gate the others."""
    return f"candle:{symbol}"


@dataclass
class SurfaceState:
    name: str
    updated_ms: int = 0
    epoch: int = 0
    generation: int = 0
    signature: Any = None
    changed: bool = False


@dataclass
class SymbolBlock:
    symbol: str
    reason: str
    required_surfaces: frozenset[str]
    min_epoch: int
    detected_ms: int
    details: dict[str, Any] = field(default_factory=dict)


class FreshnessLedger:
    """Track live data-surface freshness and per-symbol execution blocks."""

    def __init__(self, *, now_ms: int = 0) -> None:
        self.epoch = 0
        self.surfaces: dict[str, SurfaceState] = {
            surface: SurfaceState(name=surface) for surface in ACCOUNT_SURFACES
        }
        self.symbol_blocks: dict[str, SymbolBlock] = {}
        self.created_ms = int(now_ms or 0)

    # ── epoch / stamping ────────────────────────────────────────────

    def begin_epoch(self) -> int:
        """Advance the tick epoch. Call once at the top of each tick."""
        self.epoch += 1
        return self.epoch

    def stamp(
        self,
        surface: str,
        signature: Any = None,
        *,
        now_ms: int,
        epoch: int | None = None,
    ) -> bool:
        """Mark ``surface`` as freshly refreshed at the current epoch.
        Returns True if the signature changed since the last stamp.
        Clears any symbol block whose required surfaces are now satisfied.
        """
        state = self.surfaces.get(surface)
        if state is None:
            state = SurfaceState(name=surface)
            self.surfaces[surface] = state
        changed = state.signature != signature
        state.signature = signature
        state.updated_ms = int(now_ms)
        state.epoch = int(self.epoch if epoch is None else epoch)
        state.changed = changed
        if changed:
            state.generation += 1
        self._clear_satisfied_symbol_blocks()
        return changed

    # ── queries ─────────────────────────────────────────────────────

    def surface_epoch(self, surface: str) -> int:
        state = self.surfaces.get(surface)
        return int(state.epoch if state else 0)

    def surface_updated_ms(self, surface: str) -> int:
        state = self.surfaces.get(surface)
        return int(state.updated_ms if state else 0)

    def surfaces_missing_after(
        self, surfaces: set[str] | frozenset[str], min_epoch: int
    ) -> list[str]:
        """Which of ``surfaces`` have NOT refreshed at/after ``min_epoch``."""
        return sorted(
            surface
            for surface in surfaces
            if self.surface_epoch(surface) < int(min_epoch)
        )

    # ── symbol blocks ───────────────────────────────────────────────

    def flag_symbol_block(
        self,
        symbol: str,
        *,
        reason: str,
        required_surfaces: set[str] | frozenset[str],
        min_epoch: int,
        detected_ms: int,
        details: dict[str, Any] | None = None,
    ) -> SymbolBlock:
        """Block execution for ``symbol`` until ``required_surfaces`` all
        refresh at/after ``min_epoch``."""
        block = SymbolBlock(
            symbol=str(symbol),
            reason=str(reason),
            required_surfaces=frozenset(required_surfaces),
            min_epoch=int(min_epoch),
            detected_ms=int(detected_ms),
            details=dict(details or {}),
        )
        self.symbol_blocks[block.symbol] = block
        self._clear_satisfied_symbol_blocks()
        return block

    def is_blocked(self, symbol: str) -> bool:
        self._clear_satisfied_symbol_blocks()
        return str(symbol) in self.symbol_blocks

    def block_for(self, symbol: str) -> SymbolBlock | None:
        self._clear_satisfied_symbol_blocks()
        return self.symbol_blocks.get(str(symbol))

    def blocked_symbols(self) -> dict[str, SymbolBlock]:
        self._clear_satisfied_symbol_blocks()
        return dict(self.symbol_blocks)

    def clear_symbol(self, symbol: str) -> None:
        self.symbol_blocks.pop(str(symbol), None)

    def _clear_satisfied_symbol_blocks(self) -> None:
        if not self.symbol_blocks:
            return
        for symbol, block in list(self.symbol_blocks.items()):
            if not self.surfaces_missing_after(
                block.required_surfaces, block.min_epoch
            ):
                self.symbol_blocks.pop(symbol, None)
