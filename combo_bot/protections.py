"""freqtrade-inspired pluggable protections framework.

This module provides a layered safety system that sits between order
generation and the global risk manager. Each :class:`IProtection` is a
small, focused rule that returns transient :class:`ProtectionLock`
records when its condition fires; the :class:`ProtectionManager`
aggregates active locks and drops new entries that match them.

Design departures from freqtrade:

* freqtrade locks at (pair, side) granularity — we add ``source`` so a
  trend-overlay lock doesn't also pause the grid bucket on the same
  pair (matches the SourcedPosition isolation from Stage 3).
* freqtrade evaluates protections lazily on trade open/close events; we
  evaluate every tick with the fills produced that tick, because grid
  bots stream many fills per "trade".
* :class:`IProtection.evaluate` returns *new* locks rather than mutating
  manager state — keeps each rule trivially testable.

The two protections that ship in-tree (:class:`StoplossGuard`,
:class:`CooldownPeriod`) cover the two failure modes most likely to
compound losses in a grid+trend fusion: cascading SL hits and
re-entering immediately after a bad close.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

from combo_bot.types import AccountState, Fill, Order, OrderSource, Side


# ---------------------------------------------------------------------------
# Lock records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProtectionLock:
    """A single active block on new entries.

    ``None`` on any scope field means "no filter on that dimension":
    ``symbol=None`` locks every symbol; ``side=None`` locks both sides;
    ``source=None`` locks both grid and trend buckets.
    """

    until_ms: int
    reason: str
    symbol: str | None = None
    side: Side | None = None
    source: OrderSource | None = None

    def matches(self, symbol: str, side: Side, source: OrderSource) -> bool:
        if self.symbol is not None and self.symbol != symbol:
            return False
        if self.side is not None and self.side != side:
            return False
        if self.source is not None and self.source != source:
            return False
        return True


# ---------------------------------------------------------------------------
# Protection interface
# ---------------------------------------------------------------------------


class IProtection(ABC):
    """Base class for pluggable protection rules.

    A protection sees every fill that occurred since the last tick, the
    full account state, and the current timestamp. It returns zero or
    more :class:`ProtectionLock` records to add to the manager. Locks
    automatically expire when their ``until_ms`` passes — no explicit
    release is required.

    Concrete protections own their own history buffers so they can
    answer "how many losses in the last hour?" without re-scanning the
    fill stream from the manager.
    """

    @abstractmethod
    def evaluate(
        self,
        fills: list[Fill],
        account: AccountState,
        now_ms: int,
    ) -> list[ProtectionLock]:
        """Return new locks to register. Return ``[]`` if no rule fired."""


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class ProtectionManager:
    """Coordinates multiple :class:`IProtection` instances.

    Each tick:
      1. Caller invokes :meth:`update` with this tick's fills + state.
      2. Manager prunes expired locks, runs each protection, collects
         new locks.
      3. Caller invokes :meth:`filter_orders` to drop new entries that
         hit any active lock. Reduce-only orders always pass — the
         framework is about stopping new exposure, not stranding open
         positions.
    """

    def __init__(self, protections: Iterable[IProtection] = ()) -> None:
        self.protections: list[IProtection] = list(protections)
        self.locks: list[ProtectionLock] = []

    def update(
        self,
        new_fills: list[Fill],
        account: AccountState,
        now_ms: int,
    ) -> None:
        # Prune expired locks first so re-evaluation never sees them.
        self.locks = [lock for lock in self.locks if lock.until_ms > now_ms]
        for protection in self.protections:
            new_locks = protection.evaluate(new_fills, account, now_ms)
            if new_locks:
                self.locks.extend(new_locks)

    def is_locked(
        self,
        symbol: str,
        side: Side,
        source: OrderSource,
        now_ms: int,
    ) -> bool:
        for lock in self.locks:
            if lock.until_ms <= now_ms:
                continue
            if lock.matches(symbol, side, source):
                return True
        return False

    def filter_orders(self, orders: list[Order], now_ms: int) -> list[Order]:
        """Drop new (non-reduce-only) entries that match an active lock."""
        result: list[Order] = []
        for order in orders:
            if order.reduce_only:
                result.append(order)
                continue
            if self.is_locked(order.symbol, order.side, order.source, now_ms):
                continue
            result.append(order)
        return result


# ---------------------------------------------------------------------------
# StoplossGuard
# ---------------------------------------------------------------------------


@dataclass
class StoplossGuardConfig:
    """Configuration for :class:`StoplossGuard`.

    Defaults are tight enough to be useful for a high-leverage bot
    without being so eager that any losing trade trips them. ``required_profit_pct``
    treats fills with realized P&L *below* this absolute USD value as
    "stoplosses" — set to 0 to count any losing close. Setting negative
    values lets you ignore small breakeven-ish closes.
    """

    lookback_period_ms: int = 60 * 60 * 1_000  # 1 hour
    trade_limit: int = 4
    stop_duration_ms: int = 30 * 60 * 1_000  # 30 minutes
    only_per_pair: bool = False
    only_per_side: bool = False
    only_per_source: bool = False
    required_profit_pct: float = 0.0


class StoplossGuard(IProtection):
    """Lock when N losing fills cluster within a lookback window.

    Mirrors freqtrade's ``StoplossGuard``. The granularity of the lock
    is controlled by the ``only_per_*`` flags: leave them all False for
    a single global lock that pauses the whole bot, or enable any
    combination to scope locks more tightly.
    """

    def __init__(self, config: StoplossGuardConfig | None = None) -> None:
        self.config = config or StoplossGuardConfig()
        self._losses: deque[Fill] = deque()

    def evaluate(
        self,
        fills: list[Fill],
        account: AccountState,
        now_ms: int,
    ) -> list[ProtectionLock]:
        for fill in fills:
            # Compare NET PnL (after fees) against threshold. A close
            # at +$0.10 gross with $0.50 in fees is a real loss that
            # would otherwise slip past a 0-USD threshold. Skip pure
            # opens (gross == 0 AND fee >= 0, i.e. net <= 0 but no
            # close action) by also requiring gross != 0.
            net_pnl = fill.realized_pnl - fill.fee
            if fill.realized_pnl != 0 and net_pnl < self.config.required_profit_pct:
                self._losses.append(fill)

        # Prune the deque to the active window.
        cutoff = now_ms - self.config.lookback_period_ms
        while self._losses and self._losses[0].timestamp < cutoff:
            self._losses.popleft()

        # Group surviving losses by the configured scope and emit one
        # lock per group whose size hits the threshold.
        groups: dict[tuple[str | None, Side | None, OrderSource | None], int] = {}
        for fill in self._losses:
            key = (
                fill.symbol if self.config.only_per_pair else None,
                fill.side if self.config.only_per_side else None,
                fill.source if self.config.only_per_source else None,
            )
            groups[key] = groups.get(key, 0) + 1

        locks: list[ProtectionLock] = []
        for (symbol, side, source), count in groups.items():
            if count >= self.config.trade_limit:
                locks.append(ProtectionLock(
                    until_ms=now_ms + self.config.stop_duration_ms,
                    reason=f"stoploss_guard:{count}_losses",
                    symbol=symbol,
                    side=side,
                    source=source,
                ))
        return locks


# ---------------------------------------------------------------------------
# CooldownPeriod
# ---------------------------------------------------------------------------


@dataclass
class CooldownPeriodConfig:
    """Configuration for :class:`CooldownPeriod`.

    For grid bots, you almost always want ``required_profit_pct < 0``
    so winning TPs don't trigger a cooldown (otherwise the bot just
    stops itself after every grid take-profit). The default counts only
    real losses (realized_pnl < 0).
    """

    stop_duration_ms: int = 10 * 60 * 1_000  # 10 minutes
    only_per_pair: bool = True
    only_per_side: bool = True
    only_per_source: bool = True
    required_profit_pct: float = 0.0


class CooldownPeriod(IProtection):
    """Lock a (symbol, side, source) tuple briefly after each loss.

    Maps to freqtrade's ``CooldownPeriod`` but defaults to per-pair /
    per-side / per-source granularity, because in a fusion bot you
    rarely want a grid TP on BTC to lock SOL too.
    """

    def __init__(self, config: CooldownPeriodConfig | None = None) -> None:
        self.config = config or CooldownPeriodConfig()

    def evaluate(
        self,
        fills: list[Fill],
        account: AccountState,
        now_ms: int,
    ) -> list[ProtectionLock]:
        locks: list[ProtectionLock] = []
        for fill in fills:
            # Only closing fills produce non-zero gross PnL — opens have
            # realized_pnl == 0. Compare NET (after fees) so a fill that
            # eked out a tiny gross but lost to fees still cools down.
            if fill.realized_pnl == 0:
                continue
            net_pnl = fill.realized_pnl - fill.fee
            if net_pnl >= self.config.required_profit_pct:
                continue
            locks.append(ProtectionLock(
                until_ms=now_ms + self.config.stop_duration_ms,
                reason="cooldown_after_loss",
                symbol=fill.symbol if self.config.only_per_pair else None,
                side=fill.side if self.config.only_per_side else None,
                source=fill.source if self.config.only_per_source else None,
            ))
        return locks


__all__ = [
    "ProtectionLock",
    "IProtection",
    "ProtectionManager",
    "StoplossGuardConfig",
    "StoplossGuard",
    "CooldownPeriodConfig",
    "CooldownPeriod",
]
