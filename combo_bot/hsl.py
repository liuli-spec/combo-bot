"""Hard Stop Loss supervisor — tier classification independent of enforcement.

Mirrors passivbot's ``equity_hard_stop_loss`` design:
  * Green / Yellow / Orange / Red tiers based on drawdown from equity peak.
  * EMA-smoothed drawdown score guards against flash spikes and stale recovery.
  * RED latch locks the tier until an explicit operator reset.
  * Cooldown after RED prevents immediate re-entry.

This module is PURE classification — it never touches orders, positions, or
exchange state. RiskManager consumes it to decide what enforcement actions to
apply.

Extracted from RiskManager (Stage 6+) so the tier decision can be tested,
persisted, and reasoned about independently of the enforcement machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from combo_bot.types import AccountState


class HslTier(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"


@dataclass
class HslConfig:
    """HSL tier thresholds and EMA smoothing parameters.

    ``yellow_threshold`` and ``orange_threshold`` are fractions of
    ``red_threshold`` in passivbot — they're absolute here for
    backward compatibility with the existing RiskConfig schema.
    """

    red_threshold: float = 0.25
    yellow_threshold: float = 0.10
    orange_threshold: float = 0.18
    dd_ema_span_minutes: float = 30.0
    red_latch_enabled: bool = True
    cooldown_after_red_minutes: int = 60
    # When > 0, the RED latch automatically releases after this many
    # MINUTES of wall-clock time since it was set. 0 = manual reset
    # only (passivbot semantic). Default 240 = 4h, leaning toward the
    # high-risk profile: after a deep drawdown the bot self-heals and
    # resumes trading instead of needing operator intervention.
    # Re-entry into RED still requires drawdown to cross the threshold
    # again, so a flapping market can't infinitely re-latch unless the
    # account is genuinely bleeding.
    red_latch_auto_release_minutes: int = 240


class HslSupervisor:
    """Drawdown-based tier classifier with EMA smoothing and RED latch.

    Usage::

        hsl = HslSupervisor(HslConfig(...))
        tier = hsl.assess(account, timestamp_ms)
        if tier == HslTier.RED:
            ...  # consumer decides what to do

    State fields (``tier``, ``red_latched``, ``red_cooldown_until``,
    ``dd_ema``) are public so they can be persisted/restored by the
    live trader's state file.
    """

    def __init__(self, config: HslConfig | None = None) -> None:
        self.config = config or HslConfig()
        self.tier: HslTier = HslTier.GREEN
        self.red_latched: bool = False
        # Wallclock ms when the latch was set — used for auto-release.
        # 0 means "no latch active" (or pre-auto-release feature).
        self.red_latched_at_ms: int = 0
        self.red_cooldown_until: int = 0
        self.dd_ema: float = 0.0
        self._last_assess_minute: int = 0
        self._dd_initialized: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(
        self, account: AccountState, timestamp_ms: int = 0
    ) -> HslTier:
        """Classify the current account drawdown and return the tier.

        ``timestamp_ms`` drives the EMA decay AND the RED latch
        auto-release. Pass 0 for single-shot use (tests, backtests
        with no real clock — in that case auto-release is effectively
        disabled because ``red_latched_at_ms`` won't have advanced).
        """
        # RED latch auto-release: clear the latch if wall-clock time
        # since it was set exceeds the configured window. This lets the
        # bot self-heal after a deep drawdown without operator action,
        # matching the high-risk-tolerance profile. The dd_ema check
        # below still gates re-entry: if drawdown is still above
        # red_threshold, the next classification will re-RED.
        if (
            self.red_latched
            and self.config.red_latch_auto_release_minutes > 0
            and self.red_latched_at_ms > 0
            and timestamp_ms > 0
        ):
            elapsed_min = (timestamp_ms - self.red_latched_at_ms) / 60_000.0
            if elapsed_min >= self.config.red_latch_auto_release_minutes:
                self.red_latched = False
                self.red_latched_at_ms = 0
                self.red_cooldown_until = 0

        raw = account.drawdown
        score = self._update_dd_score(raw, timestamp_ms)

        if score >= self.config.red_threshold:
            base = HslTier.RED
        elif score >= self.config.orange_threshold:
            base = HslTier.ORANGE
        elif score >= self.config.yellow_threshold:
            base = HslTier.YELLOW
        else:
            base = HslTier.GREEN

        if base == HslTier.RED and self.config.red_latch_enabled:
            # Set the wallclock marker ONLY on the latch transition —
            # subsequent RED-while-latched ticks don't reset the auto-
            # release countdown. Coerce a 0-timestamp to 1 so the
            # "has been set" check downstream still works for tests
            # that drive assess with timestamp_ms=0.
            if not self.red_latched:
                self.red_latched_at_ms = timestamp_ms if timestamp_ms > 0 else 1
            self.red_latched = True

        self.tier = HslTier.RED if self.red_latched else base
        return self.tier

    def reset_red_latch(self) -> None:
        """Clear the RED latch and any active cooldown.

        Intended for explicit operator action or tests. Re-entry into
        RED requires the drawdown threshold to be crossed again.
        """
        self.red_latched = False
        self.red_latched_at_ms = 0
        self.red_cooldown_until = 0
        self.tier = HslTier.GREEN

    def within_cooldown(self, timestamp_ms: int) -> bool:
        return timestamp_ms < self.red_cooldown_until

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_dd_score(self, raw: float, timestamp_ms: int) -> float:
        """Return the drawdown score used for tier classification.

        ``score = min(raw, ema)`` per passivbot's hard-stop design:
          * A single-tick flash spike (raw briefly > red) doesn't latch
            RED because ema is still low.
          * A stale EMA after recovery doesn't keep us in RED because
            raw has dropped below the threshold again.

        Disabled (returns raw) when ``dd_ema_span_minutes <= 0`` or on
        the very first call.
        """
        if self.config.dd_ema_span_minutes <= 0:
            return raw

        current_minute = timestamp_ms // 60_000
        if not self._dd_initialized:
            self.dd_ema = raw
            self._last_assess_minute = current_minute
            self._dd_initialized = True
            return raw

        elapsed = max(0, current_minute - self._last_assess_minute)
        if elapsed > 0:
            alpha = 2.0 / (self.config.dd_ema_span_minutes + 1.0)
            decay = (1.0 - alpha) ** elapsed
            self.dd_ema = raw + (self.dd_ema - raw) * decay
            self._last_assess_minute = current_minute

        return min(raw, self.dd_ema)


__all__ = ["HslTier", "HslConfig", "HslSupervisor"]
