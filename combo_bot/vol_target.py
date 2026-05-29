"""Portfolio-level volatility-targeting sizer.

Stage 11 adds a global throttle that scales all new entries by the
ratio of a configured target annualized volatility to the bot's
currently realized volatility. When the market gets choppy and the
strategy's equity curve gets noisier, sizes shrink. When markets calm
down, sizes grow back. The net effect is a more stable ex-ante risk
exposure — the foundation under reliable compounding for a
high-conviction strategy.

This is the same principle hedge funds use for "target vol" portfolios
(risk parity, e.g., AQR's). The bot didn't have it before because we
were sizing by Kelly (per-source edge) and correlation (cross-symbol
factor exposure); vol-targeting closes the loop at the *portfolio*
level — neither Kelly nor correlation see how violently the equity
curve itself moves.

Stacking order
--------------

Kelly throttles by *realized edge* (mean/var of per-trade returns).
Correlation throttles by *factor stacking* across symbols. Vol-target
throttles by *equity-curve volatility*. They compose cleanly:

  baseline_qty
    -> protections (binary lock)
    -> correlation gate (per-symbol factor scale)
    -> vol-target (portfolio-level scale)
    -> unstuck additions (reduce-only)
    -> risk filter (tier-based)

Cold start
----------

Below ``min_samples`` equity observations, :meth:`scale_factor` returns
1.0 — no scaling, preserving every existing test's exact behavior when
no sizer is wired in.

Edge cases
----------

* ``current_annual_vol() == 0`` (flat equity curve) → return 1.0
  rather than divide-by-zero. Once any movement appears, normal scaling
  takes over.
* Hard ``scale_min`` / ``scale_max`` clamps so a sudden volatility
  crash doesn't 100x positions, and a sudden spike doesn't drop us to
  zero overnight.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Iterable

from combo_bot.types import Order


# Total minutes per year for 1-minute bar annualization.
# 365 * 24 * 60 = 525,600.
_MINUTES_PER_YEAR = 525_600


@dataclass
class VolTargetSizerConfig:
    """Configuration for :class:`VolTargetSizer`.

    Defaults assume 1-minute equity samples (the Backtester tick rate)
    and a 30% annualized target — aggressive but reasonable for a
    high-leverage crypto bot. Adjust ``periods_per_year`` if you sample
    equity less frequently (e.g. 24*365=8760 for hourly, 365 for daily).
    """

    target_annual_vol: float = 0.30
    # Number of equity observations to keep. 1440 = ~1 day of 1m bars,
    # long enough to smooth intraday noise but short enough to react
    # to regime changes within a few hours.
    window: int = 1_440
    # Below this many observations the sizer returns 1.0 (no scaling).
    # ~1 hour of 1m bars is enough to compute a noisy but plausible
    # estimate.
    min_samples: int = 60
    # Annualization scale factor. The sizer multiplies per-period std
    # by sqrt(periods_per_year) to annualize.
    periods_per_year: int = _MINUTES_PER_YEAR
    # Bounds on the returned scale. Below scale_min, we still want some
    # exposure (otherwise drawdowns trap the bot at 0%); above
    # scale_max, we cap leverage growth in unrealistically calm
    # periods.
    scale_min: float = 0.1
    scale_max: float = 2.0


class VolTargetSizer:
    """Maintains a rolling equity-return history; returns a global qty
    scale factor that targets a configured annualized volatility."""

    def __init__(self, config: VolTargetSizerConfig | None = None) -> None:
        self.config = config or VolTargetSizerConfig()
        # +1 because we need N+1 equity points to produce N returns.
        self._equity: deque[float] = deque(maxlen=self.config.window + 1)

    def record_equity(self, equity: float) -> None:
        """Append the current account equity. Non-positive values are
        skipped — they typically indicate liquidation or pre-init state."""
        if equity > 0:
            self._equity.append(equity)

    def current_annual_vol(self) -> float | None:
        """Annualized realized volatility of the equity curve.

        Returns ``None`` when there's insufficient history to estimate
        — callers should treat that as "no information, no scaling".
        """
        if len(self._equity) < self.config.min_samples + 1:
            return None

        equity_list = list(self._equity)
        returns: list[float] = []
        for i in range(1, len(equity_list)):
            prev = equity_list[i - 1]
            if prev > 0:
                returns.append(equity_list[i] / prev - 1.0)

        if len(returns) < 2:
            return None

        mean = sum(returns) / len(returns)
        # Population variance — see KellySizer for the same rationale.
        var = sum((r - mean) ** 2 for r in returns) / len(returns)
        if var <= 0:
            return 0.0

        per_period_std = var ** 0.5
        return per_period_std * (self.config.periods_per_year ** 0.5)

    def scale_factor(self) -> float:
        """Sizing multiplier in [scale_min, scale_max].

        Cold start or zero-vol fallback both return 1.0 — i.e. trade at
        baseline size. Once enough history is available and vol is
        positive, return ``clamp(target / current, [min, max])``.
        """
        current = self.current_annual_vol()
        if current is None:
            return 1.0
        if current <= 1e-12:
            return 1.0
        raw_scale = self.config.target_annual_vol / current
        return max(
            self.config.scale_min,
            min(self.config.scale_max, raw_scale),
        )

    def filter_orders(self, orders: Iterable[Order]) -> list[Order]:
        """Apply the current scale to every new (non reduce-only) order.

        Reduce-only orders pass through untouched — vol-targeting is a
        new-exposure throttle, not a position-closing one.
        """
        scale = self.scale_factor()
        if scale == 1.0:
            # Fast path: no scaling, return the list as-is.
            return list(orders)

        result: list[Order] = []
        for order in orders:
            if order.reduce_only:
                result.append(order)
                continue
            new_qty = order.qty * scale
            if new_qty <= 0:
                continue
            result.append(replace(order, qty=new_qty))
        return result

    def sample_size(self) -> int:
        """Number of *return* samples currently available — one less
        than the equity buffer length."""
        return max(0, len(self._equity) - 1)


__all__ = ["VolTargetSizerConfig", "VolTargetSizer"]
