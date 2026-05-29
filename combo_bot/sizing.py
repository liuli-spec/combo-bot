"""Dynamic sizing via fractional Kelly on per-source realized returns.

Stage 9 converts the per-bucket P&L history that Stage 4 already
records into a live feedback signal: the trend overlay's qty is scaled
by a fractional Kelly fraction computed from recent returns. When the
overlay is winning consistently with low variance, its size goes up;
when it's losing or noisy, size shrinks; when edge is negative the
fraction clamps to 0 (and the overlay effectively pauses itself until
the empirical edge recovers).

Math
----

Continuous Kelly for a strategy with per-trade return ``R``:

    f* ≈ E[R] / Var[R]

This maximizes log-growth and is the standard Kelly-criterion
approximation for small returns. We then multiply by a configurable
``fractional_kelly`` (typically 0.25–0.5) because:

  * the empirical estimate of E[R]/Var[R] has noise that grows with
    smaller samples — full Kelly under uncertain edges leads to
    excessive drawdowns;
  * a high-leverage perpetual-futures bot already amplifies
    bet-sizing through margin, so the multiplier should sit below the
    theoretical optimum.

Cold start
----------

Until at least ``min_samples`` closing fills have been recorded for a
source, :meth:`KellySizer.fraction` returns 1.0 — i.e. no scaling. This
keeps the bot trading at its baseline configured size while it
accumulates the history needed to estimate edge.

Negative edge
-------------

If mean return is negative, Kelly says don't bet — :meth:`fraction`
returns 0.0 and the overlay sizing collapses to zero. The overlay
re-activates automatically once the rolling window's mean turns
positive again. This stacks cleanly with the Stage 4 per-source
drawdown breaker: Kelly responds to *return distribution* while the
breaker responds to *cumulative drawdown*. Either can shut the overlay
off; both must agree before the overlay re-opens.

Notes on what gets measured
---------------------------

Each closing :class:`Fill` contributes one sample equal to
``realized_pnl / (qty * fill_price)`` — return on the closed notional.
Entry fills (realized_pnl == 0) are intentionally ignored: they're
just exposure changes, not realized P&L. Per-symbol attribution
isn't done here; we pool returns across symbols within a source bucket
because the overlay scaling is global to the source (Stage 9 doesn't
do per-symbol sizing — that's a Stage 10 candidate).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

from combo_bot.types import Fill, OrderSource


@dataclass
class KellySizerConfig:
    """Configuration for :class:`KellySizer`.

    ``min_samples`` is intentionally generous — a tiny window
    overfits to recent noise and produces large swings in the
    fraction. ``fractional_kelly`` is the standard quarter-Kelly
    default.
    """

    window: int = 100
    min_samples: int = 20
    fractional_kelly: float = 0.25
    # Hard cap on the returned fraction. Even if Kelly says 5x, we
    # cap at 1x because the overlay sizing has its own budget limits
    # in MergerConfig — we only want Kelly to *throttle*, not amplify.
    max_fraction: float = 1.0


class KellySizer:
    """Per-source rolling-Kelly sizer.

    Construct with no args for sensible defaults, or pass a
    :class:`KellySizerConfig`. Feed closing fills via
    :meth:`record_fill`. Query the current sizing multiplier with
    :meth:`fraction`.

    Sample tracking is per-source so the trend overlay's Kelly doesn't
    get polluted by grid TPs (which have a very different return
    distribution). The grid bucket gets tracked too even though Stage 9
    only consumes the trend fraction — the data is useful for
    diagnostics, and a future stage can hook it.
    """

    def __init__(self, config: KellySizerConfig | None = None) -> None:
        self.config = config or KellySizerConfig()
        self._returns: dict[OrderSource, deque[float]] = {
            OrderSource.GRID: deque(maxlen=self.config.window),
            OrderSource.TREND: deque(maxlen=self.config.window),
        }

    def record_fill(self, fill: Fill) -> None:
        """Capture the per-fill return for any closing fill.

        Entry fills (``realized_pnl == 0``) are skipped. RISK source
        fills route to the grid bucket so they don't pollute the trend
        return distribution — matches the fill-routing rule from
        Stage 3 (RISK closes the grid bucket).
        """
        # Use net PnL (after fees) so Kelly estimates the real edge
        # the strategy actually captures, not the pre-fee gross.
        net_pnl = fill.realized_pnl - fill.fee
        if net_pnl == 0:
            return
        notional = abs(fill.qty * fill.price)
        if notional <= 0:
            return
        return_pct = net_pnl / notional
        bucket = (
            OrderSource.TREND
            if fill.source == OrderSource.TREND
            else OrderSource.GRID
        )
        self._returns[bucket].append(return_pct)

    def record_fills(self, fills: Iterable[Fill]) -> None:
        for fill in fills:
            self.record_fill(fill)

    def fraction(self, source: OrderSource) -> float:
        """Current sizing multiplier in [0, max_fraction]."""
        bucket = (
            OrderSource.TREND
            if source == OrderSource.TREND
            else OrderSource.GRID
        )
        returns = self._returns[bucket]
        n = len(returns)
        if n < self.config.min_samples:
            # Cold start — trade at baseline sizing until we have data.
            return 1.0

        mean = sum(returns) / n
        # Population variance — we're using the whole sample as the
        # estimator, not as a draw from a larger population.
        var = sum((r - mean) ** 2 for r in returns) / n

        if var <= 1e-18:
            # Degenerate case: all returns identical. Sign of mean
            # determines whether we trade at all.
            return self.config.max_fraction if mean > 0 else 0.0

        full_kelly = mean / var
        scaled = full_kelly * self.config.fractional_kelly
        return max(0.0, min(self.config.max_fraction, scaled))

    def sample_size(self, source: OrderSource) -> int:
        """Number of return samples currently recorded for ``source``.

        Useful for surfacing "cold start vs warm" to dashboards.
        """
        bucket = (
            OrderSource.TREND
            if source == OrderSource.TREND
            else OrderSource.GRID
        )
        return len(self._returns[bucket])


__all__ = ["KellySizerConfig", "KellySizer"]
