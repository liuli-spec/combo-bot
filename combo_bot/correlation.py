"""Cross-symbol correlation gate for multi-symbol portfolios.

Stage 10 closes the last big gap in the multi-symbol story: when N
symbols all move together (e.g. BTC + ETH + SOL during a broad crypto
move), opening a same-side position on each one effectively multiplies
the bot's exposure to a single market factor. The gate reduces or
blocks new entries that *increase* same-factor exposure, while leaving
hedges (opposite-side entries on correlated symbols) untouched.

Sign convention — the "effective" correlation
---------------------------------------------

Raw Pearson correlation tells us how two symbols' returns move
together, but the *position* impact depends on which side each leg
takes. We compute an effective correlation that flips sign when the
two legs are on opposite sides:

    effective_corr = corr if same_side else -corr

That gives the right answer in every quadrant:

  * positive corr + same side  → effective +, penalize (factor stack);
  * positive corr + opp side   → effective -, ignore (hedge);
  * negative corr + same side  → effective -, ignore (diversifier);
  * negative corr + opp side   → effective +, penalize (still same factor).

Soft + hard thresholds
----------------------

Below ``soft_threshold``: no scaling, order passes unchanged.
Between ``soft_threshold`` and ``hard_threshold``: qty is linearly
interpolated from 1.0 down to 0.0. At or above ``hard_threshold``: the
order is dropped entirely. This gives a smooth degradation without a
"correlation cliff" that adversaries could game with marginal entries.

Reduce-only orders are never touched — closing existing exposure is
always allowed.

What we don't do (intentional)
------------------------------

* per-source (grid vs trend) correlation isolation — correlated factor
  exposure compounds the same way regardless of which bucket holds it,
  so the gate sums across both buckets when evaluating existing
  positions;
* directional adjustment per the entry's projected size — we scale qty
  but don't try to compute "net beta" exposure, because that requires
  asset-specific betas we don't have. The qty scale is a heuristic, not
  a proof.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Iterable

from combo_bot.types import AccountState, Order, Side


# ---------------------------------------------------------------------------
# CorrelationTracker
# ---------------------------------------------------------------------------


class CorrelationTracker:
    """Maintains a rolling close-price history per symbol and computes
    pairwise Pearson correlation of returns.

    The first close-price for a symbol seeds the buffer; from the second
    onward, simple returns ``c[i]/c[i-1] - 1`` populate the return
    series used for correlation.
    """

    def __init__(self, window: int = 100) -> None:
        self.window = max(2, int(window))
        self._closes: dict[str, deque[float]] = {}

    def update(self, symbol: str, close: float) -> None:
        if close <= 0:
            return
        buf = self._closes.get(symbol)
        if buf is None:
            buf = deque(maxlen=self.window)
            self._closes[symbol] = buf
        buf.append(close)

    def returns(self, symbol: str) -> list[float]:
        closes = list(self._closes.get(symbol, ()))
        if len(closes) < 2:
            return []
        out: list[float] = []
        for i in range(1, len(closes)):
            prev = closes[i - 1]
            if prev > 0:
                out.append(closes[i] / prev - 1.0)
        return out

    def sample_size(self, symbol: str) -> int:
        """Number of *return* observations available — one less than the
        number of stored closes."""
        return max(0, len(self._closes.get(symbol, ())) - 1)

    def correlation(self, a: str, b: str) -> float:
        """Pearson correlation of overlapping returns. Returns 0 when
        either symbol has fewer than 2 return samples or when either
        return series has zero variance (no movement to correlate)."""
        if a == b:
            return 1.0
        ra = self.returns(a)
        rb = self.returns(b)
        n = min(len(ra), len(rb))
        if n < 2:
            return 0.0
        # Align the trailing N returns so we compare over the same period.
        ra = ra[-n:]
        rb = rb[-n:]
        mean_a = sum(ra) / n
        mean_b = sum(rb) / n
        cov = sum(
            (ra[i] - mean_a) * (rb[i] - mean_b) for i in range(n)
        ) / n
        var_a = sum((r - mean_a) ** 2 for r in ra) / n
        var_b = sum((r - mean_b) ** 2 for r in rb) / n
        denom_sq = var_a * var_b
        if denom_sq <= 1e-24:
            return 0.0
        return cov / (denom_sq ** 0.5)


# ---------------------------------------------------------------------------
# CorrelationGate
# ---------------------------------------------------------------------------


@dataclass
class CorrelationGateConfig:
    """Tracker and gate parameters.

    ``window`` is the close-price history length. ``min_samples``
    requires at least N return observations for both symbols before the
    gate considers their correlation (otherwise the estimate is too
    noisy to act on). ``soft_threshold`` < ``hard_threshold`` < 1.0.
    """

    window: int = 60
    min_samples: int = 30
    soft_threshold: float = 0.6
    hard_threshold: float = 0.9


class CorrelationGate:
    """Scales or drops new entries that increase same-factor exposure.

    Typical usage by a host (e.g. :class:`Backtester`):

      1. On every tick, call :meth:`update_prices` for each symbol's
         latest close to keep the rolling correlation in sync.
      2. After all order-generating layers have produced ``all_orders``,
         call :meth:`filter_orders` to receive the post-gate list.
    """

    def __init__(self, config: CorrelationGateConfig | None = None) -> None:
        self.config = config or CorrelationGateConfig()
        self.tracker = CorrelationTracker(self.config.window)

    def update_prices(self, prices: Iterable[tuple[str, float]]) -> None:
        for symbol, close in prices:
            self.tracker.update(symbol, close)

    def filter_orders(
        self, orders: list[Order], account: AccountState,
    ) -> list[Order]:
        result: list[Order] = []
        for order in orders:
            if order.reduce_only:
                result.append(order)
                continue
            max_eff = self._max_effective_correlation(order, account)
            if max_eff >= self.config.hard_threshold:
                # Drop — adding here piles onto an already-overcrowded factor.
                continue
            if max_eff <= self.config.soft_threshold:
                result.append(order)
                continue
            # Linear ramp between soft and hard.
            span = self.config.hard_threshold - self.config.soft_threshold
            scale = (self.config.hard_threshold - max_eff) / max(span, 1e-12)
            new_qty = order.qty * scale
            if new_qty <= 0:
                continue
            result.append(replace(order, qty=new_qty))
        return result

    def _max_effective_correlation(
        self, order: Order, account: AccountState,
    ) -> float:
        cfg = self.config
        if self.tracker.sample_size(order.symbol) < cfg.min_samples:
            return 0.0

        max_eff = 0.0
        for sym, ss in account.symbols.items():
            if sym == order.symbol:
                continue
            if self.tracker.sample_size(sym) < cfg.min_samples:
                continue
            # Both grid and trend buckets count: same-factor exposure
            # compounds whichever bucket holds the position.
            for side, pos in (
                (Side.LONG, ss.position_long),
                (Side.LONG, ss.trend_long),
                (Side.SHORT, ss.position_short),
                (Side.SHORT, ss.trend_short),
            ):
                if not pos.is_open:
                    continue
                raw = self.tracker.correlation(order.symbol, sym)
                eff = raw if side == order.side else -raw
                if eff > max_eff:
                    max_eff = eff
        return max_eff


__all__ = [
    "CorrelationTracker",
    "CorrelationGateConfig",
    "CorrelationGate",
]
