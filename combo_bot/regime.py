"""Regime arbiter — synthesizes trend + strategy + funding into a per-tick view.

This is the central authority over per-side grid mode and trend-overlay
activation. Backtester and LiveTrader consume the resulting :class:`RegimeView`
instead of computing modes themselves, so backtest and live paths agree on
the same decision surface.

Defaults are tuned for a high-risk / high-reward profile:
  - STRONG_BULL / STRONG_BEAR forcibly PANIC-close the opposite-side grid
    rather than waiting for TP. The reasoning is that being long in a strong
    bear (or short in a strong bull) is the canonical "stuck position"
    scenario where bleeding for hours hurts more than crystallizing the loss.
  - The favored side promotes its grid to AGGRESSIVE so entries stack faster
    and closes fire sooner.
  - A trend overlay is activated on the favored side once conviction crosses
    the overlay threshold, subject to a funding-rate veto so we don't
    overpay on lopsided perpetual funding.

Pass a custom ``RegimeArbiterConfig`` to tune (or neuter) any of this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from combo_bot.types import RegimeView, Side, TradingMode, TrendRegime, TrendSignal

if TYPE_CHECKING:
    from combo_bot.data_provider import DataProvider


@dataclass
class RegimeArbiterConfig:
    # Signal strength above which the favored-side grid switches to AGGRESSIVE.
    aggressive_strength: float = 0.4
    # Signal strength above which the trend overlay is activated.
    overlay_strength: float = 0.6
    # In STRONG_BULL/BEAR, PANIC-close the opposite side. Set False to fall
    # back to TP_ONLY (passivbot-style — wait for the position to recover).
    panic_close_opposite_on_strong: bool = True
    # Close-markup compression. <1.0 means close earlier; 1.0 means engine default.
    bull_bear_close_aggressiveness: float = 0.85
    strong_close_aggressiveness: float = 0.7
    # |funding_rate| above this vetoes a fresh trend overlay in the direction
    # that would PAY funding (so we don't entry where the carry is against us).
    funding_overlay_veto_pct: float = 0.0008
    # Multiplier on conviction when sizing the trend overlay first entry.
    overlay_sizing_scale: float = 1.5


class RegimeArbiter:
    def __init__(self, config: RegimeArbiterConfig | None = None) -> None:
        self.config = config or RegimeArbiterConfig()

    def compute(
        self,
        signal: TrendSignal,
        funding_rate: float = 0.0,
        strategy_exit_long: bool = False,
        strategy_exit_short: bool = False,
        strategy_enter_long: bool = False,
        strategy_enter_short: bool = False,
    ) -> RegimeView:
        cfg = self.config
        regime = signal.regime
        conviction = signal.strength

        long_mode = TradingMode.NORMAL
        short_mode = TradingMode.NORMAL
        allow_long = True
        allow_short = True
        overlay: Side | None = None
        overlay_scale = 0.0
        close_aggr = 1.0
        veto: list[str] = []

        long_overlay_blocked = funding_rate > cfg.funding_overlay_veto_pct
        short_overlay_blocked = funding_rate < -cfg.funding_overlay_veto_pct

        if regime == TrendRegime.STRONG_BULL:
            if cfg.panic_close_opposite_on_strong:
                short_mode = TradingMode.PANIC
                allow_short = False
            else:
                short_mode = TradingMode.TP_ONLY
            if conviction >= cfg.aggressive_strength:
                long_mode = TradingMode.AGGRESSIVE
            close_aggr = cfg.strong_close_aggressiveness
            if conviction >= cfg.overlay_strength:
                if long_overlay_blocked:
                    veto.append(f"funding={funding_rate:.4f} vetoes long overlay")
                else:
                    overlay = Side.LONG
                    overlay_scale = min(1.0, conviction * cfg.overlay_sizing_scale)

        elif regime == TrendRegime.STRONG_BEAR:
            if cfg.panic_close_opposite_on_strong:
                long_mode = TradingMode.PANIC
                allow_long = False
            else:
                long_mode = TradingMode.TP_ONLY
            if conviction >= cfg.aggressive_strength:
                short_mode = TradingMode.AGGRESSIVE
            close_aggr = cfg.strong_close_aggressiveness
            if conviction >= cfg.overlay_strength:
                if short_overlay_blocked:
                    veto.append(f"funding={funding_rate:.4f} vetoes short overlay")
                else:
                    overlay = Side.SHORT
                    overlay_scale = min(1.0, conviction * cfg.overlay_sizing_scale)

        elif regime == TrendRegime.BULL:
            short_mode = TradingMode.TP_ONLY
            close_aggr = cfg.bull_bear_close_aggressiveness

        elif regime == TrendRegime.BEAR:
            long_mode = TradingMode.TP_ONLY
            close_aggr = cfg.bull_bear_close_aggressiveness

        # NEUTRAL keeps all defaults.

        # Strategy exit signals are the final word — they downgrade modes
        # to TP_ONLY (unless something stricter is already set) and drop the
        # trend overlay on the matching side.
        if strategy_exit_long and long_mode not in (TradingMode.PANIC, TradingMode.GRACEFUL_STOP):
            long_mode = TradingMode.TP_ONLY
            veto.append("strategy exit_long")
        if strategy_exit_short and short_mode not in (TradingMode.PANIC, TradingMode.GRACEFUL_STOP):
            short_mode = TradingMode.TP_ONLY
            veto.append("strategy exit_short")
        if strategy_exit_long and overlay == Side.LONG:
            overlay = None
            overlay_scale = 0.0
        if strategy_exit_short and overlay == Side.SHORT:
            overlay = None
            overlay_scale = 0.0

        # Strategy entry signals are symmetric to exits: they PROMOTE the
        # matching side's mode toward AGGRESSIVE and force-activate the
        # trend overlay if it wasn't already on. Exit signals on the same
        # side take precedence — if the strategy says both enter_long
        # and exit_long, exit wins (we already downgraded to TP_ONLY
        # above; the promotion below skips that case). Defensive modes
        # (PANIC / GRACEFUL_STOP) also override — strategy entry must
        # not punch through a risk-driven graceful stop. Funding veto
        # still applies to the overlay activation.
        def _promote(
            mode: TradingMode, side: Side, overlay_blocked: bool,
            current_overlay: Side | None, current_scale: float,
        ) -> tuple[TradingMode, Side | None, float]:
            if mode in (
                TradingMode.PANIC,
                TradingMode.GRACEFUL_STOP,
                TradingMode.TP_ONLY,
            ):
                return mode, current_overlay, current_scale
            new_mode = TradingMode.AGGRESSIVE
            if current_overlay is None and not overlay_blocked:
                # Force overlay activation with a conservative scale —
                # the strategy committed to a direction, but we still
                # want size discipline. Use the lesser of conviction
                # and a 0.5 floor so a cold-start NEUTRAL regime can
                # also benefit.
                new_overlay = side
                new_scale = max(current_scale, 0.5)
                return new_mode, new_overlay, new_scale
            return new_mode, current_overlay, current_scale

        if strategy_enter_long and not strategy_exit_long:
            long_mode, overlay, overlay_scale = _promote(
                long_mode, Side.LONG, long_overlay_blocked, overlay, overlay_scale,
            )
            veto.append("strategy enter_long → AGGRESSIVE")
        if strategy_enter_short and not strategy_exit_short:
            short_mode, overlay, overlay_scale = _promote(
                short_mode, Side.SHORT, short_overlay_blocked, overlay, overlay_scale,
            )
            veto.append("strategy enter_short → AGGRESSIVE")

        return RegimeView(
            primary=regime,
            conviction=conviction,
            long_mode=long_mode,
            short_mode=short_mode,
            allow_grid_long=allow_long,
            allow_grid_short=allow_short,
            trend_overlay=overlay,
            trend_qty_scale=overlay_scale,
            close_aggressiveness=close_aggr,
            veto_reasons=tuple(veto),
        )


def read_strategy_signals(
    dp: "DataProvider", symbol: str
) -> tuple[bool, bool, bool, bool]:
    """Read the latest row's ``enter_long / enter_short / exit_long / exit_short``
    flags from the strategy-augmented DataFrame.

    Uses freqtrade's convention: a missing column or a value other than 1
    means "no signal" (returns False). Safe to call even if the strategy
    never wrote any signal columns.
    """
    df = dp.get_dataframe(symbol)
    if len(df) == 0:
        return (False, False, False, False)
    row = df.iloc[-1]

    def _is_set(col: str) -> bool:
        if col not in df.columns:
            return False
        v = row.get(col, 0)
        try:
            return float(v) == 1.0
        except (TypeError, ValueError):
            return False

    return (
        _is_set("enter_long"),
        _is_set("enter_short"),
        _is_set("exit_long"),
        _is_set("exit_short"),
    )


__all__ = ["RegimeArbiter", "RegimeArbiterConfig", "read_strategy_signals"]
