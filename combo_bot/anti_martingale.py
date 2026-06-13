"""Anti-martingale (pyramid-into-winners) leveraged trend engine.

This is the *mathematically sound* inverse of the grid/martingale engine that
lives in ``grid_engine.py``. Where a martingale **adds into losers** (raising
effective leverage exactly when the position is going against you — the
anti-Kelly direction), this engine:

  * stays **flat** while price is below a slow trend filter (no counter-trend
    bag-holding),
  * opens a base-leverage long when price reclaims the trend,
  * **pyramids INTO winners** — exposure ramps up only as the move proves
    itself (Kelly-aligned: bet more once the edge confirms), capped at
    ``l_max``,
  * **reduces / exits** on a trailing stop or a trend break (cuts the left
    tail instead of feeding it),
  * scales the whole thing by a realized-vol target so risk ≈ constant, and
  * models leverage honestly: fees on turnover, a daily carry cost for the
    borrowed notional, and single-bar **liquidation** when an adverse move
    exceeds ``1 / leverage``.

It is intentionally self-contained and transparent — every assumption is a
named constant so a backtest number can be traced to a rule, not a black box.
The point is not to promise high returns; it is to show, on real data, what a
*positive-expectancy* leveraged auto-sizing engine actually yields (and where
it still bleeds), in contrast to the martingale fantasy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --- carry / cost defaults (per-day, perpetual-long realistic order of magnitude) ---
DEFAULT_FEE_RATE = 0.0005  # 5 bps taker, charged on leverage turnover
DEFAULT_CARRY_DAILY = 0.0003  # ~11%/yr financing on borrowed (leverage>1) notional
TRADING_DAYS = 365.0  # crypto trades every day


@dataclass(frozen=True)
class AntiMartingaleConfig:
    """All knobs for the anti-martingale engine. Frozen — configs are values."""

    # Trend filter
    trend_sma: int = 100  # bars; stay flat below this SMA
    # Pyramid (add-into-winners) ramp
    l_base: float = 1.0  # leverage on fresh trend entry
    l_max: float = 3.0  # leverage cap once the move has fully confirmed
    pyramid_full: float = 0.40  # fractional advance from entry at which L hits l_max
    # Exit / risk
    trail_stop: float = 0.20  # exit to flat if price falls this far from peak-since-entry
    # Volatility targeting
    vol_target_annual: float = 0.60  # target annualized vol of the *position* exposure
    vol_lookback: int = 30  # bars for realized-vol estimate
    vol_scale_cap: float = 1.5  # max multiplier vol-targeting may apply
    use_vol_target: bool = True
    # Costs
    fee_rate: float = DEFAULT_FEE_RATE
    carry_daily: float = DEFAULT_CARRY_DAILY
    # Liquidation: adverse single-bar return beyond -(1/L)*(1-buffer) wipes equity
    maint_buffer: float = 0.005  # 0.5% maintenance buffer


@dataclass
class AntiMartingaleResult:
    equity_curve: np.ndarray
    leverage_curve: np.ndarray
    final_mult: float
    cagr: float
    max_drawdown: float
    sharpe: float
    liquidated: bool
    liq_index: int | None
    n_years: float
    turnover: float
    meta: dict = field(default_factory=dict)


def _realized_vol_annual(returns: np.ndarray, i: int, lookback: int) -> float:
    lo = max(1, i - lookback)
    window = returns[lo:i]
    if window.size < 2:
        return 0.0
    return float(np.std(window) * np.sqrt(TRADING_DAYS))


def backtest_anti_martingale(
    closes: np.ndarray,
    lows: np.ndarray,
    cfg: AntiMartingaleConfig,
) -> AntiMartingaleResult:
    """Run the engine over a price path. ``closes``/``lows`` are bar arrays.

    Uses a rebalanced-leverage model: each bar we compute a target leverage
    ``L_t`` from trend/pyramid/trail/vol rules, then apply the bar's realized
    return at that leverage, netting fees on the change in leverage and a daily
    carry on the borrowed portion. Intrabar liquidation is checked against the
    bar low so a violent wick can still wipe the account — no look-ahead.
    """
    n = closes.size
    if n < cfg.trend_sma + 2:
        raise ValueError("not enough bars for the configured trend SMA")

    rets = np.zeros(n)
    rets[1:] = closes[1:] / closes[:-1] - 1.0
    # low-relative return within the bar (worst point), for liquidation checks
    low_rets = np.zeros(n)
    low_rets[1:] = lows[1:] / closes[:-1] - 1.0

    sma = np.full(n, np.nan)
    csum = np.cumsum(closes)
    w = cfg.trend_sma
    sma[w - 1 :] = (csum[w - 1 :] - np.concatenate(([0.0], csum[:-w]))) / w

    equity = np.ones(n)
    lev = np.zeros(n)
    in_trend = False
    entry_price = 0.0
    peak_price = 0.0
    prev_L = 0.0
    turnover = 0.0
    liquidated = False
    liq_index: int | None = None

    for i in range(1, n):
        price = closes[i - 1]  # decide using info available at close of prior bar
        s = sma[i - 1]

        # --- determine target leverage from the rules (all causal) ---
        L = 0.0
        if not np.isnan(s) and price > s:
            if not in_trend:
                in_trend = True
                entry_price = price
                peak_price = price
            peak_price = max(peak_price, price)
            if price <= peak_price * (1.0 - cfg.trail_stop):
                L = 0.0  # trailing stop hit -> flat, wait for re-entry
                in_trend = False
            else:
                advance = price / entry_price - 1.0
                pyr = float(np.clip(advance / cfg.pyramid_full, 0.0, 1.0))
                L = cfg.l_base + (cfg.l_max - cfg.l_base) * pyr
        else:
            in_trend = False

        if cfg.use_vol_target and L > 0:
            rv = _realized_vol_annual(rets, i, cfg.vol_lookback)
            if rv > 1e-9:
                scale = min(cfg.vol_target_annual / rv, cfg.vol_scale_cap)
                L *= scale
        L = float(np.clip(L, 0.0, cfg.l_max))
        lev[i] = L

        # --- liquidation check against the bar's low (no look-ahead) ---
        if L > 0 and low_rets[i] <= -(1.0 / L) * (1.0 - cfg.maint_buffer):
            equity[i:] = 0.0
            liquidated = True
            liq_index = i
            break

        # --- apply the bar ---
        gross = 1.0 + L * rets[i]
        turn = abs(L - prev_L)
        turnover += turn
        fee = cfg.fee_rate * turn
        carry = cfg.carry_daily * max(L - 1.0, 0.0)
        equity[i] = equity[i - 1] * gross * (1.0 - fee) * (1.0 - carry)
        prev_L = L

    n_years = n / TRADING_DAYS
    final = float(equity[-1])
    if final > 0:
        peak = np.maximum.accumulate(equity)
        dd = float(((equity - peak) / peak).min())
        cagr = final ** (1.0 / n_years) - 1.0
        eq_r = equity[1:] / equity[:-1] - 1.0
        eq_r = eq_r[np.isfinite(eq_r)]
        sharpe = (
            float(np.mean(eq_r) / np.std(eq_r) * np.sqrt(TRADING_DAYS))
            if eq_r.size > 2 and np.std(eq_r) > 0
            else 0.0
        )
    else:
        dd, cagr, sharpe = -1.0, -1.0, 0.0

    return AntiMartingaleResult(
        equity_curve=equity,
        leverage_curve=lev,
        final_mult=final,
        cagr=cagr,
        max_drawdown=dd,
        sharpe=sharpe,
        liquidated=liquidated,
        liq_index=liq_index,
        n_years=n_years,
        turnover=turnover,
        meta={"mean_leverage": float(np.mean(lev)), "time_in_market": float(np.mean(lev > 0))},
    )
