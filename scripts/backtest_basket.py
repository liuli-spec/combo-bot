"""Portfolio anti-martingale: run the engine per asset, combine equal-weight.

This is where leveraged trend-following legitimately earns its keep vs holding:
across a basket, the trend filter keeps capital in the assets that are working
and flat in the ones that aren't, smoothing portfolio vol so leverage can
compound instead of getting liquidated.

Usage: python scripts/backtest_basket.py <dir_with_*USDT-1h.feather>
"""

import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from combo_bot.anti_martingale import (  # noqa: E402
    AntiMartingaleConfig,
    backtest_anti_martingale,
    DEFAULT_FEE_RATE,
    DEFAULT_CARRY_DAILY,
    TRADING_DAYS,
)


def load_daily_closes(path):
    df = pd.read_feather(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    d = df.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    return d


def metrics(eq, years):
    final = float(eq[-1])
    if final <= 0:
        return final, -1.0, -1.0, 0.0
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min())
    cagr = final ** (1 / years) - 1
    r = eq[1:] / eq[:-1] - 1
    r = r[np.isfinite(r)]
    sh = float(np.mean(r) / np.std(r) * np.sqrt(TRADING_DAYS)) if r.size > 2 and np.std(r) > 0 else 0.0
    return final, cagr, dd, sh


def line(name, eq, years, liq=None):
    final, cagr, dd, sh = metrics(eq, years)
    fin = f"{final:,.2f}x" if final > 0 else "0 LIQ"
    flag = "" if liq is None else f"  💀@bar{liq}"
    return f"{name:<40} {fin:>14} {cagr*100:>8.1f}% {dd*100:>8.0f}% {sh:>7.2f}{flag}"


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/freqtrade/user_data/data/binance"
    files = sorted(glob(f"{d}/*USDT-1h.feather"))
    series = {Path(f).stem.split("_")[0]: load_daily_closes(f) for f in files}

    # align on the common date index so portfolio weights are well-defined
    common = None
    for s in series.values():
        common = s.index if common is None else common.intersection(s.index)
    common = common.sort_values()
    syms = list(series)
    closes = {k: series[k].reindex(common)["close"].to_numpy() for k in syms}
    lows = {k: series[k].reindex(common)["low"].to_numpy() for k in syms}
    years = len(common) / TRADING_DAYS
    print(f"\nBasket: {len(syms)} assets  {common[0].date()} -> {common[-1].date()}  ({years:.2f}y)")
    print(f"  {', '.join(syms)}\n")

    cfg = AntiMartingaleConfig(l_max=3.0, trail_stop=0.20)

    # per-asset leverage curves (causal, equity-independent) + daily returns
    levs, rets = {}, {}
    for k in syms:
        r = backtest_anti_martingale(closes[k], lows[k], cfg)
        levs[k] = r.leverage_curve
        rr = np.zeros(len(closes[k])); rr[1:] = closes[k][1:] / closes[k][:-1] - 1
        rets[k] = rr

    n = len(common); N = len(syms)
    # --- equal-weight buy&hold basket (L=1) ---
    bh = np.ones(n)
    for i in range(1, n):
        port = np.mean([rets[k][i] for k in syms])
        bh[i] = bh[i - 1] * (1 + port)

    # --- portfolio anti-martingale: equal capital, each levered by its own signal ---
    pf = np.ones(n); liq = None
    for i in range(1, n):
        port = np.mean([levs[k][i] * rets[k][i] for k in syms])
        gross_lev = np.mean([levs[k][i] for k in syms])
        turn = np.mean([abs(levs[k][i] - levs[k][i - 1]) for k in syms])
        port = port - DEFAULT_FEE_RATE * turn - DEFAULT_CARRY_DAILY * max(gross_lev - 1, 0)
        if port <= -1.0:
            pf[i:] = 0.0; liq = i; break
        pf[i] = pf[i - 1] * (1 + port)

    # baselines
    btc = series.get("BTC")
    print(f"{'strategy':<40} {'final':>14} {'CAGR':>9} {'maxDD':>9} {'Sharpe':>7}")
    print("-" * 90)
    if btc is not None:
        bc = btc.reindex(common)["close"].to_numpy()
        bhb = np.ones(n)
        for i in range(1, n):
            bhb[i] = bhb[i - 1] * (bc[i] / bc[i - 1]) if not np.isnan(bc[i]) and not np.isnan(bc[i-1]) else bhb[i-1]
        print(line("BTC buy & hold (L=1)", bhb, years))
    print(line("equal-weight basket buy & hold (L=1)", bh, years))
    print("-" * 90)
    print(line("PORTFOLIO anti-mart (Lmax=3, equal risk)", pf, years, liq))
    print(f"   mean gross leverage = {np.mean([np.mean(levs[k]) for k in syms]):.2f}")
    print()


if __name__ == "__main__":
    main()
