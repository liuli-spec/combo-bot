"""Backtest the anti-martingale engine on real data, against honest baselines.

Usage:
    python scripts/backtest_anti_martingale.py <feather_or_parquet> [TF]

Loads an OHLCV file (freqtrade feather or parquet), resamples to daily, and
compares:
  * buy & hold (L=1)
  * naive rebalanced leverage (L=2, L=3) -- the "just crank it" path
  * a martingale baseline (add into losses) -- the path that blows up
  * the anti-martingale engine (pyramid into winners) at a few configs
Every run models fees, carry, and single-bar liquidation identically.
"""

import sys
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


def load_daily(path: str) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_feather(p) if p.suffix == ".feather" else pd.read_parquet(p)
    tcol = "date" if "date" in df.columns else ("timestamp" if "timestamp" in df.columns else df.columns[0])
    if pd.api.types.is_numeric_dtype(df[tcol]):
        df[tcol] = pd.to_datetime(df[tcol], unit="ms")
    else:
        df[tcol] = pd.to_datetime(df[tcol])
    df = df.set_index(tcol).sort_index()
    daily = df.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return daily


def _liq_rebalanced(closes, lows, L, fee=DEFAULT_FEE_RATE, carry=DEFAULT_CARRY_DAILY):
    """Constant-leverage rebalanced long with honest liquidation + carry."""
    n = closes.size
    rets = np.zeros(n); rets[1:] = closes[1:] / closes[:-1] - 1.0
    lret = np.zeros(n); lret[1:] = lows[1:] / closes[:-1] - 1.0
    eq = np.ones(n); liq = None
    for i in range(1, n):
        if lret[i] <= -(1.0 / L) * (1 - 0.005):
            eq[i:] = 0.0; liq = i; break
        eq[i] = eq[i - 1] * (1 + L * rets[i]) * (1 - carry * max(L - 1, 0))
    return eq, liq


def _martingale(closes, lows, base_lev=0.5, add_drop=0.05, add_mult=1.5, max_lev=3.0,
                tp=0.03, fee=DEFAULT_FEE_RATE, carry=DEFAULT_CARRY_DAILY):
    """Toy martingale: open long, add (bigger) every -add_drop, take profit at +tp
    from avg entry. Exposure rises as price falls -- the anti-Kelly direction."""
    n = closes.size
    eq = np.ones(n); liq = None
    in_pos = False; avg = 0.0; lev = 0.0; last_add = 0.0
    rets = np.zeros(n); rets[1:] = closes[1:] / closes[:-1] - 1.0
    lret = np.zeros(n); lret[1:] = lows[1:] / closes[:-1] - 1.0
    for i in range(1, n):
        price = closes[i - 1]
        if lev > 0 and lret[i] <= -(1.0 / lev) * (1 - 0.005):
            eq[i:] = 0.0; liq = i; break
        if not in_pos:
            in_pos = True; avg = price; lev = base_lev; last_add = price
        else:
            if price <= last_add * (1 - add_drop) and lev < max_lev:
                add = min(lev * (add_mult - 1), max_lev - lev)
                avg = (avg * lev + price * add) / (lev + add)
                lev += add; last_add = price
            elif price >= avg * (1 + tp):
                in_pos = False; lev = 0.0
        eq[i] = eq[i - 1] * (1 + lev * rets[i]) * (1 - carry * max(lev - 1, 0))
    return eq, liq


def stats(eq, name, years, liq):
    final = float(eq[-1])
    if final > 0:
        peak = np.maximum.accumulate(eq); dd = float(((eq - peak) / peak).min())
        cagr = final ** (1 / years) - 1
        r = eq[1:] / eq[:-1] - 1; r = r[np.isfinite(r) & (eq[:-1] > 0)]
        sh = float(np.mean(r) / np.std(r) * np.sqrt(TRADING_DAYS)) if r.size > 2 and np.std(r) > 0 else 0.0
    else:
        dd, cagr, sh = -1.0, -1.0, 0.0
    fin = f"{final:,.2f}x" if final > 0 else "0 LIQUIDATED"
    flag = "" if liq is None else f"  💀@bar{liq}"
    return f"{name:<34} {fin:>16} {cagr*100:>8.1f}% {dd*100:>8.0f}% {sh:>7.2f}{flag}"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/freqtrade/user_data/data/binance/BTC_USDT-1h.feather"
    daily = load_daily(path)
    closes = daily["close"].to_numpy(); lows = daily["low"].to_numpy()
    years = len(closes) / TRADING_DAYS
    print(f"\nData: {path}")
    print(f"  {daily.index[0].date()} -> {daily.index[-1].date()}  ({years:.2f}y, {len(closes)} daily bars)")
    print(f"  fees={DEFAULT_FEE_RATE*1e4:.0f}bps/turn  carry={DEFAULT_CARRY_DAILY*TRADING_DAYS*100:.0f}%/yr on borrowed\n")
    print(f"{'strategy':<34} {'final':>16} {'CAGR':>9} {'maxDD':>9} {'Sharpe':>7}")
    print("-" * 86)

    eq, liq = _liq_rebalanced(closes, lows, 1.0); print(stats(eq, "buy & hold (L=1)", years, liq))
    for L in (2.0, 3.0):
        eq, liq = _liq_rebalanced(closes, lows, L); print(stats(eq, f"naive leverage (L={L:.0f})", years, liq))
    eq, liq = _martingale(closes, lows); print(stats(eq, "MARTINGALE (add into losses)", years, liq))
    print("-" * 86)

    configs = {
        "anti-mart  Lmax=2 trail20% volT": AntiMartingaleConfig(l_max=2.0),
        "anti-mart  Lmax=3 trail20% volT": AntiMartingaleConfig(l_max=3.0),
        "anti-mart  Lmax=3 trail25% noVT": AntiMartingaleConfig(l_max=3.0, trail_stop=0.25, use_vol_target=False),
        "anti-mart  Lmax=4 trail20% volT": AntiMartingaleConfig(l_max=4.0),
    }
    for name, cfg in configs.items():
        r = backtest_anti_martingale(closes, lows, cfg)
        liq = r.liq_index if r.liquidated else None
        line = stats(r.equity_curve, name, years, liq)
        print(line + f"   [mkt%={r.meta['time_in_market']*100:.0f} L̄={r.meta['mean_leverage']:.2f}]")
    print()


if __name__ == "__main__":
    main()
