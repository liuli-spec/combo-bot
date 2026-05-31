"""Live trader read-only monitor.

Polls the active state file + the exchange for a periodic snapshot
of the things you want to see at 3am when something looks off:

* Balance, equity, drawdown from peak
* Number of open exchange orders (per symbol)
* Per-bucket position sizes (grid + trend, both sides)
* RiskManager tier (GREEN/YELLOW/ORANGE/RED) and latch state
* Pending overlay claims (TREND non-reduce in flight)
* Stuck-fill / persistence-failed flags
* STOPPED sentinel presence (red banner if active)

Usage::

    python -m combo_bot.monitor --config config.testnet.json [--testnet] \\
        [--interval 30]

Read-only — never mutates state or sends orders. Safe to run alongside
a live trader on the same state file. Output is one snapshot per
interval, prefixed with a UTC timestamp.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("combo_bot.monitor")


def _fmt_pos(bucket: dict) -> str:
    size = float(bucket.get("size", 0) or 0)
    if abs(size) < 1e-12:
        return "flat"
    return f"{size:.6f}@{float(bucket.get('entry_price', 0) or 0):.4f}"


def _load_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        logger.exception("[monitor] failed to parse %s", path)
        return None


async def _exchange_snapshot(exchange, symbols: list[str]) -> dict[str, Any]:
    """One-shot exchange poll — open-orders count + balance + positions."""
    snap: dict[str, Any] = {
        "balance": None,
        "open_orders_by_symbol": {},
        "exchange_positions": {},
    }
    try:
        bal = await exchange.fetch_balance()
        usdt = bal.get("USDT", {}) if isinstance(bal, dict) else {}
        snap["balance"] = float(usdt.get("total", 0) or 0)
    except Exception:
        logger.exception("[monitor] fetch_balance failed")

    for symbol in symbols:
        try:
            opens = await exchange.fetch_open_orders(symbol)
            snap["open_orders_by_symbol"][symbol] = len(opens)
        except Exception:
            logger.exception("[monitor] fetch_open_orders(%s) failed", symbol)
            snap["open_orders_by_symbol"][symbol] = "ERR"
        try:
            positions = await exchange.fetch_positions([symbol])
            pos_info = []
            for p in positions or []:
                if p.get("symbol") != symbol:
                    continue
                contracts = float(p.get("contracts", 0) or 0)
                if abs(contracts) < 1e-12:
                    continue
                pos_info.append(
                    {
                        "side": str(p.get("side", "")).lower(),
                        "contracts": contracts,
                        "entryPrice": float(p.get("entryPrice", 0) or 0),
                    }
                )
            snap["exchange_positions"][symbol] = pos_info
        except Exception:
            logger.exception("[monitor] fetch_positions(%s) failed", symbol)
            snap["exchange_positions"][symbol] = "ERR"
    return snap


def _print_snapshot(
    state: dict | None,
    exchange_snap: dict,
    sentinel_path: Path,
    config_path: Path,
) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"\n=== combo_bot monitor @ {ts}Z (config={config_path}) ===")

    if sentinel_path.exists():
        print(
            f"\033[1;31m  !! STOPPED sentinel present at {sentinel_path} — "
            "trader is halted until the file is removed !!\033[0m"
        )

    if state is None:
        print("  (no state file yet — trader may not have run / saved)")
    else:
        # The state file uses a FLAT top-level shape (see live.py
        # _save_state) — balance/equity/peak/risk_* sit at the root.
        balance_local = float(state.get("balance", 0) or 0)
        equity_local = float(state.get("equity", 0) or 0)
        peak = float(state.get("equity_peak", 0) or 0)
        dd = (peak - equity_local) / peak if peak > 0 else 0.0
        print(
            f"  state.balance={balance_local:.2f}  equity={equity_local:.2f}  ", end=""
        )
        print(f"peak={peak:.2f}  dd={dd:.2%}")
        print(
            f"  risk.tier={state.get('risk_tier', '?')}  red_latched="
            f"{state.get('risk_red_latched', '?')}  cooldown_until="
            f"{state.get('risk_red_cooldown_until', '?')}"
        )
        fe = state.get("fill_events", {}) or {}
        stuck = fe.get("stuck_symbols", []) or []
        unknown = state.get("unknown_overlay", []) or []
        pending = state.get("pending_overlay", []) or []
        if stuck:
            print(f"\033[1;33m  !! stuck_symbols={stuck} !!\033[0m")
        if unknown:
            print(f"\033[1;33m  !! unknown_overlay={unknown} !!\033[0m")
        if pending:
            print(f"  pending_overlay={pending}")
        # Trend bucket persistence: per-symbol dict with optional
        # long_size/long_entry_price and short_size/short_entry_price.
        # Grid positions are NOT in the state file — they're re-derived
        # from fetch_positions every tick, so check the exchange poll
        # section below for current grid exposure.
        trend_buckets = state.get("trend_buckets", {}) or {}
        if trend_buckets:
            for sym, bucket in trend_buckets.items():
                long_repr = (
                    f"{bucket.get('long_size', 0):.6f}@"
                    f"{bucket.get('long_entry_price', 0):.4f}"
                    if "long_size" in bucket
                    else "flat"
                )
                short_repr = (
                    f"{bucket.get('short_size', 0):.6f}@"
                    f"{bucket.get('short_entry_price', 0):.4f}"
                    if "short_size" in bucket
                    else "flat"
                )
                print(f"  {sym}  trend_long={long_repr}  trend_short={short_repr}")
        else:
            print("  trend_buckets: (no open trend positions)")

    print("  --- exchange poll ---")
    bal = exchange_snap.get("balance")
    if bal is not None:
        print(f"  exchange.balance={bal:.2f}")
    for sym, count in (exchange_snap.get("open_orders_by_symbol") or {}).items():
        print(f"  {sym}  open_orders={count}")
    for sym, pos in (exchange_snap.get("exchange_positions") or {}).items():
        if pos == "ERR":
            print(f"  {sym}  positions=ERR")
        elif not pos:
            print(f"  {sym}  positions=flat")
        else:
            for p in pos:
                print(
                    f"  {sym}  ex_position {p['side']}={p['contracts']:.6f}"
                    f"@{p['entryPrice']:.4f}"
                )


async def _monitor_loop(
    config_path: Path,
    testnet: bool,
    interval_seconds: float,
    once: bool,
) -> int:
    from combo_bot.data import create_exchange

    cfg = json.loads(config_path.read_text())
    symbols = cfg.get("symbols", [])
    profile = "testnet" if testnet else "real"
    default_state = f"state.{profile}.json"
    state_path = Path(cfg.get("state_file", default_state))
    sentinel_path = state_path.with_suffix(".STOPPED")

    exchange = create_exchange(testnet=testnet)
    try:
        await exchange.load_markets()
        while True:
            state = _load_state(state_path)
            exchange_snap = await _exchange_snapshot(exchange, symbols)
            _print_snapshot(state, exchange_snap, sentinel_path, config_path)
            sys.stdout.flush()
            if once:
                return 0
            await asyncio.sleep(interval_seconds)
    finally:
        try:
            await exchange.close()
        except Exception:
            pass


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Read-only live trader monitor")
    parser.add_argument("-c", "--config", default="config.json")
    parser.add_argument("--testnet", action="store_true")
    parser.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="Seconds between snapshots (default 30).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print one snapshot and exit (useful for cron / scripts).",
    )
    args = parser.parse_args()
    sys.exit(
        asyncio.run(
            _monitor_loop(
                config_path=Path(args.config),
                testnet=args.testnet,
                interval_seconds=max(1.0, args.interval),
                once=args.once,
            )
        )
    )


if __name__ == "__main__":
    main()
