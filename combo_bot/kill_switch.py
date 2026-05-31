"""Emergency stop: cancel all open orders, flat-close all positions,
write the STOPPED sentinel so the trader refuses to restart.

Usage::

    python -m combo_bot.kill_switch --config config.testnet.json [--testnet]

Operations contract:

1. Connects to the configured exchange (testnet or real per flag).
2. For every symbol in the config: cancels every open order, then
   submits a market reduce-only order for any non-zero position on
   either side.
3. Writes the STOPPED sentinel next to the state file. LiveTrader
   refuses to ``start()`` while the sentinel exists.

Idempotent — running it twice in a row is safe; the second run sees
no open orders / no positions and just (re)writes the sentinel.
Designed to be invoked by oncall in the middle of the night without
needing to remember any internal detail.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("combo_bot.kill_switch")


async def _flatten_symbol(exchange, symbol: str) -> dict:
    """Cancel every open order, then market-close every open position
    on ``symbol``. Returns a per-action summary dict."""
    summary: dict = {
        "symbol": symbol,
        "cancelled": 0,
        "cancel_failed": 0,
        "closed_long": None,
        "closed_short": None,
        "close_long_failed": False,
        "close_short_failed": False,
    }

    # 1) Cancel all open orders.
    try:
        opens = await exchange.fetch_open_orders(symbol)
    except Exception:
        logger.exception("[kill_switch] fetch_open_orders(%s) failed", symbol)
        opens = []
    for o in opens:
        oid = o.get("id")
        if not oid:
            continue
        try:
            await exchange.cancel_order(oid, symbol)
            summary["cancelled"] += 1
        except Exception:
            logger.exception("[kill_switch] cancel_order(%s, %s) failed", oid, symbol)
            summary["cancel_failed"] += 1

    # 2) Flat-close every open position. Use fetch_positions to get
    # ground truth from the exchange — don't trust the local state
    # file because the operator may be running kill_switch precisely
    # BECAUSE local state diverged.
    try:
        positions = await exchange.fetch_positions(symbol)
    except Exception:
        logger.exception("[kill_switch] fetch_positions(%s) failed", symbol)
        positions = []

    long_size = 0.0
    short_size = 0.0
    for p in positions or []:
        if p.get("symbol") != symbol:
            continue
        side = str(p.get("side", "")).lower()
        contracts = float(p.get("contracts", 0) or 0)
        if side == "long":
            long_size += abs(contracts)
        elif side == "short":
            short_size += abs(contracts)

    if long_size > 0:
        try:
            await exchange.create_order(
                symbol,
                "market",
                "sell",  # close a long = sell
                long_size,
                None,
                {"reduceOnly": True},
            )
            summary["closed_long"] = long_size
            logger.info("[kill_switch] flattened LONG %.10f %s", long_size, symbol)
        except Exception:
            logger.exception(
                "[kill_switch] failed to flatten long %s qty=%s", symbol, long_size
            )
            summary["close_long_failed"] = True

    if short_size > 0:
        try:
            await exchange.create_order(
                symbol,
                "market",
                "buy",  # close a short = buy
                short_size,
                None,
                {"reduceOnly": True},
            )
            summary["closed_short"] = short_size
            logger.info("[kill_switch] flattened SHORT %.10f %s", short_size, symbol)
        except Exception:
            logger.exception(
                "[kill_switch] failed to flatten short %s qty=%s", symbol, short_size
            )
            summary["close_short_failed"] = True

    return summary


async def kill_switch(
    config_path: Path,
    testnet: bool = False,
    sentinel_reason: str = "manual",
) -> int:
    """Run the kill switch. Returns process exit code (0 = clean)."""
    from combo_bot.data import create_exchange

    cfg = json.loads(config_path.read_text())
    symbols = cfg.get("symbols", [])
    if not symbols:
        logger.error("No symbols in %s — nothing to flatten", config_path)
        return 2

    # Match the state-file naming the live CLI uses so the sentinel
    # lands beside the active profile's state. Without an override
    # we use the testnet/real default; explicit state_file in the
    # config wins (matches combo_bot/main.py:cmd_live).
    profile = "testnet" if testnet else "real"
    default_state = f"state.{profile}.json"
    state_path = Path(cfg.get("state_file", default_state))
    sentinel_path = state_path.with_suffix(".STOPPED")

    exchange = create_exchange(testnet=testnet)
    await exchange.load_markets()

    summaries: list[dict] = []
    try:
        for symbol in symbols:
            summary = await _flatten_symbol(exchange, symbol)
            summaries.append(summary)
    finally:
        # Always close the exchange session, even on partial failure,
        # so we don't leak HTTP keep-alives.
        try:
            await exchange.close()
        except Exception:
            pass

    # Write the sentinel UNCONDITIONALLY — even if some flatten calls
    # failed. Operator review is mandatory after an emergency stop
    # regardless of whether the flatten succeeded.
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel_path.write_text(
        json.dumps(
            {
                "reason": sentinel_reason,
                "summaries": summaries,
                "config": str(config_path),
                "testnet": testnet,
            },
            indent=2,
        )
        + "\n"
    )
    logger.info("[kill_switch] sentinel written to %s", sentinel_path)

    # Report exit code: non-zero if ANY flatten or cancel failed so
    # oncall scripts can detect partial failure.
    any_failed = any(
        s["cancel_failed"] or s["close_long_failed"] or s["close_short_failed"]
        for s in summaries
    )
    return 1 if any_failed else 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Emergency stop: cancel all open orders + flat-close all "
            "positions + write STOPPED sentinel."
        )
    )
    parser.add_argument("-c", "--config", default="config.json")
    parser.add_argument("--testnet", action="store_true")
    parser.add_argument(
        "--reason",
        default="manual",
        help="Free-form note recorded in the sentinel file.",
    )
    args = parser.parse_args()
    code = asyncio.run(
        kill_switch(
            config_path=Path(args.config),
            testnet=args.testnet,
            sentinel_reason=args.reason,
        )
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
