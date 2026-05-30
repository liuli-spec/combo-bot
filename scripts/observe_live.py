#!/usr/bin/env python3
"""Live-trader observability dump.

Wraps a LiveTrader instance and records every meaningful state
transition to per-symbol CSV files in ``--out-dir``. Designed to
answer the four "is execution correct?" questions a testnet run
needs to settle before any real-money exposure:

* Are we double-emitting TREND market entries?
  → ``tick.csv`` shows ``pending_overlay`` size vs ``trend_long.size``;
    every overlay create should leave pending != 0 until a fill row
    shows up in ``fills.csv``.

* Is the cOID cache actually being matched by reconcile, or are we
  silently cancel-recreating?
  → ``reconcile.csv`` logs every desired order's cOID and whether it
    matched an existing exchange order; fuzzy fallback counts should
    stay near zero once warmed up.

* Are local buckets drifting from the exchange's aggregate?
  → ``buckets.csv`` records per-tick local vs exchange totals.

* Does ledger PnL track what the exchange reports?
  → ``pnl.csv`` records grid/trend realized_pnl + the latest
    fill.realized_pnl from FillEventManager so you can diff.

USAGE
-----

    python -m scripts.observe_live --config config.testnet.json \
        --out-dir runs/$(date +%Y%m%d-%H%M)

The observer never modifies trading decisions — it taps the existing
LiveTrader and logs. Stops on Ctrl-C, flushing CSVs.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("observe_live")


# ----------------------------------------------------------------------
# CSV writers
# ----------------------------------------------------------------------


class _CsvLog:
    def __init__(self, path: Path, header: list[str]) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", newline="")
        self._w = csv.writer(self._file)
        self._w.writerow(header)
        self._file.flush()

    def append(self, row: list[Any]) -> None:
        self._w.writerow(row)
        self._file.flush()

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass


# ----------------------------------------------------------------------
# Observer
# ----------------------------------------------------------------------


class LiveObserver:
    """Subscribe to LiveTrader tick events via method wrapping."""

    def __init__(self, trader, out_dir: Path) -> None:
        self.trader = trader
        self.out_dir = out_dir
        self.tick_log = _CsvLog(
            out_dir / "tick.csv",
            [
                "ts_iso", "tick_ms", "balance", "equity", "drawdown",
                "risk_tier", "red_latched", "pending_overlay_count",
                "cid_cache_size", "open_orders_count",
            ],
        )
        self.bucket_log = _CsvLog(
            out_dir / "buckets.csv",
            [
                "ts_iso", "symbol", "side",
                "grid_size", "grid_entry_price",
                "trend_size", "trend_entry_price",
                "local_total", "best_price",
            ],
        )
        self.fill_log = _CsvLog(
            out_dir / "fills.csv",
            [
                "ts_iso", "fill_ts", "symbol", "side", "source",
                "reduce_only", "price", "qty", "fee", "realized_pnl",
            ],
        )
        self.pnl_log = _CsvLog(
            out_dir / "pnl.csv",
            [
                "ts_iso", "grid_realized_pnl", "trend_realized_pnl",
                "grid_equity_peak", "trend_equity_peak",
                "grid_loss_24h", "trend_loss_24h",
            ],
        )
        self.reconcile_log = _CsvLog(
            out_dir / "reconcile.csv",
            [
                "ts_iso", "symbol", "side", "source", "reduce_only",
                "is_market", "price", "qty", "client_order_id",
                "decision",
            ],
        )
        self._wrap()

    def _wrap(self) -> None:
        orig_tick = self.trader._tick
        orig_create = self.trader._create_order
        orig_cancel = self.trader._cancel_order

        async def _wrapped_tick() -> None:
            await orig_tick()
            try:
                self._snapshot_tick()
            except Exception:
                logger.exception("snapshot_tick failed")

        async def _wrapped_create(order):
            # Log the create AFTER cOID has been stamped — earlier
            # rounds put the log call before stamp_cid in reconcile,
            # so the cOID column was always blank. Hook at _create_order
            # is the right place: this is exactly the moment the bot
            # has decided to send.
            try:
                self._snapshot_reconcile("create", order)
            except Exception:
                logger.exception("snapshot_reconcile create failed")
            return await orig_create(order)

        async def _wrapped_cancel(existing):
            try:
                self._snapshot_cancel(existing)
            except Exception:
                logger.exception("snapshot_reconcile cancel failed")
            return await orig_cancel(existing)

        self.trader._tick = _wrapped_tick  # type: ignore[assignment]
        self.trader._create_order = _wrapped_create  # type: ignore[assignment]
        self.trader._cancel_order = _wrapped_cancel  # type: ignore[assignment]

        # Wrap the trader's enrich path so we see every emitted fill.
        orig_enrich = self.trader._enrich_fill_pnl

        def _wrapped_enrich(fill):
            enriched = orig_enrich(fill)
            try:
                self._snapshot_fill(enriched)
            except Exception:
                logger.exception("snapshot_fill failed")
            return enriched

        self.trader._enrich_fill_pnl = _wrapped_enrich  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------

    def _iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _snapshot_tick(self) -> None:
        tr = self.trader
        ts_iso = self._iso()
        try:
            risk_tier = getattr(tr.risk.tier, "value", str(tr.risk.tier))
        except Exception:
            risk_tier = "?"
        open_orders_count = sum(
            len(v) for v in tr._open_orders.values()
        )
        self.tick_log.append([
            ts_iso, tr._now_ms(),
            f"{tr.account.balance:.4f}",
            f"{tr.account.equity:.4f}",
            f"{tr.account.drawdown:.6f}",
            risk_tier,
            bool(getattr(tr.risk, "red_latched", False)),
            len(tr._pending_overlay),
            len(tr._cid_by_desired),
            open_orders_count,
        ])

        # Buckets — one row per symbol-side.
        for sym, ss in tr.account.symbols.items():
            self.bucket_log.append([
                ts_iso, sym, "long",
                f"{ss.position_long.size:.8f}",
                f"{ss.position_long.entry_price:.4f}",
                f"{ss.trend_long.size:.8f}",
                f"{ss.trend_long.entry_price:.4f}",
                f"{ss.position_long.size + ss.trend_long.size:.8f}",
                f"{ss.position_long.best_price:.4f}",
            ])
            self.bucket_log.append([
                ts_iso, sym, "short",
                f"{ss.position_short.size:.8f}",
                f"{ss.position_short.entry_price:.4f}",
                f"{ss.trend_short.size:.8f}",
                f"{ss.trend_short.entry_price:.4f}",
                f"{ss.position_short.size + ss.trend_short.size:.8f}",
                f"{ss.position_short.best_price:.4f}",
            ])

        # PnL ledger snapshot.
        try:
            grid_l = -tr.account.loss_24h(__import__("combo_bot.types", fromlist=["OrderSource"]).OrderSource.GRID, tr._now_ms())
            trend_l = -tr.account.loss_24h(__import__("combo_bot.types", fromlist=["OrderSource"]).OrderSource.TREND, tr._now_ms())
        except Exception:
            grid_l = trend_l = 0.0
        self.pnl_log.append([
            ts_iso,
            f"{tr.account.grid_realized_pnl:.4f}",
            f"{tr.account.trend_realized_pnl:.4f}",
            f"{tr.account.grid_equity_peak:.4f}",
            f"{tr.account.trend_equity_peak:.4f}",
            f"{grid_l:.4f}",
            f"{trend_l:.4f}",
        ])

    def _snapshot_reconcile(self, decision: str, order) -> None:
        """Per-order decision log. ``decision`` is the action the
        trader is about to take: ``create``, ``cancel``."""
        cid = getattr(order, "client_order_id", "") or ""
        self.reconcile_log.append([
            self._iso(), order.symbol, order.side.value, order.source.value,
            bool(order.reduce_only), bool(order.is_market),
            f"{order.price:.4f}", f"{order.qty:.8f}",
            cid, decision,
        ])

    def _snapshot_cancel(self, existing: dict) -> None:
        cid = (
            existing.get("clientOrderId")
            or (existing.get("info") or {}).get("clientOrderId")
            or ""
        )
        self.reconcile_log.append([
            self._iso(),
            existing.get("symbol", ""),
            str(existing.get("side", "")).lower(),
            "?",   # source unknown from exchange-side record
            bool(existing.get("reduceOnly", False)),
            False,
            f"{float(existing.get('price', 0) or 0):.4f}",
            f"{float(existing.get('amount', 0) or 0):.8f}",
            str(cid),
            "cancel",
        ])

    def _snapshot_fill(self, fill) -> None:
        ts_iso = self._iso()
        self.fill_log.append([
            ts_iso, fill.timestamp, fill.symbol, fill.side.value,
            fill.source.value, bool(fill.reduce_only),
            f"{fill.price:.4f}", f"{fill.qty:.8f}",
            f"{fill.fee:.4f}", f"{fill.realized_pnl:.4f}",
        ])

    def close(self) -> None:
        for log in (
            self.tick_log, self.bucket_log, self.fill_log,
            self.pnl_log, self.reconcile_log,
        ):
            log.close()


# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------


def _build_trader(cfg: dict[str, Any], dry_run_override: bool | None):
    from combo_bot.data import create_exchange
    from combo_bot.grid_engine import GridConfig
    from combo_bot.live import LiveConfig, LiveTrader
    from combo_bot.merger import MergerConfig
    from combo_bot.risk import RiskConfig
    from combo_bot.trend_signal import TrendConfig
    from combo_bot.fusion_config import build_fusion, build_regime_config

    timeframe = str(cfg.get("timeframe", "1m"))
    _TF_TO_MIN = {
        "1m": 1.0, "3m": 3.0, "5m": 5.0, "15m": 15.0, "30m": 30.0,
        "1h": 60.0, "2h": 120.0, "4h": 240.0,
    }
    bar_min = cfg.get("bar_interval_minutes") or _TF_TO_MIN.get(timeframe, 1.0)

    dry_run = bool(cfg.get("dry_run", True))
    if dry_run_override is not None:
        dry_run = dry_run_override

    live_cfg = LiveConfig(
        symbols=cfg.get("symbols", []),
        leverage=cfg.get("leverage", 5),
        dry_run=dry_run,
        candle_timeframe=timeframe,
        bar_interval_minutes=float(bar_min),
        loop_interval_seconds=float(cfg.get("loop_interval_seconds", 60)),
        grid=GridConfig(**{
            k: v for k, v in cfg.get("grid", {}).items() if not k.startswith("_")
        }),
        trend=TrendConfig(**{
            k: v for k, v in cfg.get("trend", {}).items() if not k.startswith("_")
        }),
        merger=MergerConfig(**{
            k: v for k, v in cfg.get("merger", {}).items() if not k.startswith("_")
        }),
        risk=RiskConfig(**{
            k: v for k, v in cfg.get("risk", {}).items() if not k.startswith("_")
        }),
        regime=build_regime_config(cfg),
    )
    fusion = build_fusion(cfg)
    exchange = create_exchange(testnet=True)
    trader = LiveTrader(live_cfg, exchange, **fusion)
    return trader


async def _run(trader, observer, stop_event: asyncio.Event) -> None:
    try:
        task = asyncio.create_task(trader.start())
        await stop_event.wait()
        await trader.stop()
        await task
    finally:
        observer.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-o", "--out-dir", required=True)
    parser.add_argument(
        "--force-dry-run", action="store_true",
        help="ignore config.dry_run and force dry_run=True for safety.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = json.loads(Path(args.config).read_text())
    out_dir = Path(args.out_dir)

    trader = _build_trader(
        cfg, dry_run_override=True if args.force_dry_run else None,
    )
    observer = LiveObserver(trader, out_dir)

    stop_event = asyncio.Event()

    def _signal_handler(*_):
        logger.info("Stop signal received; shutting down")
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    asyncio.run(_run(trader, observer, stop_event))
    logger.info("Observer CSVs written to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
