"""FastAPI app exposing the operator UI.

Endpoints:

* ``GET  /``                   — main page (Jinja2)
* ``GET  /api/status``         — state file + exchange poll + pnl/regime
* ``GET  /api/equity``         — equity curve samples for the chart
* ``GET  /api/fills``          — recent fills from the JSONL sidecar
* ``GET  /api/logs/stream``    — SSE stream of trader stdout
* ``POST /api/control/start``  — spawn trader subprocess
* ``POST /api/control/stop``   — graceful SIGINT then SIGKILL fallback
* ``POST /api/control/kill``   — kill_switch (cancel+flat+STOPPED sentinel)
* ``POST /api/control/clear_sentinel`` — remove the STOPPED sentinel

Kept narrow on purpose: the UI is an OPERATOR console, not a remote
trading API. No order placement / config editing exposed — those
still go through CLI / config files.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from combo_bot.webui.process_manager import (
    TraderProcessConfig,
    TraderProcessManager,
)

logger = logging.getLogger("combo_bot.webui")

_PKG_DIR = Path(__file__).parent


def create_app(
    config_path: Path,
    testnet: bool,
    real: bool,
) -> FastAPI:
    """Build the FastAPI app bound to a specific config file and
    trader profile. ``real=True`` means actually submit orders;
    leaving it False makes the Start button launch a dry-run trader."""

    proc_cfg = TraderProcessConfig(config_path=config_path, testnet=testnet, real=real)
    proc = TraderProcessManager(proc_cfg)

    # Resolve the active state-file path the SAME way LiveTrader does
    # (see main.py:cmd_live). Without this, the UI would read a
    # different file than the trader is writing.
    cfg_data = json.loads(config_path.read_text())
    profile = "testnet" if testnet else ("real" if real else "dryrun")
    default_state_file = f"state.{profile}.json"
    state_path = Path(cfg_data.get("state_file", default_state_file))
    sentinel_path = state_path.with_suffix(".STOPPED")
    fills_path = state_path.with_suffix(".fills.jsonl")

    # Equity ring buffer — every status poll captures (ts, equity)
    # so the chart can render a curve without an external time-series
    # store. 720 samples = ~6h at 30s polling.
    equity_history: collections.deque[tuple[int, float]] = collections.deque(maxlen=720)

    app = FastAPI(title="combo_bot UI", docs_url=None, redoc_url=None)
    templates = Jinja2Templates(directory=str(_PKG_DIR / "templates"))
    app.mount(
        "/static",
        StaticFiles(directory=str(_PKG_DIR / "static")),
        name="static",
    )

    # ─── helpers ───────────────────────────────────────────────────

    def _load_state() -> dict[str, Any] | None:
        if not state_path.exists():
            return None
        try:
            return json.loads(state_path.read_text())
        except Exception:
            logger.exception("[ui] failed to parse state file %s", state_path)
            return None

    def _derive_pnl(state: dict | None) -> dict[str, Any]:
        """Per-source realized P&L for the UI's equity-split row.
        Frontend reads grid_equity / trend_equity."""
        if not state:
            return {"grid_equity": 0.0, "trend_equity": 0.0}
        return {
            "grid_equity": float(state.get("grid_realized_pnl", 0) or 0),
            "trend_equity": float(state.get("trend_realized_pnl", 0) or 0),
        }

    def _derive_regime(state: dict | None) -> dict[str, Any]:
        """Aggregate regime summary for the mini badge. Picks the
        highest-conviction symbol (|direction| * strength) as the
        'primary' regime; the per-symbol detail card shows the rest."""
        if not state:
            return {"primary": "neutral", "conviction": 0.0}
        detail = state.get("symbols_detail", {}) or {}
        best_primary = "neutral"
        best_conviction = 0.0
        for _sym, sd in detail.items():
            direction = abs(float(sd.get("signal_direction", 0) or 0))
            strength = float(sd.get("signal_strength", 0) or 0)
            conviction = direction * strength
            if conviction >= best_conviction:
                best_conviction = conviction
                best_primary = sd.get("signal_regime", "neutral") or "neutral"
        return {"primary": best_primary, "conviction": best_conviction}

    def _read_fills(limit: int) -> list[dict]:
        """Tail the trader's fills JSONL sidecar, oldest→newest."""
        if not fills_path.exists():
            return []
        limit = max(1, min(limit, 2000))
        try:
            lines = fills_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            logger.exception("[ui] failed to read fills log %s", fills_path)
            return []
        out: list[dict] = []
        for line in lines[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out

    async def _fetch_orders(exchange, sym: str) -> list[dict] | dict:
        try:
            opens = await exchange.fetch_open_orders(sym)
            return [
                {
                    "id": o.get("id"),
                    "side": str(o.get("side", "")).lower(),
                    "price": float(o.get("price", 0) or 0),
                    "amount": float(o.get("amount", 0) or 0),
                    "reduceOnly": bool(
                        o.get("reduceOnly")
                        or (o.get("info") or {}).get("reduceOnly") in (True, "true")
                    ),
                    "timestamp": int(o.get("timestamp", 0) or 0),
                }
                for o in (opens or [])
            ]
        except Exception as exc:
            return {"error": str(exc)}

    async def _fetch_positions(exchange, sym: str) -> list[dict] | dict:
        try:
            positions = await exchange.fetch_positions([sym])
            return [
                {
                    "side": str(p.get("side", "")).lower(),
                    "contracts": float(p.get("contracts", 0) or 0),
                    "entryPrice": float(p.get("entryPrice", 0) or 0),
                    "markPrice": float(p.get("markPrice", 0) or 0),
                    "unrealizedPnl": float(p.get("unrealizedPnl", 0) or 0),
                }
                for p in (positions or [])
                if abs(float(p.get("contracts", 0) or 0)) > 1e-12
            ]
        except Exception as exc:
            return {"error": str(exc)}

    async def _exchange_snapshot() -> dict[str, Any]:
        """One-shot exchange poll for the UI's live panels. Never
        raises — returns {error: "..."} on failure so the frontend
        can render a graceful degraded state."""
        try:
            from combo_bot.data import create_exchange

            exchange = create_exchange(testnet=testnet)
        except Exception as exc:
            return {"error": f"create_exchange failed: {exc}"}

        snap: dict[str, Any] = {
            "balance": None,
            "open_orders_by_symbol": {},
            "positions_by_symbol": {},
        }
        try:
            await exchange.load_markets()
            try:
                bal = await exchange.fetch_balance()
                usdt = bal.get("USDT", {}) if isinstance(bal, dict) else {}
                snap["balance"] = float(usdt.get("total", 0) or 0)
            except Exception as exc:
                snap["balance_error"] = str(exc)
            for sym in cfg_data.get("symbols", []):
                snap["open_orders_by_symbol"][sym] = await _fetch_orders(exchange, sym)
                snap["positions_by_symbol"][sym] = await _fetch_positions(exchange, sym)
        except Exception:
            logger.exception("[ui] exchange snapshot failed")
        finally:
            try:
                await exchange.close()
            except Exception:
                pass
        return snap

    # ─── routes ────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> Any:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "config_path": str(config_path),
                "profile": profile,
                "symbols": cfg_data.get("symbols", []),
                "real": real,
                "testnet": testnet,
            },
        )

    @app.get("/api/status")
    async def status_endpoint() -> JSONResponse:
        state = _load_state()
        exchange_snap = await _exchange_snapshot()
        if state is not None:
            equity_value = float(state.get("equity", 0) or 0)
            if equity_value > 0:
                equity_history.append((int(time.time() * 1000), equity_value))
        payload: dict[str, Any] = {
            "trader": {
                "state": proc.state.value,
                "is_running": proc.is_running(),
                "exit_code": proc.exit_code,
                "config_path": str(config_path),
                "profile": profile,
                "real": real,
                "testnet": testnet,
            },
            "sentinel_present": sentinel_path.exists(),
            "sentinel_path": str(sentinel_path),
            "state_file_present": state is not None,
            "state": state or {},
            "exchange": exchange_snap,
            "pnl": _derive_pnl(state),
            "regime": _derive_regime(state),
            "symbols_detail": (state or {}).get("symbols_detail", {}),
            "now_ms": int(time.time() * 1000),
        }
        # Inject recent sidecar fills where the frontend's live-merge
        # path looks (state.fill_events.recent_fills) so the trade
        # table updates every poll without a manual reload.
        if state is not None:
            fe = payload["state"].setdefault("fill_events", {})
            fe["recent_fills"] = _read_fills(50)[::-1]
        return JSONResponse(payload)

    @app.get("/api/equity")
    async def equity_endpoint() -> JSONResponse:
        return JSONResponse(
            {"samples": [{"ts": ts, "equity": eq} for ts, eq in equity_history]}
        )

    @app.get("/api/fills")
    async def fills_endpoint(limit: int = 200) -> JSONResponse:
        """Recent fills from the sidecar, newest→oldest for the table."""
        fills = _read_fills(limit)
        return JSONResponse({"fills": fills[::-1]})

    @app.get("/api/logs/stream")
    async def logs_stream() -> StreamingResponse:
        """Server-Sent Events stream of trader stdout. Sends the
        last 200 lines on connect, then new lines as they arrive.
        Keepalive comment every 15s prevents idle timeouts."""

        async def gen():
            last_idx = max(0, len(proc.log_buffer) - 200)
            for line in list(proc.log_buffer)[last_idx:]:
                yield f"data: {line.rstrip()}\n\n"
            last_idx = len(proc.log_buffer)
            last_keepalive = time.time()
            try:
                while True:
                    current = list(proc.log_buffer)
                    if len(current) > last_idx:
                        for line in current[last_idx:]:
                            yield f"data: {line.rstrip()}\n\n"
                        last_idx = len(current)
                        last_keepalive = time.time()
                    if time.time() - last_keepalive > 15:
                        yield ": keepalive\n\n"
                        last_keepalive = time.time()
                    await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                return

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/api/control/start")
    async def control_start() -> JSONResponse:
        ok, msg = await proc.start()
        return JSONResponse({"ok": ok, "message": msg, "state": proc.state.value})

    @app.post("/api/control/stop")
    async def control_stop() -> JSONResponse:
        ok, msg = await proc.stop()
        return JSONResponse(
            {
                "ok": ok,
                "message": msg,
                "state": proc.state.value,
                "exit_code": proc.exit_code,
            }
        )

    @app.post("/api/control/kill")
    async def control_kill() -> JSONResponse:
        # Stop the running trader first so kill_switch doesn't race
        # with reconcile, then run the operational kill switch.
        if proc.is_running():
            await proc.stop()
        try:
            from combo_bot.kill_switch import kill_switch as ks

            rc = await ks(
                config_path=config_path,
                testnet=testnet,
                sentinel_reason="ui_kill_switch",
            )
        except Exception as exc:
            logger.exception("[ui] kill_switch failed")
            return JSONResponse(
                {"ok": False, "message": f"kill_switch failed: {exc}"},
                status_code=500,
            )
        return JSONResponse(
            {
                "ok": rc == 0,
                "message": f"kill_switch returned exit code {rc}",
                "sentinel_present": sentinel_path.exists(),
            }
        )

    @app.post("/api/control/clear_sentinel")
    async def clear_sentinel() -> JSONResponse:
        """Remove STOPPED sentinel so the trader can start again."""
        if not sentinel_path.exists():
            return JSONResponse({"ok": True, "message": "no sentinel to remove"})
        try:
            sentinel_path.unlink()
        except Exception as exc:
            return JSONResponse(
                {"ok": False, "message": f"unlink failed: {exc}"}, status_code=500
            )
        return JSONResponse({"ok": True, "message": "sentinel removed"})

    return app


def run(
    config_path: Path,
    testnet: bool,
    real: bool,
    host: str,
    port: int,
) -> None:
    """Entry point used by the CLI. Blocks until killed."""
    import uvicorn

    app = create_app(config_path=config_path, testnet=testnet, real=real)
    uvicorn.run(app, host=host, port=port, log_level="warning")
