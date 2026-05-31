"""FastAPI app exposing the operator UI.

Endpoints:

* ``GET  /``                   — main page (Jinja2)
* ``GET  /api/status``         — combined state file + exchange poll snapshot
* ``GET  /api/equity``         — equity curve (recent samples) for the chart
* ``GET  /api/logs/stream``    — SSE stream of trader stdout
* ``POST /api/control/start``  — spawn trader subprocess
* ``POST /api/control/stop``   — graceful SIGINT then SIGKILL fallback
* ``POST /api/control/kill``   — invoke combo_bot.kill_switch + STOPPED sentinel

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
    # (see main.py:cmd_live). Without this, the UI's "Status" panel
    # would read a different file than the trader is writing.
    cfg_data = json.loads(config_path.read_text())
    profile = "testnet" if testnet else ("real" if real else "dryrun")
    default_state_file = f"state.{profile}.json"
    state_path = Path(cfg_data.get("state_file", default_state_file))
    sentinel_path = state_path.with_suffix(".STOPPED")

    # Equity ring buffer — every status poll captures (ts, equity)
    # so the chart can render a curve without needing an external
    # time-series store. 720 samples = ~6h at 30s polling.
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
                try:
                    opens = await exchange.fetch_open_orders(sym)
                    snap["open_orders_by_symbol"][sym] = [
                        {
                            "id": o.get("id"),
                            "side": str(o.get("side", "")).lower(),
                            "price": float(o.get("price", 0) or 0),
                            "amount": float(o.get("amount", 0) or 0),
                            "reduceOnly": bool(
                                o.get("reduceOnly")
                                or (o.get("info") or {}).get("reduceOnly")
                                in (True, "true")
                            ),
                            "timestamp": int(o.get("timestamp", 0) or 0),
                        }
                        for o in (opens or [])
                    ]
                except Exception as exc:
                    snap["open_orders_by_symbol"][sym] = {"error": str(exc)}
                try:
                    positions = await exchange.fetch_positions([sym])
                    snap["positions_by_symbol"][sym] = [
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
                    snap["positions_by_symbol"][sym] = {"error": str(exc)}
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
        # Append equity sample if we just learned a new value.
        equity_value = None
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
            "now_ms": int(time.time() * 1000),
        }
        return JSONResponse(payload)

    @app.get("/api/equity")
    async def equity_endpoint() -> JSONResponse:
        return JSONResponse(
            {
                "samples": [{"ts": ts, "equity": eq} for ts, eq in equity_history],
            }
        )

    @app.get("/api/logs/stream")
    async def logs_stream() -> StreamingResponse:
        """Server-Sent Events stream of trader stdout. Sends the
        last 200 lines on connect, then any new lines as they
        arrive. Keepalive comment every 15s prevents idle
        timeouts on reverse proxies."""

        async def gen():
            last_idx = max(0, len(proc.log_buffer) - 200)
            # Initial burst.
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
        # First stop the running trader (if any) so kill_switch
        # doesn't race with reconcile. Then run the operational
        # kill switch (cancel+flat+sentinel). Then make sure the
        # subprocess is fully reaped.
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
        """Operator action: remove STOPPED sentinel so the trader can
        be started again. Mirrors `rm state.<profile>.STOPPED`."""
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
