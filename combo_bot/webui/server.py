"""FastAPI app exposing the operator UI.

Endpoints:

* ``GET  /``                   — main page (Jinja2)
* ``GET  /api/status``         — state + exchange + regime + fills summary
* ``GET  /api/equity``         — equity curve (recent samples) for the chart
* ``GET  /api/logs/stream``    — SSE stream of trader stdout
* ``POST /api/control/start``  — spawn trader subprocess
* ``POST /api/control/stop``   — graceful SIGINT then SIGKILL fallback
* ``POST /api/control/kill``   — invoke combo_bot.kill_switch + STOPPED sentinel
* ``GET  /api/fills``          — recent fills from intent journal

Kept narrow on purpose: the UI is an OPERATOR console, not a remote
trading API. No order placement / config editing exposed — those
still go through CLI / config files.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from contextlib import asynccontextmanager

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

    # Fills ring buffer — populated from state file each poll so the
    # `/api/fills` endpoint is cheap. 500 entries = ~hours of trading.
    fills_cache: collections.deque[dict[str, Any]] = collections.deque(maxlen=500)

    # ─── exchange connection cache ─────────────────────────────────
    # Instead of creating + load_markets + close on every 2s poll
    # (~1800 API calls/hr), keep one authenticated connection alive
    # and refresh every 60s.  Matches the pattern Hummingbot and
    # FreqUI use (cached gateway / WS-backed connector).
    # Must be defined BEFORE FastAPI because lifespan closure captures it.

    _exchange_cache: dict[str, Any] = {
        "exchange": None,
        "last_refresh_ms": 0,
        "ttl_ms": 60_000,  # refresh exchange every 60s
    }

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        """Cleanup cached exchange on shutdown."""
        try:
            yield
        finally:
            old = _exchange_cache.get("exchange")
            if old is not None:
                try:
                    await old.close()
                except Exception:
                    pass

    app = FastAPI(
        title="combo_bot UI", docs_url=None, redoc_url=None, lifespan=_lifespan
    )
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

    async def _get_exchange():
        """Return a cached, authenticated exchange instance.  Refreshes
        every ``ttl_ms`` so credentials / session tokens don't stale."""
        now = int(time.time() * 1000)
        cache = _exchange_cache
        if (
            cache["exchange"] is not None
            and (now - cache["last_refresh_ms"]) < cache["ttl_ms"]
        ):
            return cache["exchange"]
        # Close old one before replacing.
        old = cache["exchange"]
        if old is not None:
            try:
                await old.close()
            except Exception:
                pass
        from combo_bot.data import create_exchange

        exchange = create_exchange(testnet=testnet)
        try:
            await exchange.load_markets()
        except Exception as exc:
            logger.warning("[ui] exchange load_markets failed: %s", exc)
        cache["exchange"] = exchange
        cache["last_refresh_ms"] = now
        return exchange

    def _derive_pnl(state: dict | None) -> dict[str, Any]:
        """Per-source realized P&L for the UI's equity-split row."""
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

    async def _exchange_snapshot() -> dict[str, Any]:
        """Poll exchange with a cached connection. Never raises."""
        snap: dict[str, Any] = {
            "balance": None,
            "open_orders_by_symbol": {},
            "positions_by_symbol": {},
        }
        try:
            exchange = await _get_exchange()
        except Exception as exc:
            return {"error": f"create_exchange failed: {exc}"}

        try:
            try:
                bal = await exchange.fetch_balance()
                if isinstance(bal, dict):
                    usdt = bal.get("USDT", {}) or {}
                    snap["balance"] = float(usdt.get("total", 0) or 0)
            except Exception as exc:
                snap["balance_error"] = str(exc)
            for sym in cfg_data.get("symbols", []):
                snap["open_orders_by_symbol"][sym] = await _fetch_orders(
                    exchange, sym
                )
                snap["positions_by_symbol"][sym] = await _fetch_positions(
                    exchange, sym
                )
        except Exception:
            logger.exception("[ui] exchange snapshot failed")
        return snap

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
                        or (o.get("info") or {}).get("reduceOnly")
                        in (True, "true")
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

    def _refresh_fills_cache(state: dict[str, Any] | None) -> None:
        """Merge fill-events from state into the fills ring buffer."""
        if not state:
            return
        fe = state.get("fill_events") or {}
        recent = fe.get("recent_fills") or []
        if not recent:
            return
        seen = {json.dumps(f, sort_keys=True, default=str) for f in fills_cache}
        for f in recent:
            key = json.dumps(f, sort_keys=True, default=str)
            if key not in seen:
                fills_cache.append(f)
                seen.add(key)

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
        if state is not None:
            equity_value = float(state.get("equity", 0) or 0)
            if equity_value > 0:
                equity_history.append((int(time.time() * 1000), equity_value))
            _refresh_fills_cache(state)
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
        # ── enrich with grid/trend PnL split + regime ────────────
        if state is not None:
            payload["pnl"] = {
                "grid_realized_pnl": float(state.get("grid_realized_pnl", 0) or 0),
                "trend_realized_pnl": float(state.get("trend_realized_pnl", 0) or 0),
                "grid_equity": float(state.get("grid_equity", 0) or 0),
                "trend_equity": float(state.get("trend_equity", 0) or 0),
            }
            payload["regime"] = state.get("regime") or {}
            # Per-symbol mode snapshot (one row per symbol × side)
            symbols_detail = state.get("symbols_detail") or {}
            payload["symbols_detail"] = symbols_detail
        else:
            payload["pnl"] = {}
            payload["regime"] = {}
        return JSONResponse(payload)

    @app.get("/api/equity")
    async def equity_endpoint() -> JSONResponse:
        return JSONResponse(
            {
                "samples": [{"ts": ts, "equity": eq} for ts, eq in equity_history],
            }
        )

    @app.get("/api/fills")
    async def fills_endpoint(
        limit: int = 100, symbol: str | None = None
    ) -> JSONResponse:
        """Recent fills from state journal, filterable by symbol.

        Query params:
          limit  — max entries (default 100, capped at 500)
          symbol — optional filter (e.g. ``?symbol=BTC/USDT:USDT``)
        """
        limit = max(1, min(limit, 500))
        fills = list(fills_cache)
        if symbol:
            # Normalise: strip leading/trailing whitespace, uppercase
            # the base.  The journal stores symbols as-is from ccxt
            # (e.g. "BTC/USDT:USDT"), so we do case-sensitive match
            # but strip spaces.
            target = symbol.strip()
            fills = [f for f in fills if f.get("symbol", "").strip() == target]
        # Return most-recent-first.
        fills.reverse()
        fills = fills[:limit]
        return JSONResponse(
            {"fills": fills, "total": len(fills_cache), "returned": len(fills)}
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