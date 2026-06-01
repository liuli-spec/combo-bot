"""FastAPI app exposing the operator UI.

Endpoints:

* ``GET  /``                   — main page (Jinja2)
* ``GET  /api/status``         — state file + cached exchange data + pnl/regime
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

Exchange data is refreshed by a background asyncio task every 10 s
(singleton ccxt connection) rather than on every HTTP request, avoiding
rate-limit exhaustion on busy UIs.
"""

from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
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
    clear_stuck: bool = False,
) -> FastAPI:
    """Build the FastAPI app bound to a specific config file and
    trader profile. ``real=True`` means actually submit orders;
    leaving it False makes the Start button launch a dry-run trader."""

    proc_cfg = TraderProcessConfig(
        config_path=config_path, testnet=testnet, real=real, clear_stuck=clear_stuck
    )
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
    equity_path = state_path.with_suffix(".equity.jsonl")
    symbols: list[str] = cfg_data.get("symbols", [])

    # ─── exchange snapshot cache ────────────────────────────────────
    # Written every 10 s by the background refresh task; the status
    # endpoint reads from here instead of making live API calls.
    # One-element list so the background task can replace the whole
    # dict atomically (_exch_snap[0] = new_snap) without a lock —
    # asyncio is single-threaded so dict.update() mid-read is safe
    # too, but this is cleaner.
    _exch_snap: list[dict[str, Any]] = [
        {"balance": None, "open_orders_by_symbol": {}, "positions_by_symbol": {}}
    ]

    # ─── fills incremental cache ────────────────────────────────────
    # Tracks the byte offset of the last fully-read newline so each
    # call reads only the new bytes appended since the last poll.
    # Never re-reads the whole file regardless of how large it grows.
    _fills_state: dict[str, Any] = {"offset": 0, "rows": []}

    # ─── equity ring-buffer + sidecar ───────────────────────────────
    # In-memory: up to 720 samples for the chart.
    # On-disk:   equity_path sidecar so the curve survives server restart.
    equity_history: collections.deque[tuple[int, float]] = collections.deque(maxlen=720)
    _equity_state: dict[str, int] = {"last_written_ts": 0}

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
        """Per-source equity for the UI's split row (realized + unrealized).
        Prefers the grid_equity / trend_equity fields written by current
        LiveTrader (realized + open unrealized). Falls back to realized-only
        for state files written before those fields were persisted."""
        if not state:
            return {"grid_equity": 0.0, "trend_equity": 0.0}
        grid_eq = state.get("grid_equity")
        trend_eq = state.get("trend_equity")
        if grid_eq is None:
            grid_eq = state.get("grid_realized_pnl", 0)
        if trend_eq is None:
            trend_eq = state.get("trend_realized_pnl", 0)
        return {
            "grid_equity": float(grid_eq or 0),
            "trend_equity": float(trend_eq or 0),
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

    async def _fetch_orders(exchange: Any, sym: str) -> list[dict] | dict:
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

    async def _fetch_positions(exchange: Any, sym: str) -> list[dict] | dict:
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

    async def _do_exchange_refresh(exchange: Any) -> None:
        """Build a fresh exchange snapshot and store it atomically."""
        snap: dict[str, Any] = {
            "balance": None,
            "open_orders_by_symbol": {},
            "positions_by_symbol": {},
        }
        try:
            bal = await exchange.fetch_balance()
            usdt = bal.get("USDT", {}) if isinstance(bal, dict) else {}
            snap["balance"] = float(usdt.get("total", 0) or 0)
        except Exception as exc:
            snap["balance_error"] = str(exc)
        for sym in symbols:
            snap["open_orders_by_symbol"][sym] = await _fetch_orders(exchange, sym)
            snap["positions_by_symbol"][sym] = await _fetch_positions(exchange, sym)
        _exch_snap[0] = snap  # atomic replacement

    def _update_fills_cache() -> None:
        """Read only new bytes from the fills JSONL since the last call.
        Advances the byte offset only past complete lines so a partial
        last-line (mid-write by the trader) is safely retried next time."""
        rows: list[dict] = _fills_state["rows"]
        if not fills_path.exists():
            # File removed (state reset or rotation) — clear the in-memory
            # cache so the UI shows an empty table rather than stale data.
            rows.clear()
            _fills_state["offset"] = 0
            return
        size = fills_path.stat().st_size
        offset: int = _fills_state["offset"]
        if size < offset:
            # File was truncated or replaced — reset.
            _fills_state["offset"] = 0
            rows.clear()
            offset = 0
        if size == offset:
            return
        try:
            with fills_path.open("rb") as fh:
                fh.seek(offset)
                chunk = fh.read(size - offset)
        except Exception:
            logger.exception("[ui] incremental fills read failed")
            return
        # Only advance past complete lines (newline-terminated).
        last_nl = chunk.rfind(b"\n")
        if last_nl == -1:
            return  # no complete line yet; wait for next poll
        complete = chunk[: last_nl + 1]
        _fills_state["offset"] += len(complete)
        for line in complete.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        if len(rows) > 2000:
            del rows[:-2000]

    def _get_fills(limit: int, symbol: str | None = None) -> list[dict]:
        _update_fills_cache()
        rows = _fills_state["rows"][::-1]  # newest first
        if symbol:
            target = symbol.strip()
            rows = [r for r in rows if str(r.get("symbol", "")).strip() == target]
        return rows[: max(1, min(limit, 2000))]

    def _load_equity_history() -> None:
        """Pre-populate the ring-buffer from the equity sidecar at startup."""
        if not equity_path.exists():
            return
        try:
            lines = equity_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return
        for line in lines[-720:]:  # no need to load more than the deque holds
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                equity_history.append((int(rec["ts"]), float(rec["equity"])))
            except Exception:
                continue

    def _record_equity(ts: int, equity: float) -> None:
        """Add a sample to the in-memory ring-buffer and persist to disk.
        Writes are throttled to once per 30 s to avoid churning the sidecar."""
        equity_history.append((ts, equity))
        if ts - _equity_state["last_written_ts"] < 30_000:
            return
        _equity_state["last_written_ts"] = ts
        try:
            with equity_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": ts, "equity": equity}) + "\n")
        except Exception:
            logger.exception("[ui] equity sidecar write failed")

    # Warm up both caches synchronously before the first HTTP request.
    _load_equity_history()
    _update_fills_cache()

    # ─── job manager (backtest / optimize) ─────────────────────────
    # CPU-bound jobs run in a thread pool; the asyncio loop reads job
    # state via simple dict access (GIL protects scalar field writes).

    _thread_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="bt_job"
    )
    _jobs: dict[str, dict[str, Any]] = {}

    _MAX_JOBS = 50

    def _new_job(kind: str) -> str:
        jid = uuid.uuid4().hex[:8]
        _jobs[jid] = {
            "id": jid,
            "kind": kind,
            "status": "pending",
            "progress": 0,
            "progress_msg": "",
            "result": None,
            "error": None,
            "created_ms": int(time.time() * 1000),
        }
        # Evict the oldest FINISHED jobs once the table grows past the
        # cap so a long-lived UI session doesn't leak memory. Running /
        # pending jobs are never evicted.
        if len(_jobs) > _MAX_JOBS:
            finished = sorted(
                (j for j in _jobs.values() if j["status"] in ("done", "error")),
                key=lambda j: j["created_ms"],
            )
            for old in finished[: len(_jobs) - _MAX_JOBS]:
                _jobs.pop(old["id"], None)
        return jid

    def _downsample_curve(arr: Any, n: int) -> list:
        """Thin the equity curve to at most n points for the chart."""
        try:
            import numpy as np  # already a dep of backtest

            arr = np.asarray(arr)
            if len(arr) <= n:
                return arr.tolist()
            idx = np.linspace(0, len(arr) - 1, n, dtype=int)
            return arr[idx].tolist()
        except Exception:
            return []

    def _run_backtest_job(jid: str, body: dict) -> None:
        job = _jobs[jid]
        job["status"] = "running"
        job["progress"] = 5
        job["progress_msg"] = "加载行情数据…"
        try:
            from combo_bot.backtest import BacktestConfig, Backtester
            from combo_bot.data import load_cached_data
            from combo_bot.fusion_config import build_regime_config
            from combo_bot.grid_engine import GridConfig
            from combo_bot.merger import MergerConfig
            from combo_bot.risk import RiskConfig
            from combo_bot.trend_signal import TrendConfig

            data_dir = cfg_data.get("data_dir", "data")
            candle_data = load_cached_data(symbols, data_dir=data_dir)
            if not candle_data:
                raise ValueError("无缓存行情数据，请先运行 combo-futures download")

            job["progress"] = 20
            job["progress_msg"] = "运行回测…"

            ov = body.get("config", {})

            def _merge(base_key: str, override_key: str) -> dict:
                d = dict(cfg_data.get(base_key, {}))
                d.update(ov.get(override_key or base_key, {}))
                return d

            def _dc_kwargs(cls: type, d: dict) -> dict:
                valid = set(cls.__dataclass_fields__)
                return {k: v for k, v in d.items() if k in valid}

            bt_config = BacktestConfig(
                starting_balance=float(
                    ov.get("starting_balance", cfg_data.get("starting_balance", 10000))
                ),
                grid=GridConfig(**_dc_kwargs(GridConfig, _merge("grid", "grid"))),
                trend=TrendConfig(**_dc_kwargs(TrendConfig, _merge("trend", "trend"))),
                merger=MergerConfig(
                    **_dc_kwargs(MergerConfig, _merge("merger", "merger"))
                ),
                risk=RiskConfig(**_dc_kwargs(RiskConfig, cfg_data.get("risk", {}))),
                regime=build_regime_config(cfg_data),
                symbols=symbols,
            )

            result = Backtester(bt_config).run(candle_data)

            job["progress"] = 100
            job["status"] = "done"
            job["result"] = {
                "final_balance": result.final_balance,
                "total_pnl": result.total_pnl,
                "total_fees": result.total_fees,
                "n_trades": result.n_trades,
                "win_rate": result.win_rate,
                "adg": result.adg,
                "max_drawdown": result.max_drawdown,
                "sharpe_ratio": result.sharpe_ratio,
                "sortino_ratio": result.sortino_ratio,
                "calmar_ratio": result.calmar_ratio,
                "grid_pnl": result.grid_pnl,
                "trend_pnl": result.trend_pnl,
                "duration_days": result.duration_days,
                "equity_curve": _downsample_curve(result.equity_curve, 500),
            }
        except Exception as exc:
            job["status"] = "error"
            job["error"] = str(exc)
            logger.exception("[ui] backtest job %s failed", jid)

    def _run_optimize_job(jid: str, body: dict) -> None:
        job = _jobs[jid]
        job["status"] = "running"
        job["progress"] = 5
        job["progress_msg"] = "加载行情数据…"
        try:
            from combo_bot.data import load_cached_data
            from combo_bot.optimize import Optimizer, OptimizeConfig

            data_dir = cfg_data.get("data_dir", "data")
            candle_data = load_cached_data(symbols, data_dir=data_dir)
            if not candle_data:
                raise ValueError("无缓存行情数据，请先运行 combo-futures download")

            n_trials = int(body.get("n_trials", 100))
            sampler = str(body.get("sampler", "tpe"))
            wf_splits = int(body.get("walk_forward_splits", 3))
            # Optional multi-objective (Pareto) optimization. A non-empty
            # list like ["adg:max", "max_drawdown:min"] switches to a true
            # multi-objective NSGA-II run; empty/None keeps the legacy scalar.
            raw_obj = body.get("objectives") or None
            objectives = (
                [str(o) for o in raw_obj]
                if isinstance(raw_obj, list) and raw_obj
                else None
            )
            multi = objectives is not None

            opt_config = OptimizeConfig(
                n_trials=n_trials,
                n_jobs=1,
                sampler=sampler,
                walk_forward_splits=max(2, wf_splits),
                objectives=objectives,
                study_name=f"ui_{jid}",
            )

            completed: list[int] = [0]

            def _progress_cb(study: Any, trial: Any) -> None:
                completed[0] += 1
                pct = 10 + int(completed[0] / max(n_trials, 1) * 85)
                job["progress"] = min(pct, 95)
                if multi:
                    # Multi-objective studies have no single best_value;
                    # report the Pareto front size instead.
                    try:
                        front_n = len(study.best_trials)
                    except Exception:
                        front_n = 0
                    job["progress_msg"] = (
                        f"试验 {completed[0]}/{n_trials} | 帕累托前沿: {front_n}"
                    )
                else:
                    try:
                        best = study.best_value
                        best_str = f"{best:.4f}" if best is not None else "—"
                    except Exception:
                        best_str = "—"
                    job["progress_msg"] = (
                        f"试验 {completed[0]}/{n_trials} | 当前最优: {best_str}"
                    )

            job["progress"] = 10
            job["progress_msg"] = f"开始优化 (0/{n_trials})…"

            opt = Optimizer(opt_config, candle_data)
            result = opt.run(callbacks=[_progress_cb])

            # Persist the result so the best params survive a UI restart
            # (the in-memory job table is volatile). One JSON per run under
            # the data dir; best-effort, never fails the job.
            try:
                out_dir = Path(data_dir) / "optimize_results"
                out_dir.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                out_path = out_dir / f"{stamp}_{jid}.json"
                out_path.write_text(
                    json.dumps(
                        {
                            "job_id": jid,
                            "created_ms": _jobs[jid]["created_ms"],
                            "n_trials": n_trials,
                            "sampler": sampler,
                            "walk_forward_splits": wf_splits,
                            "result": result,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )
                job["result_path"] = str(out_path)
                logger.info("[ui] optimize result saved → %s", out_path)
            except Exception:
                logger.exception("[ui] failed to persist optimize result")

            job["progress"] = 100
            job["status"] = "done"
            job["result"] = result
        except ImportError:
            job["status"] = "error"
            job["error"] = "optuna 未安装，请运行: pip install optuna"
        except Exception as exc:
            job["status"] = "error"
            job["error"] = str(exc)
            logger.exception("[ui] optimize job %s failed", jid)

    # ─── lifespan: exchange singleton + background refresh ──────────

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[override]
        exchange = None
        refresh_task: asyncio.Task | None = None
        try:
            from combo_bot.data import create_exchange as _ce

            exchange = _ce(testnet=testnet)
            await exchange.load_markets()
            logger.info("[ui] exchange singleton initialised")
        except Exception:
            logger.exception(
                "[ui] exchange init failed — snapshot data will be unavailable"
            )

        async def _refresh_loop() -> None:
            while True:
                if exchange is not None:
                    try:
                        await _do_exchange_refresh(exchange)
                    except Exception:
                        logger.exception("[ui] exchange refresh cycle failed")
                await asyncio.sleep(10)

        refresh_task = asyncio.create_task(_refresh_loop())
        yield
        # ── shutdown ──────────────────────────────────────────────
        if refresh_task is not None:
            refresh_task.cancel()
            try:
                await refresh_task
            except asyncio.CancelledError:
                pass
        if exchange is not None:
            try:
                await exchange.close()
                logger.info("[ui] exchange singleton closed")
            except Exception:
                pass

    # ─── app ────────────────────────────────────────────────────────

    app = FastAPI(
        title="combo_bot UI", docs_url=None, redoc_url=None, lifespan=lifespan
    )
    templates = Jinja2Templates(directory=str(_PKG_DIR / "templates"))
    app.mount(
        "/static",
        StaticFiles(directory=str(_PKG_DIR / "static")),
        name="static",
    )

    # ─── routes ────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> Any:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "config_path": str(config_path),
                "profile": profile,
                "symbols": symbols,
                "real": real,
                "testnet": testnet,
            },
        )

    @app.get("/api/status")
    async def status_endpoint() -> JSONResponse:
        state = _load_state()
        ts = int(time.time() * 1000)
        if state is not None:
            equity_value = float(state.get("equity", 0) or 0)
            if equity_value > 0:
                _record_equity(ts, equity_value)
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
            "exchange": _exch_snap[0],  # ← cached; never blocks on live API
            "pnl": _derive_pnl(state),
            "regime": _derive_regime(state),
            "symbols_detail": (state or {}).get("symbols_detail", {}),
            "now_ms": ts,
        }
        # Inject incremental sidecar fills where the frontend's live-merge
        # path looks (state.fill_events.recent_fills).
        if state is not None:
            fe = payload["state"].setdefault("fill_events", {})
            fe["recent_fills"] = _get_fills(50)
        return JSONResponse(payload)

    @app.get("/api/equity")
    async def equity_endpoint() -> JSONResponse:
        return JSONResponse(
            {"samples": [{"ts": ts, "equity": eq} for ts, eq in equity_history]}
        )

    @app.get("/api/fills")
    async def fills_endpoint(
        limit: int = 200, symbol: str | None = None
    ) -> JSONResponse:
        """Recent fills from the sidecar, newest→oldest, optional
        symbol filter (e.g. ``?symbol=BTC/USDT:USDT``)."""
        fills = _get_fills(limit, symbol)
        return JSONResponse({"fills": fills, "returned": len(fills)})

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

    # ─── backtest / optimize routes ────────────────────────────────

    @app.get("/api/jobs")
    async def jobs_list() -> JSONResponse:
        recent = sorted(_jobs.values(), key=lambda j: j["created_ms"], reverse=True)[:20]
        return JSONResponse({"jobs": [_job_summary(j) for j in recent]})

    @app.get("/api/job/{job_id}")
    async def job_detail(job_id: str) -> JSONResponse:
        job = _jobs.get(job_id)
        if job is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(job)

    @app.post("/api/backtest/run")
    async def backtest_run(request: Request) -> JSONResponse:
        body = await request.json()
        jid = _new_job("backtest")
        loop = asyncio.get_event_loop()
        loop.run_in_executor(_thread_pool, _run_backtest_job, jid, body)
        return JSONResponse({"job_id": jid})

    @app.post("/api/optimize/run")
    async def optimize_run(request: Request) -> JSONResponse:
        body = await request.json()
        running = [
            j for j in _jobs.values()
            if j["kind"] == "optimize" and j["status"] == "running"
        ]
        if running:
            return JSONResponse(
                {"error": "已有优化任务在运行", "job_id": running[0]["id"]},
                status_code=409,
            )
        jid = _new_job("optimize")
        loop = asyncio.get_event_loop()
        loop.run_in_executor(_thread_pool, _run_optimize_job, jid, body)
        return JSONResponse({"job_id": jid})

    def _job_summary(j: dict) -> dict:
        """Strip large result fields for the list endpoint."""
        s = dict(j)
        if s.get("result") and s["kind"] == "backtest":
            r = dict(s["result"])
            r.pop("equity_curve", None)
            s["result"] = r
        return s

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

    @app.post("/api/control/clear_stuck")
    async def clear_stuck_fill(request: Request) -> JSONResponse:
        """Remove one symbol from the fill-event STUCK set in the persisted
        state file.  Effective on the next trader start — a running trader
        must be stopped and restarted for the change to take effect.

        Body JSON: ``{"symbol": "BTC/USDT:USDT"}``
        """
        body = await request.json()
        symbol = (body.get("symbol") or "").strip()
        if not symbol:
            return JSONResponse(
                {"ok": False, "error": "symbol required"}, status_code=400
            )
        if not state_path.exists():
            return JSONResponse({"ok": False, "error": "state file not found"})
        try:
            state_data = json.loads(state_path.read_text())
        except Exception as exc:
            return JSONResponse(
                {"ok": False, "error": f"read failed: {exc}"}, status_code=500
            )

        fe = state_data.get("fill_events") or {}
        stuck: list = list(fe.get("stuck_symbols") or [])
        if symbol not in stuck:
            return JSONResponse(
                {"ok": True, "symbol": symbol, "note": "不在 STUCK 列表中"}
            )

        stuck.remove(symbol)
        fe["stuck_symbols"] = stuck
        # Reset the escalation counter so the symbol starts fresh on restart.
        sc: dict = dict(fe.get("stuck_count") or {})
        sc.pop(symbol, None)
        fe["stuck_count"] = sc
        # Also clear the fail count if present (pre-6.x field name).
        fc: dict = dict(fe.get("fail_count") or {})
        fc.pop(symbol, None)
        fe["fail_count"] = fc
        state_data["fill_events"] = fe

        try:
            state_path.write_text(
                json.dumps(state_data, indent=2, ensure_ascii=False)
            )
        except Exception as exc:
            return JSONResponse(
                {"ok": False, "error": f"write failed: {exc}"}, status_code=500
            )

        logger.warning(
            "[ui] operator cleared STUCK for %s via web UI; remaining stuck: %s",
            symbol,
            stuck,
        )
        return JSONResponse(
            {
                "ok": True,
                "symbol": symbol,
                "remaining_stuck": stuck,
                "note": "已从状态文件移除，重启机器人后生效",
            }
        )

    return app


def run(
    config_path: Path,
    testnet: bool,
    real: bool,
    host: str,
    port: int,
    clear_stuck: bool = False,
) -> None:
    """Entry point used by the CLI. Blocks until killed."""
    import uvicorn

    app = create_app(
        config_path=config_path,
        testnet=testnet,
        real=real,
        clear_stuck=clear_stuck,
    )
    uvicorn.run(app, host=host, port=port, log_level="warning")
