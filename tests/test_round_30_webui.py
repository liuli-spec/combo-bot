"""Round-30 web UI tests.

* FastAPI app constructs without launching anything live.
* Index page renders and references static assets.
* /api/status returns the expected envelope even with no state file.
* Start/Stop endpoints are reachable without a real combo-futures
  install (the start fails cleanly, which we assert).
* clear_sentinel correctly removes the STOPPED file.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


def _ensure_fastapi() -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")


def _write_config(path: Path, symbols=None) -> None:
    path.write_text(
        json.dumps(
            {
                "symbols": symbols or ["BTC/USDT:USDT"],
                "state_file": str(path.parent / "state.json"),
            }
        )
    )


def test_create_app_imports_and_mounts_static():
    _ensure_fastapi()
    from combo_bot.webui.server import create_app

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "config.json"
        _write_config(cfg_path)
        app = create_app(config_path=cfg_path, testnet=True, real=False)
        # Static mount, templates, routes — verify presence.
        route_paths = {getattr(r, "path", None) for r in app.routes}
        assert "/" in route_paths
        assert "/api/status" in route_paths
        assert "/api/control/start" in route_paths
        assert "/api/control/stop" in route_paths
        assert "/api/control/kill" in route_paths


def test_index_renders_with_profile_info():
    _ensure_fastapi()
    from fastapi.testclient import TestClient

    from combo_bot.webui.server import create_app

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "config.json"
        _write_config(cfg_path, symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"])
        app = create_app(config_path=cfg_path, testnet=True, real=True)
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.text
        # Brand + profile pill must render (round-30b: Chinese UI).
        assert "COMBO" in body
        # The profile class name keeps the english slug for CSS.
        assert "profile-testnet" in body
        # Real flag should surface the Chinese tag.
        assert "真实下单" in body
        # Testnet tag also present.
        assert "测试网" in body
        # Symbols appear in the control meta.
        assert "BTC/USDT:USDT" in body
        assert "ETH/USDT:USDT" in body


def test_static_assets_served():
    _ensure_fastapi()
    from fastapi.testclient import TestClient

    from combo_bot.webui.server import create_app

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "config.json"
        _write_config(cfg_path)
        app = create_app(config_path=cfg_path, testnet=True, real=False)
        client = TestClient(app)
        css = client.get("/static/style.css")
        js = client.get("/static/app.js")
        assert css.status_code == 200
        assert js.status_code == 200
        # Sanity: the polish bits we promised should be present.
        assert "--emerald" in css.text
        assert "fmtUsd" in js.text


def test_status_includes_pnl_regime_and_symbol_detail(monkeypatch):
    """Round-30b: /api/status surfaces pnl split, aggregated regime,
    and per-symbol detail derived from the state file."""
    _ensure_fastapi()
    from fastapi.testclient import TestClient

    from combo_bot.webui.server import create_app

    monkeypatch.setattr(
        "combo_bot.data.create_exchange",
        lambda testnet=False: (_ for _ in ()).throw(RuntimeError("no net")),
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        cfg_path = tmpd / "config.json"
        state_path = tmpd / "state.json"
        cfg_path.write_text(
            json.dumps({"symbols": ["BTC/USDT:USDT"], "state_file": str(state_path)})
        )
        state_path.write_text(
            json.dumps(
                {
                    "equity": 5000,
                    "balance": 5000,
                    "equity_peak": 5000,
                    "risk_tier": "green",
                    "grid_realized_pnl": 12.5,
                    "trend_realized_pnl": -3.0,
                    "symbols_detail": {
                        "BTC/USDT:USDT": {
                            "last_price": 73000,
                            "mode_long": "tp_only",
                            "mode_short": "normal",
                            "signal_direction": -0.4,
                            "signal_strength": 0.6,
                            "signal_regime": "bear",
                        }
                    },
                }
            )
        )
        client = TestClient(create_app(config_path=cfg_path, testnet=True, real=False))
        data = client.get("/api/status").json()
        assert data["pnl"]["grid_equity"] == 12.5
        assert data["pnl"]["trend_equity"] == -3.0
        # |−0.4| * 0.6 = 0.24 conviction, regime bear.
        assert data["regime"]["primary"] == "bear"
        assert abs(data["regime"]["conviction"] - 0.24) < 1e-9
        assert data["symbols_detail"]["BTC/USDT:USDT"]["mode_long"] == "tp_only"


def test_fills_endpoint_tails_sidecar(monkeypatch):
    """Round-30b: /api/fills reads the JSONL sidecar beside the
    state file, newest entries within the limit window."""
    _ensure_fastapi()
    from fastapi.testclient import TestClient

    from combo_bot.webui.server import create_app

    monkeypatch.setattr(
        "combo_bot.data.create_exchange",
        lambda testnet=False: (_ for _ in ()).throw(RuntimeError("no net")),
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        cfg_path = tmpd / "config.json"
        state_path = tmpd / "state.json"
        cfg_path.write_text(
            json.dumps({"symbols": ["BTC/USDT:USDT"], "state_file": str(state_path)})
        )
        fills_path = state_path.with_suffix(".fills.jsonl")
        rows = [
            {
                "timestamp": i,
                "symbol": "BTC/USDT:USDT",
                "side": "sell",
                "source": "grid",
                "qty": 0.001,
                "price": 73000 + i,
                "fee": 0.05,
                "realized_pnl": 1.0,
            }
            for i in range(5)
        ]
        fills_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        client = TestClient(create_app(config_path=cfg_path, testnet=True, real=False))
        # limit=3, newest-first → timestamps 4,3,2.
        data = client.get("/api/fills?limit=3").json()
        assert len(data["fills"]) == 3
        assert [f["timestamp"] for f in data["fills"]] == [4, 3, 2]
        # symbol filter keeps only matching rows.
        data_f = client.get("/api/fills?symbol=BTC/USDT:USDT").json()
        assert len(data_f["fills"]) == 5
        # No sidecar → empty list, not an error.
        fills_path.unlink()
        data2 = client.get("/api/fills").json()
        assert data2["fills"] == []


def test_status_endpoint_returns_envelope(monkeypatch):
    """/api/status must succeed even when the state file is absent
    and the exchange creation raises (no network in tests). The
    payload shape stays stable so the frontend can render."""
    _ensure_fastapi()
    from fastapi.testclient import TestClient

    from combo_bot.webui.server import create_app

    # Force the exchange-create path to fail so the test doesn't
    # depend on network / ccxt being installed.
    def _boom_create(testnet=False):
        raise RuntimeError("no exchange in test environment")

    monkeypatch.setattr("combo_bot.data.create_exchange", _boom_create)

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "config.json"
        _write_config(cfg_path)
        app = create_app(config_path=cfg_path, testnet=True, real=False)
        client = TestClient(app)
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "trader" in data
        assert data["trader"]["is_running"] is False
        assert data["state_file_present"] is False
        assert data["sentinel_present"] is False
        assert "exchange" in data
        # exchange poll failed → error key present, but the envelope still parses.
        assert "error" in data["exchange"]


def test_clear_sentinel_removes_file():
    _ensure_fastapi()
    from fastapi.testclient import TestClient

    from combo_bot.webui.server import create_app

    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        cfg_path = tmpd / "config.json"
        _write_config(cfg_path)
        # Create the STOPPED sentinel as the kill switch would.
        sentinel = (tmpd / "state.json").with_suffix(".STOPPED")
        sentinel.write_text('{"reason": "test"}')
        app = create_app(config_path=cfg_path, testnet=True, real=False)
        client = TestClient(app)
        resp = client.post("/api/control/clear_sentinel")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert not sentinel.exists()


def test_start_endpoint_is_reachable(monkeypatch):
    """The start endpoint should respond cleanly even if the
    combo-futures subprocess can't be spawned in the test env
    (no entry-point installed in the venv pytest runs in). We
    just verify the call lands and returns an ok=false JSON."""
    _ensure_fastapi()
    from fastapi.testclient import TestClient

    from combo_bot.webui.server import create_app

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "config.json"
        _write_config(cfg_path)
        app = create_app(config_path=cfg_path, testnet=True, real=False)
        client = TestClient(app)
        # Make the spawn deterministically fail by forcing the
        # subprocess to point at a missing binary.
        monkeypatch.setenv("PATH", "/nonexistent")
        resp = client.post("/api/control/start")
        assert resp.status_code == 200
        data = resp.json()
        # ok may be False (FileNotFoundError) — we just need a
        # well-formed envelope.
        assert "state" in data
        assert "message" in data


def test_process_manager_state_transitions_idle():
    """TraderProcessManager state defaults to STOPPED and stop()
    while not running is a no-op."""
    import asyncio

    from combo_bot.webui.process_manager import (
        TraderProcessConfig,
        TraderProcessManager,
        TraderState,
    )

    with tempfile.TemporaryDirectory() as tmp:
        cfg = TraderProcessConfig(
            config_path=Path(tmp) / "x.json", testnet=True, real=False
        )
        mgr = TraderProcessManager(cfg)
        assert mgr.state == TraderState.STOPPED
        assert mgr.is_running() is False
        ok, msg = asyncio.run(mgr.stop())
        assert ok is False
        assert "not running" in msg
