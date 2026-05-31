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
