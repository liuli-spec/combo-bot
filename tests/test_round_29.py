"""Round-29 operational tooling tests:

* LiveTrader.start() refuses to run when the STOPPED sentinel exists.
* Removing the sentinel allows restart.
* kill_switch cancels open orders, market-flattens both sides via
  fetch_positions ground truth, writes the sentinel.
* kill_switch writes the sentinel even when a flatten call fails
  (operator review must be unconditional).
* monitor renders a snapshot when the state file is missing without
  crashing.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

# ────────────────────────────────────────────────────────────────────
# STOPPED sentinel blocks LiveTrader.start()
# ────────────────────────────────────────────────────────────────────


class _NullExchange:
    """Minimal stub that records calls so we can assert what start()
    DID (or didn't) do after seeing the sentinel."""

    def __init__(self):
        self.load_markets_calls = 0
        self.create_order_calls = 0

    async def load_markets(self):
        self.load_markets_calls += 1
        return {}

    def market(self, _):
        return {
            "precision": {"amount": 0.001, "price": 0.01},
            "limits": {"amount": {"min": 0.001}, "cost": {"min": 5.0}},
            "maker": 0.0002,
            "taker": 0.0005,
        }

    async def fetch_balance(self, _=None):
        return {"USDT": {"total": 10_000.0}}

    async def fetch_positions(self, _):
        return []

    async def fetch_funding_rate(self, _):
        return {"fundingRate": 0.0}

    async def fetch_ohlcv(self, *a, **k):
        return []

    async def fetch_my_trades(self, *a, **k):
        return []

    async def fetch_open_orders(self, _):
        return []

    async def create_order(self, *a, **k):
        self.create_order_calls += 1
        return {"id": "ex-1", "status": "open"}

    async def cancel_order(self, *a, **k):
        return {}

    async def set_leverage(self, *a, **k):
        return {}

    async def set_margin_mode(self, *a, **k):
        return {}


def test_live_trader_refuses_to_start_when_stopped_sentinel_exists():
    """Round-29: the kill-switch sentinel is a hard interlock. Trader
    start() must short-circuit BEFORE _init_exchange so no exchange
    calls / leverage attempts / order activity happens until operator
    explicitly removes the sentinel."""
    from combo_bot.live import LiveConfig, LiveTrader

    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        state_file = tmpd / "state.json"
        sentinel = state_file.with_suffix(".STOPPED")
        sentinel.write_text('{"reason": "test"}\n')

        ex = _NullExchange()
        cfg = LiveConfig(
            symbols=["BTC/USDT:USDT"],
            dry_run=False,
            state_file=str(state_file),
        )
        trader = LiveTrader(cfg, ex)
        asyncio.run(trader.start())

    # No exchange interaction at all.
    assert ex.load_markets_calls == 0, (
        f"start() must short-circuit before load_markets when sentinel "
        f"exists; got load_markets_calls={ex.load_markets_calls}"
    )
    assert ex.create_order_calls == 0
    # The main loop never started.
    assert trader._running is False


def test_live_trader_starts_when_sentinel_removed():
    """After the operator removes the sentinel, start() must proceed.
    We assert by observing the exchange.load_markets call (the first
    thing _init_exchange does)."""
    from combo_bot.live import LiveConfig, LiveTrader

    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        state_file = tmpd / "state.json"
        # Sentinel never created — start() proceeds.
        ex = _NullExchange()
        cfg = LiveConfig(
            symbols=["BTC/USDT:USDT"],
            dry_run=False,
            state_file=str(state_file),
        )
        trader = LiveTrader(cfg, ex)
        # We don't want the main loop to actually run forever, so we
        # patch _running to False after init via a custom _tick that
        # stops on first invocation.
        original_tick = trader._tick

        async def _one_shot_tick():
            await original_tick()
            trader._running = False

        trader._tick = _one_shot_tick  # type: ignore[assignment]
        asyncio.run(trader.start())

    assert ex.load_markets_calls >= 1, (
        "without sentinel, start() must reach _init_exchange "
        "(load_markets call expected)"
    )


# ────────────────────────────────────────────────────────────────────
# kill_switch
# ────────────────────────────────────────────────────────────────────


class _KillExchange:
    """Stub exchange that records cancel + create_order calls so the
    test can verify the kill switch issued the right reduce-only
    flat-close orders."""

    def __init__(self, open_orders=None, positions=None, fail_flatten=False):
        self._opens = open_orders or []
        self._positions = positions or []
        self.cancels: list[tuple[str, str]] = []
        self.creates: list[dict] = []
        self.fail_flatten = fail_flatten
        self.closed = False

    async def load_markets(self):
        return {}

    async def fetch_open_orders(self, symbol):
        return [o for o in self._opens if o.get("symbol") == symbol]

    async def fetch_positions(self, symbols):
        # ccxt binanceusdm requires a LIST argument; accept either
        # form so the test stays robust to wrapper changes.
        if isinstance(symbols, str):
            symbols = [symbols]
        return [p for p in self._positions if p.get("symbol") in symbols]

    async def cancel_order(self, order_id, symbol):
        self.cancels.append((symbol, order_id))
        return {}

    async def create_order(self, symbol, order_type, side, qty, price, params):
        if self.fail_flatten:
            raise RuntimeError("simulated exchange failure")
        self.creates.append(
            {
                "symbol": symbol,
                "type": order_type,
                "side": side,
                "qty": qty,
                "price": price,
                "params": params,
            }
        )
        return {"id": f"flatten-{len(self.creates)}", "status": "closed"}

    async def close(self):
        self.closed = True


def _write_config(path: Path, symbols: list[str]) -> None:
    path.write_text(
        json.dumps({"symbols": symbols, "state_file": str(path.parent / "state.json")})
    )


def test_kill_switch_cancels_orders_and_flattens_both_sides(monkeypatch):
    """kill_switch must cancel every open order then submit market
    reduce-only orders for every non-zero position on both sides."""
    from combo_bot import kill_switch as ks_mod

    opens = [
        {"id": "o-1", "symbol": "BTC/USDT:USDT"},
        {"id": "o-2", "symbol": "BTC/USDT:USDT"},
    ]
    positions = [
        {"symbol": "BTC/USDT:USDT", "side": "long", "contracts": 0.05},
        {"symbol": "BTC/USDT:USDT", "side": "short", "contracts": 0.02},
    ]
    ex = _KillExchange(open_orders=opens, positions=positions)
    monkeypatch.setattr(
        "combo_bot.data.create_exchange",
        lambda testnet=False: ex,
    )

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "config.json"
        _write_config(cfg_path, ["BTC/USDT:USDT"])
        rc = asyncio.run(
            ks_mod.kill_switch(
                config_path=cfg_path, testnet=True, sentinel_reason="testing"
            )
        )

    assert rc == 0, "all cancels + flattens succeeded → exit code 0"
    # Both orders cancelled.
    assert sorted(ex.cancels) == [
        ("BTC/USDT:USDT", "o-1"),
        ("BTC/USDT:USDT", "o-2"),
    ], f"both orders must be cancelled; got {ex.cancels}"
    # Two flatten orders — one sell (close long), one buy (close short).
    flatten_sides = sorted(c["side"] for c in ex.creates)
    assert flatten_sides == ["buy", "sell"], (
        f"expected one buy (close short) + one sell (close long); "
        f"got sides {flatten_sides}"
    )
    for c in ex.creates:
        assert c["type"] == "market"
        assert c["params"].get("reduceOnly") is True
    assert ex.closed, "exchange session must be closed on the way out"


def test_kill_switch_writes_sentinel_with_summary(monkeypatch):
    """The sentinel file must contain reason + per-symbol summary
    (cancel/flatten counts) so operator review has a starting point."""
    from combo_bot import kill_switch as ks_mod

    opens = [{"id": "o-1", "symbol": "BTC/USDT:USDT"}]
    positions = [
        {"symbol": "BTC/USDT:USDT", "side": "long", "contracts": 0.05},
    ]
    ex = _KillExchange(open_orders=opens, positions=positions)
    monkeypatch.setattr("combo_bot.data.create_exchange", lambda testnet=False: ex)

    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        cfg_path = tmpd / "config.json"
        _write_config(cfg_path, ["BTC/USDT:USDT"])
        asyncio.run(
            ks_mod.kill_switch(
                config_path=cfg_path,
                testnet=True,
                sentinel_reason="api outage at 3am",
            )
        )
        sentinel = tmpd / "state.STOPPED"
        assert sentinel.exists(), f"sentinel must be written at {sentinel}"
        body = json.loads(sentinel.read_text())
        assert body["reason"] == "api outage at 3am"
        assert body["testnet"] is True
        assert len(body["summaries"]) == 1
        sym_summary = body["summaries"][0]
        assert sym_summary["symbol"] == "BTC/USDT:USDT"
        assert sym_summary["cancelled"] == 1
        assert sym_summary["closed_long"] == pytest.approx(0.05)
        assert sym_summary["closed_short"] is None


def test_kill_switch_writes_sentinel_even_when_flatten_fails(monkeypatch):
    """Partial-failure contract: sentinel is ALWAYS written so
    operator review is mandatory regardless of whether the exchange
    successfully accepted the flatten. Exit code is non-zero to
    signal partial failure to oncall scripts."""
    from combo_bot import kill_switch as ks_mod

    opens = []
    positions = [
        {"symbol": "BTC/USDT:USDT", "side": "long", "contracts": 0.05},
    ]
    ex = _KillExchange(open_orders=opens, positions=positions, fail_flatten=True)
    monkeypatch.setattr("combo_bot.data.create_exchange", lambda testnet=False: ex)

    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        cfg_path = tmpd / "config.json"
        _write_config(cfg_path, ["BTC/USDT:USDT"])
        rc = asyncio.run(
            ks_mod.kill_switch(
                config_path=cfg_path, testnet=True, sentinel_reason="forced"
            )
        )
        sentinel = tmpd / "state.STOPPED"
        assert sentinel.exists(), "sentinel must be written even on partial failure"
        body = json.loads(sentinel.read_text())
        assert body["summaries"][0]["close_long_failed"] is True
        assert rc == 1, f"partial failure must surface as exit code 1; got {rc}"


def test_kill_switch_idempotent_with_no_open_state(monkeypatch):
    """Running kill twice in a row: second run sees no orders + no
    positions and just re-writes the sentinel. No surprises."""
    from combo_bot import kill_switch as ks_mod

    ex = _KillExchange(open_orders=[], positions=[])
    monkeypatch.setattr("combo_bot.data.create_exchange", lambda testnet=False: ex)

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "config.json"
        _write_config(cfg_path, ["BTC/USDT:USDT"])
        rc = asyncio.run(
            ks_mod.kill_switch(config_path=cfg_path, testnet=True, sentinel_reason="r1")
        )
        assert rc == 0
        assert ex.cancels == []
        assert ex.creates == []
        sentinel = Path(tmp) / "state.STOPPED"
        assert sentinel.exists()


# ────────────────────────────────────────────────────────────────────
# Monitor
# ────────────────────────────────────────────────────────────────────


def test_monitor_handles_missing_state_file(capsys, monkeypatch):
    """Monitor must render a snapshot even when the state file
    doesn't exist yet (trader hasn't run / first boot)."""
    from combo_bot import monitor as mon

    ex = _NullExchange()
    monkeypatch.setattr("combo_bot.data.create_exchange", lambda testnet=False: ex)

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "config.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "symbols": ["BTC/USDT:USDT"],
                    "state_file": str(Path(tmp) / "state.json"),
                }
            )
        )
        rc = asyncio.run(
            mon._monitor_loop(
                config_path=cfg_path,
                testnet=True,
                interval_seconds=1.0,
                once=True,
            )
        )
    captured = capsys.readouterr()
    assert rc == 0
    assert "no state file" in captured.out
    # Exchange poll still happened (we got the open_orders=0 line).
    assert "open_orders=0" in captured.out


def test_monitor_renders_sentinel_warning(capsys, monkeypatch):
    """When the STOPPED sentinel exists, monitor must show the red
    banner so a glance at the snapshot tells operator the trader
    is halted."""
    from combo_bot import monitor as mon

    ex = _NullExchange()
    monkeypatch.setattr("combo_bot.data.create_exchange", lambda testnet=False: ex)

    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        cfg_path = tmpd / "config.json"
        state_path = tmpd / "state.json"
        cfg_path.write_text(
            json.dumps({"symbols": ["BTC/USDT:USDT"], "state_file": str(state_path)})
        )
        sentinel = state_path.with_suffix(".STOPPED")
        sentinel.write_text('{"reason": "test"}\n')
        asyncio.run(
            mon._monitor_loop(
                config_path=cfg_path,
                testnet=True,
                interval_seconds=1.0,
                once=True,
            )
        )
    captured = capsys.readouterr()
    assert (
        "STOPPED sentinel" in captured.out
    ), f"sentinel banner must render; got output:\n{captured.out}"


def test_monitor_surfaces_persistence_failed_and_stuck(capsys, monkeypatch):
    """Critical state flags must be visible at a glance.

    Round-29 state file uses a FLAT top-level shape (see live.py
    _save_state). stuck_symbols lives under ``state["fill_events"]
    ["stuck_symbols"]``, not the synthetic shape an earlier draft
    of this test invented.
    """
    from combo_bot import monitor as mon

    ex = _NullExchange()
    monkeypatch.setattr("combo_bot.data.create_exchange", lambda testnet=False: ex)

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
                    "balance": 1000.0,
                    "equity": 950.0,
                    "equity_peak": 1000.0,
                    "risk_tier": "green",
                    "risk_red_latched": False,
                    "risk_red_cooldown_until": 0,
                    "fill_events": {
                        "stuck_symbols": ["BTC/USDT:USDT"],
                        "seen_ids": {},
                        "last_ts_ms": {},
                    },
                    "unknown_overlay": [],
                    "pending_overlay": [],
                    "trend_buckets": {},
                }
            )
        )
        asyncio.run(
            mon._monitor_loop(
                config_path=cfg_path,
                testnet=True,
                interval_seconds=1.0,
                once=True,
            )
        )
    captured = capsys.readouterr()
    # persistence_failed isn't a top-level state field today — it's a
    # LiveTrader instance attribute that doesn't get serialized.
    # The visible-at-a-glance signals that ARE in state are
    # stuck_symbols (under fill_events) and unknown_overlay.
    assert (
        "stuck_symbols" in captured.out
    ), f"stuck_symbols warning must render; got output:\n{captured.out}"
    assert "BTC/USDT:USDT" in captured.out
