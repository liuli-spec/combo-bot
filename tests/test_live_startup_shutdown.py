from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


def test_start_sh_does_not_clear_stuck_by_default():
    """The launcher must not auto-clear fill-event STUCK state.

    STUCK means the fill ledger could not prove forward progress. Clearing it
    needs explicit operator review, not a default profile arg.
    """

    script = Path("start.sh").read_text()
    default_line = next(
        line for line in script.splitlines() if line.startswith("PROFILE_ARGS=")
    )
    assert "--clear-stuck" not in default_line


def test_cmd_live_handles_cancelled_start_without_traceback(monkeypatch, tmp_path):
    """A SIGINT during the live sleep loop cancels trader.start().

    cmd_live should turn that cancellation into a graceful trader.stop() and
    exchange.close(), instead of letting asyncio.run print a traceback.
    """

    import combo_bot.data
    import combo_bot.live
    import combo_bot.main as main_mod

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"symbols": []}))

    calls: list[str] = []

    class FakeExchange:
        async def close(self):
            calls.append("exchange.close")

    class FakeTrader:
        def __init__(self, *_args, **_kwargs):
            pass

        async def start(self):
            calls.append("trader.start")
            raise asyncio.CancelledError()

        async def stop(self):
            calls.append("trader.stop")

    monkeypatch.setattr(combo_bot.data, "create_exchange", lambda testnet=False: FakeExchange())
    monkeypatch.setattr(combo_bot.live, "LiveTrader", FakeTrader)

    args = argparse.Namespace(
        config=str(cfg_path),
        testnet=True,
        real=False,
        clear_stuck=False,
    )

    main_mod.cmd_live(args)

    assert calls == ["trader.start", "trader.stop", "exchange.close"]
