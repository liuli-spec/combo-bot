"""Manage the trader subprocess on behalf of the web UI.

Wraps ``combo-futures live`` as an async subprocess:
* Captures stdout/stderr into a bounded ring buffer that SSE
  streams to the browser.
* Sends ``YES\\n`` to stdin automatically when ``--real`` is set so
  the operator doesn't have to confirm in a terminal they can't see.
* Graceful stop via SIGINT (so LiveTrader._save_state runs); force
  kill after timeout.
* Tracks lifecycle state so the UI can render Start / Stop /
  Killed reliably.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import os
import signal
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger("combo_bot.webui.process")


class TraderState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    EXITED = "exited"
    CRASHED = "crashed"


@dataclass
class TraderProcessConfig:
    config_path: Path
    testnet: bool
    real: bool
    clear_stuck: bool = False
    log_buffer_size: int = 2000
    stop_timeout_seconds: float = 20.0


class TraderProcessManager:
    """Singleton-ish trader subprocess controller.

    Only one trader runs at a time per UI instance — Start while
    already running is a no-op that returns ``False``. Stop is
    idempotent. The log buffer is preserved across stops so the
    operator can scroll back through the last session.
    """

    def __init__(self, config: TraderProcessConfig):
        self.config = config
        self.process: asyncio.subprocess.Process | None = None
        self.state: TraderState = TraderState.STOPPED
        self.log_buffer: collections.deque[str] = collections.deque(
            maxlen=config.log_buffer_size
        )
        self.exit_code: int | None = None
        self._reader_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    def is_running(self) -> bool:
        return self.state in (TraderState.STARTING, TraderState.RUNNING)

    async def start(self) -> tuple[bool, str]:
        """Spawn the trader. Returns (ok, message). Refuses when an
        instance is already alive."""
        async with self._lock:
            if self.is_running():
                return False, f"trader already in state {self.state.value}"
            args = ["combo-futures", "live", "-c", str(self.config.config_path)]
            if self.config.testnet:
                args.append("--testnet")
            if self.config.real:
                args.append("--real")
            if self.config.clear_stuck:
                args.append("--clear-stuck")
            self.state = TraderState.STARTING
            self.exit_code = None
            self._append_log(f"[ui] launching: {' '.join(args)}\n")
            try:
                self.process = await asyncio.create_subprocess_exec(
                    *args,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    # Detach into its own process group so Ctrl-C on
                    # the UI server doesn't propagate (we manage the
                    # trader's lifecycle explicitly).
                    preexec_fn=os.setsid if os.name == "posix" else None,
                )
            except FileNotFoundError as exc:
                self.state = TraderState.CRASHED
                msg = (
                    f"combo-futures executable not found: {exc}. "
                    "Run `pip install -e .` first."
                )
                self._append_log(f"[ui] ERROR: {msg}\n")
                return False, msg
            # Auto-confirm the YES prompt when --real is set.
            if self.config.real and self.process.stdin is not None:
                try:
                    self.process.stdin.write(b"YES\n")
                    await self.process.stdin.drain()
                except Exception:
                    logger.exception("[ui] failed to send YES to trader stdin")
            self.state = TraderState.RUNNING
            self._reader_task = asyncio.create_task(self._read_loop())
            return True, "trader started"

    async def stop(self) -> tuple[bool, str]:
        """Graceful stop via SIGINT (so _save_state runs). Force kill
        after stop_timeout_seconds."""
        async with self._lock:
            if not self.process or self.process.returncode is not None:
                self.state = TraderState.STOPPED
                return False, "trader is not running"
            self.state = TraderState.STOPPING
            self._append_log("[ui] sending SIGINT (graceful stop)\n")
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(self.process.pid), signal.SIGINT)
                else:
                    self.process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                # Process already exited between checks.
                pass
            try:
                await asyncio.wait_for(
                    self.process.wait(), timeout=self.config.stop_timeout_seconds
                )
            except asyncio.TimeoutError:
                self._append_log(
                    f"[ui] graceful stop timed out after "
                    f"{self.config.stop_timeout_seconds}s — escalating to SIGKILL\n"
                )
                try:
                    if os.name == "posix":
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    else:
                        self.process.kill()
                    await self.process.wait()
                except ProcessLookupError:
                    pass
            self.exit_code = self.process.returncode
            self.state = (
                TraderState.EXITED if self.exit_code == 0 else TraderState.CRASHED
            )
            self._append_log(f"[ui] trader exited with code {self.exit_code}\n")
            return True, f"trader stopped (exit={self.exit_code})"

    async def _read_loop(self) -> None:
        assert self.process and self.process.stdout
        try:
            while True:
                line = await self.process.stdout.readline()
                if not line:
                    break
                self._append_log(line.decode("utf-8", errors="replace"))
        except Exception:
            logger.exception("[ui] log reader crashed")
        # Process ended. Update state only if we weren't already
        # mid-stop (stop() sets EXITED/CRASHED itself).
        if self.state in (TraderState.RUNNING, TraderState.STARTING):
            await self.process.wait()
            self.exit_code = self.process.returncode
            self.state = (
                TraderState.EXITED if self.exit_code == 0 else TraderState.CRASHED
            )
            self._append_log(
                f"[ui] trader process ended unexpectedly (code={self.exit_code})\n"
            )

    def _append_log(self, line: str) -> None:
        if not line.endswith("\n"):
            line = line + "\n"
        self.log_buffer.append(line)

    def snapshot_logs(self, limit: int | None = None) -> list[str]:
        """Return up to ``limit`` most recent log lines (oldest first)."""
        if limit is None:
            return list(self.log_buffer)
        return list(self.log_buffer)[-limit:]
