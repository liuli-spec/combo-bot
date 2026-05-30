"""Append-only, fsync-on-write intent journal for live execution.

Closes the window between ``_create_order`` deciding to send an order
and the next ``_save_state`` flushing memory state to disk. Without
this, a SIGKILL between those two moments loses:

* the source/side/reduce_only attribution for an in-flight cID;
* any TREND pending-overlay claim.

The exchange may still have accepted the order. On restart we read
the journal and resurrect attribution + pending, so the eventual fill
gets routed to TREND (not the default GRID fallback) and the next
overlay decision doesn't double-up.

Layout
------

JSONL — one record per line. Always written through ``flush() + fsync()``
so partial-writes on crash are bounded to the trailing line. Replay
ignores malformed trailing junk.

Record shape::

    {"ts": 1700000000000, "kind": "submit", "cid": "cb-abc",
     "symbol": "BTC/USDT:USDT", "side": "long", "source": "trend",
     "reduce_only": false, "is_market": true}
    {"ts": 1700000000050, "kind": "open",    "cid": "cb-abc",
     "exchange_id": "12345"}
    {"ts": 1700000000100, "kind": "filled",  "cid": "cb-abc"}

Terminal kinds: ``filled``, ``rejected``, ``canceled``, ``resolved``.

Compaction
----------

After every successful ``_save_state``, ``LiveTrader`` calls
:meth:`compact` — rewrites the journal keeping only non-terminal cIDs.
Cheap because terminal rows dominate during steady-state operation.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_TERMINAL = frozenset({"filled", "rejected", "canceled", "resolved"})


class IntentJournal:
    """Append-only durable record of in-flight orders.

    Single-process. Safe to call across asyncio coroutines because
    every write opens, appends, fsyncs, and closes — no cross-coroutine
    file handle state to race over.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # In-memory mirror of cid → latest record, rebuilt at startup
        # from the on-disk journal. Reading this is what callers use
        # to discover non-terminal cIDs at replay time.
        self.records: dict[str, dict[str, Any]] = {}
        # Set by ``replay()`` when the on-disk journal exists but
        # can't be read (IO error, parse blow-up beyond the tolerated
        # last-line partial). Callers must treat this the same as
        # state-file load failure — i.e. enter persistence-failed
        # mode and block new exposure. A non-existent journal is NOT
        # a failure (cold start).
        self.last_replay_failed: bool = False

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------

    def _append(self, record: dict[str, Any]) -> None:
        """Single fsync'd append. Failure here is FATAL — we'd rather
        crash than send an order whose intent never hit durable
        storage. Caller is expected to propagate the exception."""
        line = json.dumps(record, separators=(",", ":")) + "\n"
        # Use O_APPEND so concurrent processes (shouldn't happen, but
        # defence in depth) don't trample each other's writes.
        fd = os.open(
            self.path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)

    def submit(
        self,
        *,
        cid: str,
        symbol: str,
        side: str,
        source: str,
        reduce_only: bool,
        is_market: bool,
        now_ms: int,
    ) -> None:
        if not cid:
            return
        record = {
            "ts": int(now_ms),
            "kind": "submit",
            "cid": cid,
            "symbol": symbol,
            "side": side,
            "source": source,
            "reduce_only": bool(reduce_only),
            "is_market": bool(is_market),
        }
        self._append(record)
        self.records[cid] = record

    def open(self, *, cid: str, exchange_id: str, now_ms: int) -> None:
        if not cid:
            return
        record = {
            "ts": int(now_ms),
            "kind": "open",
            "cid": cid,
            "exchange_id": exchange_id,
        }
        self._append(record)
        # Preserve submit metadata; merge open on top so replay still
        # knows source/side after compaction sees an "open" alone.
        prior = self.records.get(cid) or {}
        merged = dict(prior)
        merged.update(record)
        self.records[cid] = merged

    def mark_terminal(
        self,
        *,
        cid: str,
        kind: str,
        now_ms: int,
        reason: str = "",
    ) -> None:
        if not cid:
            return
        if kind not in _TERMINAL:
            raise ValueError(f"non-terminal kind {kind!r}")
        record = {
            "ts": int(now_ms),
            "kind": kind,
            "cid": cid,
        }
        if reason:
            record["reason"] = reason
        self._append(record)
        self.records[cid] = record

    # ------------------------------------------------------------------
    # Replay / compaction
    # ------------------------------------------------------------------

    def replay(self) -> dict[str, dict[str, Any]]:
        """Read the on-disk journal and rebuild ``self.records``.

        Returns a copy of the rebuilt map — caller (LiveTrader) uses it
        to resurrect ``order_source`` / ``order_meta`` / pending
        overlay claims for non-terminal cIDs.

        On IO / parse failure, ``self.last_replay_failed`` is set so
        the caller can fail-closed; a non-existent file is treated as
        a normal cold start (not a failure).
        """
        self.records.clear()
        self.last_replay_failed = False
        if not self.path.exists():
            return {}
        try:
            # Read once so we can know which line is "last" — a partial
            # trailing line from an in-flight write at crash time is
            # tolerable, but a malformed line *in the middle* of the
            # file means data corruption and we must fail-closed.
            with self.path.open("r", encoding="utf-8") as fp:
                raw_lines = fp.readlines()
            # Pre-pass: find the last non-blank line index — that's the
            # one allowed to be partial.
            last_nonblank_idx = -1
            for i, raw in enumerate(raw_lines):
                if raw.strip():
                    last_nonblank_idx = i
            for i, raw in enumerate(raw_lines):
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    rec = json.loads(stripped)
                except json.JSONDecodeError:
                    is_last_line = i == last_nonblank_idx
                    # Only the trailing line is allowed to be partial —
                    # and only if it lacks the closing newline (a crash
                    # mid-write signature). Any other corruption means
                    # the journal isn't trustworthy.
                    trailing_partial = is_last_line and not raw.endswith("\n")
                    if trailing_partial:
                        logger.warning(
                            "[intent_journal] tolerating trailing " "partial line: %r",
                            stripped[:120],
                        )
                        continue
                    logger.error(
                        "[intent_journal] malformed mid-stream record at "
                        "line %d: %r — entering fail-closed mode",
                        i,
                        stripped[:120],
                    )
                    self.records.clear()
                    self.last_replay_failed = True
                    return {}
                cid = rec.get("cid")
                if not cid:
                    continue
                # Last-write-wins per cid; preserve metadata fields
                # from earlier records so a journal compacted down
                # to just "open" still knows source/side.
                prior = self.records.get(cid) or {}
                merged = dict(prior)
                merged.update(rec)
                self.records[cid] = merged
        except Exception:
            logger.exception(
                "[intent_journal] replay failed — flagging caller to "
                "fail-closed; in-flight cIDs may have been lost"
            )
            self.records.clear()
            self.last_replay_failed = True
        return dict(self.records)

    def non_terminal(self) -> dict[str, dict[str, Any]]:
        """In-flight cIDs (state == submit or open)."""
        return {
            cid: rec
            for cid, rec in self.records.items()
            if rec.get("kind") not in _TERMINAL
        }

    def compact(self) -> None:
        """Rewrite the journal keeping only non-terminal cIDs.

        Idempotent. Acceptable to call after every ``_save_state``;
        steady-state journals stay tiny because TPs / cancels resolve
        most rows quickly.
        """
        alive = self.non_terminal()
        tmp = self.path.with_suffix(self.path.suffix + ".compacting")
        try:
            with tmp.open("w", encoding="utf-8") as fp:
                for cid, rec in alive.items():
                    fp.write(json.dumps(rec, separators=(",", ":")) + "\n")
                fp.flush()
                os.fsync(fp.fileno())
            os.replace(tmp, self.path)
            self.records = alive
        except Exception:
            logger.exception("[intent_journal] compaction failed; journal intact")
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass


__all__ = ["IntentJournal"]
