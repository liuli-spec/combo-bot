"""Live fill-event manager.

Bridges the exchange's trade history to the bot's Fill domain type so
:class:`combo_bot.protections.ProtectionManager`, :class:`combo_bot
.sizing.KellySizer`, and :class:`combo_bot.types.AccountState.add_realized_pnl`
keep working in live the way they do in backtest.

Without this layer the live trader's per-tick ``protections.update(fills,
...)`` call gets an empty list every tick — silently disabling Stage 7
protections, Stage 9 Kelly throttling, and Stage 4 per-source drawdown
attribution. Backtest looked fine but live silently dropped half the
risk machinery.

Design
------

* Lightweight per-symbol polling via ``exchange.fetch_my_trades(symbol,
  since=watermark)``. We never assume a websocket user-data stream
  because not every supported exchange exposes one through ccxt.
* Dedup by ``trade.id`` — fetch_my_trades is allowed to overlap the
  prior watermark, so the same trade can appear in multiple polls.
* Source attribution via a sticky ``order_id → OrderSource`` map that
  the live trader populates on every successful ``create_order``.
  Trades that can't be attributed default to ``OrderSource.GRID`` —
  the safer fallback because grid is the default bucket for unknown
  fills (see ``SymbolState.bucket``).
* Realized P&L extraction prefers ``trade.info.realizedPnl`` (Binance
  USDM convention). If absent we leave realized_pnl at 0.0 and let the
  position-diff path in ``_refresh_account`` book it later — better
  than guessing.

Polling is bounded by ``poll_interval_ms`` so we don't hammer the
exchange every tick. With a 60s live loop and 30s polling, every other
tick will skip the fetch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from combo_bot.types import Fill, OrderSource, Side


logger = logging.getLogger(__name__)


@dataclass
class FillEventManagerConfig:
    # Per-symbol watermark advance. The exchange might page or
    # backfill; this is a per-poll page size cap (ccxt's `limit`).
    page_size: int = 200
    # Don't poll more often than this — fetch_my_trades is one of the
    # heavier rate-limit buckets on most exchanges.
    poll_interval_ms: int = 30_000
    # Where the bookkeeping ledger of seen trade IDs lives, capped so a
    # long-running bot doesn't grow unbounded.
    dedup_capacity: int = 4_096


class FillEventManager:
    """Polls exchange trade history and routes new fills downstream.

    The live trader owns one of these and calls :meth:`poll` per tick.
    Provide a callback that receives the freshly-discovered
    :class:`Fill` list — typically a lambda that fans out to
    ``protections.update``, ``kelly_sizer.record_fills``, and
    ``account.add_realized_pnl``.
    """

    def __init__(
        self,
        exchange: Any,
        config: FillEventManagerConfig | None = None,
    ) -> None:
        self.exchange = exchange
        self.config = config or FillEventManagerConfig()
        # Per-symbol watermark: highest trade timestamp seen.
        self._last_ts_ms: dict[str, int] = {}
        # Per-symbol last poll wallclock — gates poll_interval_ms.
        self._last_poll_ms: dict[str, int] = {}
        # Trade-id dedup buffer per symbol (FIFO with capacity cap).
        self._seen_ids: dict[str, list[str]] = {}
        # source attribution map. Live trader writes here on successful
        # create_order; we read here on fill arrival.
        self.order_source: dict[str, OrderSource] = {}
        # Same dict but for side/reduce_only metadata — used to decide
        # which combo-bot Side the trade represents (a buy can be a
        # LONG entry OR a SHORT close).
        self.order_meta: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Outgoing-order bookkeeping (called from LiveTrader._create_order)
    # ------------------------------------------------------------------

    def register_outgoing(
        self,
        exchange_order_id: str,
        source: OrderSource,
        side: Side,
        reduce_only: bool,
    ) -> None:
        """Record metadata about an order we just placed so the trade
        that fills it can be attributed correctly when it arrives."""
        if not exchange_order_id or exchange_order_id == "?":
            return
        self.order_source[exchange_order_id] = source
        self.order_meta[exchange_order_id] = {
            "side": side,
            "reduce_only": reduce_only,
        }
        # Coarse cap: drop the oldest 25% when we exceed cap to keep
        # the map bounded. Trade history beyond this isn't useful for
        # near-term attribution anyway.
        if len(self.order_source) > 4 * self.config.dedup_capacity:
            keys = list(self.order_source.keys())
            for k in keys[: len(keys) // 4]:
                self.order_source.pop(k, None)
                self.order_meta.pop(k, None)

    def snapshot(self) -> dict[str, Any]:
        """Serializable watermark/dedup/order-attribution state.

        LiveTrader persists this inside its state file so restarts do not
        replay old trades or forget that an exchange order belonged to the
        TREND bucket.
        """
        return {
            "last_ts_ms": dict(self._last_ts_ms),
            "seen_ids": {sym: list(ids) for sym, ids in self._seen_ids.items()},
            "order_source": {
                oid: source.value for oid, source in self.order_source.items()
            },
            "order_meta": {
                oid: {
                    "side": (
                        meta.get("side").value
                        if isinstance(meta.get("side"), Side)
                        else str(meta.get("side", ""))
                    ),
                    "reduce_only": bool(meta.get("reduce_only", False)),
                }
                for oid, meta in self.order_meta.items()
            },
        }

    def load_snapshot(self, data: dict[str, Any] | None) -> None:
        """Restore state produced by :meth:`snapshot`.

        Bad or stale entries are ignored rather than crashing live startup.
        """
        if not isinstance(data, dict):
            return
        raw_last = data.get("last_ts_ms") or {}
        if isinstance(raw_last, dict):
            self._last_ts_ms = {
                str(sym): int(ts)
                for sym, ts in raw_last.items()
                if self._can_int(ts)
            }

        raw_seen = data.get("seen_ids") or {}
        if isinstance(raw_seen, dict):
            cap = self.config.dedup_capacity
            self._seen_ids = {
                str(sym): [str(x) for x in list(ids)[-cap:]]
                for sym, ids in raw_seen.items()
                if isinstance(ids, list)
            }

        raw_sources = data.get("order_source") or {}
        if isinstance(raw_sources, dict):
            self.order_source = {}
            for oid, raw in raw_sources.items():
                try:
                    self.order_source[str(oid)] = OrderSource(raw)
                except ValueError:
                    continue

        raw_meta = data.get("order_meta") or {}
        if isinstance(raw_meta, dict):
            self.order_meta = {}
            for oid, meta in raw_meta.items():
                if not isinstance(meta, dict):
                    continue
                try:
                    side = Side(meta.get("side", "long"))
                except ValueError:
                    side = Side.LONG
                self.order_meta[str(oid)] = {
                    "side": side,
                    "reduce_only": bool(meta.get("reduce_only", False)),
                }

    @staticmethod
    def _can_int(value: Any) -> bool:
        try:
            int(value)
            return True
        except (TypeError, ValueError):
            return False

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def poll(
        self,
        symbol: str,
        now_ms: int,
        sink: Callable[[list[Fill]], None],
    ) -> list[Fill]:
        """Fetch trades for ``symbol`` since the last watermark and emit
        new :class:`Fill` records to ``sink``. Returns the same list so
        the caller can also branch on it (e.g. log).
        """
        # First poll for this symbol always goes through; afterwards
        # throttle by poll_interval_ms so we don't hammer the exchange.
        if symbol in self._last_poll_ms:
            last_poll = self._last_poll_ms[symbol]
            if now_ms - last_poll < self.config.poll_interval_ms:
                return []
        self._last_poll_ms[symbol] = now_ms

        since_ms = self._last_ts_ms.get(symbol)
        try:
            trades = await self.exchange.fetch_my_trades(
                symbol, since=since_ms, limit=self.config.page_size,
            )
        except Exception:
            logger.exception(
                "fetch_my_trades failed for %s since=%s", symbol, since_ms,
            )
            return []
        if not trades:
            return []

        seen = self._seen_ids.setdefault(symbol, [])
        seen_set = set(seen)
        fresh: list[Fill] = []
        max_ts = since_ms or 0
        for t in trades:
            tid = str(t.get("id") or "")
            if not tid or tid in seen_set:
                continue
            ts = int(t.get("timestamp") or 0)
            if ts <= 0:
                continue
            fill = self._to_fill(symbol, t)
            if fill is None:
                continue
            fresh.append(fill)
            seen.append(tid)
            seen_set.add(tid)
            if ts > max_ts:
                max_ts = ts

        # Trim dedup buffer to capacity (drop oldest).
        cap = self.config.dedup_capacity
        if len(seen) > cap:
            drop = len(seen) - cap
            del seen[:drop]

        if max_ts > (since_ms or 0):
            # Advance by 1ms past the latest so we don't re-fetch
            # boundary trades next poll (Binance returns trades with
            # ``timestamp >= since``, inclusive).
            self._last_ts_ms[symbol] = max_ts + 1

        if fresh:
            sink(fresh)
        return fresh

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def _to_fill(self, symbol: str, trade: dict) -> Fill | None:
        try:
            ts = int(trade["timestamp"])
            price = float(trade.get("price", 0) or 0)
            qty = float(trade.get("amount", 0) or 0)
        except (KeyError, TypeError, ValueError):
            return None
        if price <= 0 or qty <= 0:
            return None

        fee_info = trade.get("fee") or {}
        try:
            fee = float(fee_info.get("cost", 0) or 0)
        except (TypeError, ValueError):
            fee = 0.0

        order_id = str(trade.get("order") or trade.get("orderId") or "")
        source = self.order_source.get(order_id, OrderSource.GRID)
        meta = self.order_meta.get(order_id, {})
        side = self._infer_side(trade, meta)

        info = trade.get("info") or {}
        try:
            realized_pnl = float(info.get("realizedPnl", 0) or 0)
        except (TypeError, ValueError):
            realized_pnl = 0.0

        return Fill(
            timestamp=ts,
            symbol=symbol,
            side=side,
            price=price,
            qty=qty,
            fee=fee,
            realized_pnl=realized_pnl,
            source=source,
            reduce_only=bool(meta.get("reduce_only", False)),
        )

    @staticmethod
    def _infer_side(trade: dict, meta: dict) -> Side:
        """Map ccxt's buy/sell + our reduce_only metadata to combo_bot Side.

        * buy + entry  → LONG
        * sell + close → LONG (closing a long)
        * sell + entry → SHORT
        * buy + close  → SHORT (closing a short)

        When metadata is missing we assume entry semantics (the common
        case for first-fill attribution).
        """
        ex_side = str(trade.get("side", "")).lower()
        reduce_only = bool(meta.get("reduce_only", False))
        if ex_side == "buy":
            return Side.SHORT if reduce_only else Side.LONG
        return Side.LONG if reduce_only else Side.SHORT


__all__ = ["FillEventManager", "FillEventManagerConfig"]
