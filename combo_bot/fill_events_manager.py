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
        # Symbols whose pagination has confirmed stuck (same-ms full
        # page + fromId not advancing or unsupported). Polling for
        # these returns [] until an operator calls clear_stuck — the
        # alternative is silently losing every fill past the stuck
        # millisecond, which would compound forever.
        self._stuck_symbols: set[str] = set()
        # Per-symbol consecutive stuck count. We only escalate to
        # _stuck_symbols after a few attempts so a single anomalous
        # poll doesn't pause a real-money feed.
        self._stuck_count: dict[str, int] = {}
        # How many consecutive stuck polls before we hard-stop.
        self._stuck_escalate_after: int = 3
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
        # Wall-clock at which the live trader started. Trades with
        # timestamp strictly less than this are treated as historical
        # noise (the first fetch_my_trades call commonly returns
        # several days of history when since=None) and skipped.
        # 0 = not yet set (tests poll_interval_ms=0 path).
        self._bot_start_ms: int = 0

    # ------------------------------------------------------------------
    # Outgoing-order bookkeeping (called from LiveTrader._create_order)
    # ------------------------------------------------------------------

    def _trade_belongs_to_us(self, trade: dict) -> bool:
        """True when the trade's order or clientOrderId is registered
        in our outgoing-order tables."""
        order_id = str(trade.get("order") or trade.get("orderId") or "")
        if order_id and order_id in self.order_source:
            return True
        cid = str(
            trade.get("clientOrderId")
            or (trade.get("info") or {}).get("clientOrderId")
            or ""
        )
        return bool(cid and cid in self.order_source)

    def set_bot_start(self, ts_ms: int) -> None:
        """Mark the wall-clock at which the live trader started so
        historical trades (returned by fetch_my_trades when no
        watermark exists yet) are filtered out instead of polluting
        the ledger."""
        self._bot_start_ms = int(ts_ms)

    def register_outgoing(
        self,
        exchange_order_id: str,
        source: OrderSource,
        side: Side,
        reduce_only: bool,
        client_order_id: str = "",
    ) -> None:
        """Record metadata about an order we just placed so the trade
        that fills it can be attributed correctly when it arrives.

        Indexes by BOTH the exchange-assigned id AND (when present)
        the clientOrderId. Some exchanges echo only one of the two
        fields on the resulting trade — registering both means a
        TREND fill can't silently fall back to GRID just because the
        trade's ``order`` field was empty.
        """
        meta = {"side": side, "reduce_only": reduce_only}
        for key in (exchange_order_id, client_order_id):
            if not key or key == "?":
                continue
            self.order_source[str(key)] = source
            self.order_meta[str(key)] = meta
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
            # Persist bot_start_ms so a restart that crashed BEFORE the
            # first fill-poll completed doesn't move the cold-start
            # cutoff forward and silently drop the real fill.
            "bot_start_ms": int(self._bot_start_ms),
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

        bot_start = data.get("bot_start_ms")
        if self._can_int(bot_start):
            self._bot_start_ms = int(bot_start)

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

    def clear_stuck(self, symbol: str) -> None:
        """Operator-callable reset for a symbol parked by the
        pagination-stuck detector. Call after manually verifying that
        the exchange's fill stream is making forward progress again."""
        self._stuck_symbols.discard(symbol)
        self._stuck_count.pop(symbol, None)
        logger.warning("[fill_events] %s manually cleared from stuck set", symbol)

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
        if symbol in self._stuck_symbols:
            # Symbol is parked. Don't query, don't sink. Operator must
            # clear via clear_stuck() — silently continuing here would
            # mean missing every fill past the stuck millisecond, and
            # the missed fills are exactly the ones that drift trend
            # bucket vs exchange aggregate.
            logger.error(
                "[fill_events] %s in STUCK set — skipping poll. "
                "Operator must investigate exchange pagination + call "
                "FillEventManager.clear_stuck(%r)",
                symbol, symbol,
            )
            return []
        # First poll for this symbol always goes through; afterwards
        # throttle by poll_interval_ms so we don't hammer the exchange.
        if symbol in self._last_poll_ms:
            last_poll = self._last_poll_ms[symbol]
            if now_ms - last_poll < self.config.poll_interval_ms:
                return []
        self._last_poll_ms[symbol] = now_ms

        since_ms = self._last_ts_ms.get(symbol)
        # Paginated drain: keep asking for trades until a page returns
        # fewer rows than ``page_size``. Without pagination, a burst of
        # >page_size same-millisecond trades drops everything past the
        # first page — the ``max_ts + 1`` watermark advance jumps over
        # those un-fetched siblings, and we'd never see them.
        #
        # Cursor-stuck detection: when a full page comes back with the
        # max timestamp at or below our cursor (every trade in the
        # page shares the cursor's millisecond), bumping the watermark
        # past that millisecond would skip un-fetched siblings. We:
        #   * try the Binance-style ``fromId`` paginator if possible;
        #   * otherwise leave the watermark stuck on this ms so the
        #     next poll re-fetches it, relying on trade_id dedup to
        #     skip rows we've already seen.
        trades: list[dict] = []
        cursor = since_ms
        max_pages = 16  # bounded to keep tick latency predictable
        cursor_stuck_at_ts: int | None = None
        last_seen_trade_id: str | None = None
        for _ in range(max_pages):
            try:
                params = {}
                if last_seen_trade_id is not None:
                    # Best-effort fromId hint — Binance USDM honors it.
                    params["fromId"] = last_seen_trade_id
                page = await self.exchange.fetch_my_trades(
                    symbol, since=cursor, limit=self.config.page_size,
                    params=params if params else None,
                )
            except TypeError:
                # ccxt fetch_my_trades signature varies — fall back to
                # the no-params form when the broker doesn't accept it.
                try:
                    page = await self.exchange.fetch_my_trades(
                        symbol, since=cursor, limit=self.config.page_size,
                    )
                except Exception:
                    logger.exception(
                        "fetch_my_trades failed for %s since=%s",
                        symbol, cursor,
                    )
                    break
            except Exception:
                logger.exception(
                    "fetch_my_trades failed for %s since=%s", symbol, cursor,
                )
                break
            if not page:
                break
            trades.extend(page)
            if len(page) < self.config.page_size:
                cursor_stuck_at_ts = None
                # Short page = clean drain; any transient stuck
                # counter resets so a future blip doesn't escalate.
                self._stuck_count.pop(symbol, None)
                break
            page_max = max(
                int(p.get("timestamp") or 0) for p in page
            )
            page_max_id = page[-1].get("id")
            if cursor is not None and page_max <= cursor:
                # Same-ms burst — record so the watermark advance below
                # doesn't jump past the unfetched siblings.
                cursor_stuck_at_ts = page_max
                if page_max_id and str(page_max_id) != last_seen_trade_id:
                    # Try the next page via fromId; if it returns
                    # nothing new we'll exit the loop next iteration.
                    last_seen_trade_id = str(page_max_id)
                    continue
                # Same-ms stuck AND fromId didn't help. Bump consecutive
                # stuck counter; after N consecutive stuck polls we
                # park the symbol and ERROR so the operator notices.
                self._stuck_count[symbol] = (
                    self._stuck_count.get(symbol, 0) + 1
                )
                cnt = self._stuck_count[symbol]
                if cnt >= self._stuck_escalate_after:
                    self._stuck_symbols.add(symbol)
                    logger.error(
                        "[fill_events] %s STUCK after %d consecutive "
                        "same-ms full-page polls (ts=%s) — pausing fill "
                        "ingestion for this symbol. Operator must call "
                        "FillEventManager.clear_stuck(%r) once the "
                        "exchange paginates past this millisecond.",
                        symbol, cnt, page_max, symbol,
                    )
                else:
                    logger.warning(
                        "[fill_events] %s pagination stuck at ts=%s "
                        "(attempt %d/%d) — leaving watermark at this "
                        "ms; trade_id dedup will skip seen rows",
                        symbol, page_max, cnt, self._stuck_escalate_after,
                    )
                break
            cursor = page_max
            last_seen_trade_id = (
                str(page_max_id) if page_max_id is not None else None
            )
        if not trades:
            return []

        seen = self._seen_ids.setdefault(symbol, [])
        seen_set = set(seen)
        fresh: list[Fill] = []
        max_ts = since_ms or 0
        # The bot_start filter is ONLY safe for a true cold start:
        # * no persisted watermark for this symbol AND
        # * the trade isn't one we know we sent
        # Otherwise we'd silently drop a pre-restart real fill that
        # happened while the process was down — exchange aggregate
        # would include it, our ledger wouldn't, and the next
        # _refresh_account would attribute it to the grid bucket.
        cold_start_for_symbol = (
            self._bot_start_ms > 0 and symbol not in self._last_ts_ms
        )

        for t in trades:
            tid = str(t.get("id") or "")
            if not tid or tid in seen_set:
                continue
            ts = int(t.get("timestamp") or 0)
            if ts <= 0:
                continue
            # Bot-start guard runs only in the cold-start-no-watermark
            # path and only when the trade ISN'T tied to one of our
            # outgoing orders. A trade we issued must always reach the
            # ledger, even if it pre-dates the wall-clock bot_start_ms
            # (a crash between create_order and the next save_state
            # would leave bot_start_ms newer than the real fill).
            if cold_start_for_symbol and ts < self._bot_start_ms:
                if not self._trade_belongs_to_us(t):
                    seen.append(tid)
                    seen_set.add(tid)
                    if ts > max_ts:
                        max_ts = ts
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

        if cursor_stuck_at_ts is not None:
            # Same-ms burst that couldn't be drained even with fromId.
            # Hold the watermark AT (not past) that millisecond so the
            # next poll re-fetches it; trade_id dedup will skip the
            # rows we already saw and let the un-fetched siblings
            # through. Better to re-fetch a known millisecond than to
            # silently lose orders.
            self._last_ts_ms[symbol] = cursor_stuck_at_ts
        elif max_ts > (since_ms or 0):
            # Normal advance: +1ms past the latest so we don't re-fetch
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

        # Try the exchange's order id first; fall back to clientOrderId
        # echoed on the trade or its info subobject. Some exchanges
        # (notably when the order was fully filled at create time)
        # surface only the cOID on subsequent trade-history rows.
        order_id = str(trade.get("order") or trade.get("orderId") or "")
        cid = str(
            trade.get("clientOrderId")
            or (trade.get("info") or {}).get("clientOrderId")
            or ""
        )
        source = (
            self.order_source.get(order_id)
            or (self.order_source.get(cid) if cid else None)
            or OrderSource.GRID
        )
        meta = (
            self.order_meta.get(order_id)
            or (self.order_meta.get(cid) if cid else None)
            or {}
        )
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
            exchange_order_id=order_id,
            client_order_id=cid,
            trade_id=str(trade.get("id") or ""),
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
