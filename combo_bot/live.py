from __future__ import annotations
import asyncio
import json
import logging
import time
from collections import deque
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from combo_bot.types import (
    AccountState, Candle, ExchangeParams, Fill, Order, OrderSource,
    Position, Side, SymbolState, TradingMode,
)
from combo_bot.grid_engine import GridConfig, GridEngine, calc_wallet_exposure
from combo_bot.trend_signal import TrendConfig, TrendEngine
from combo_bot.merger import MergerConfig, DecisionMerger
from combo_bot.risk import RiskConfig, RiskManager, RiskTier
from combo_bot.strategy import DefaultStrategy, IStrategy, StrategyRunner, TradeContext
from combo_bot.data_provider import DataProvider
from combo_bot.regime import RegimeArbiter, RegimeArbiterConfig, read_strategy_signals
from combo_bot.protections import IProtection, ProtectionManager
from combo_bot.sizing import KellySizer
from combo_bot.correlation import CorrelationGate
from combo_bot.vol_target import VolTargetSizer
from combo_bot.fill_events_manager import (
    FillEventManager, FillEventManagerConfig,
)
from combo_bot.intent_journal import IntentJournal
from combo_bot.types import RegimeView

logger = logging.getLogger(__name__)


@dataclass
class LiveConfig:
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT:USDT"])
    leverage: int = 5
    margin_mode: str = "cross"
    dry_run: bool = True
    loop_interval_seconds: float = 60.0
    max_orders_per_batch: int = 5
    order_match_tolerance_pct: float = 0.002
    state_file: str = "live_state.json"
    # Candle cadence the live trader fetches. Must match the timeframe
    # the backtest was tuned for — using a 1m default in live while the
    # backtest validated at 1h would give you a different trend signal,
    # volatility EMA, and Sharpe denominator. Pair with
    # ``bar_interval_minutes`` so VolatilityState / VolTargetSizer
    # annualize correctly.
    candle_timeframe: str = "1m"
    bar_interval_minutes: float = 1.0
    grid: GridConfig = field(default_factory=GridConfig)
    trend: TrendConfig = field(default_factory=TrendConfig)
    merger: MergerConfig = field(default_factory=MergerConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    regime: RegimeArbiterConfig = field(default_factory=RegimeArbiterConfig)


class LiveTrader:
    def __init__(
        self,
        config: LiveConfig,
        exchange,
        strategy: IStrategy | None = None,
        protections: list[IProtection] | None = None,
        kelly_sizer: KellySizer | None = None,
        correlation_gate: CorrelationGate | None = None,
        vol_target_sizer: VolTargetSizer | None = None,
        fill_events_config: FillEventManagerConfig | None = None,
    ):
        self.config = config
        self.exchange = exchange
        self.grid = GridEngine(config.grid)
        self.trend = TrendEngine(config.trend)
        self.merger = DecisionMerger(config.merger)
        self.risk = RiskManager(config.risk)
        self.regime_arbiter = RegimeArbiter(config.regime)
        self.strategy: IStrategy = strategy or DefaultStrategy()
        self.strategy_runner = StrategyRunner(self.strategy)
        self.data_provider = DataProvider(max_rows=1000)
        self.protections = ProtectionManager(protections or [])
        # Fill event manager — polls fetch_my_trades, dedups by trade
        # ID, attributes source via the order_id we record at create
        # time, and feeds the result to protections / kelly_sizer /
        # account.add_realized_pnl. Without this layer the Stage 7-11
        # bookkeeping silently no-ops in live (backtest still works).
        self.fill_events = FillEventManager(exchange, fill_events_config)
        # Stage 9 — fractional-Kelly trend overlay throttle. Now live
        # too: fills surfaced by FillEventManager are routed in.
        self.kelly_sizer = kelly_sizer
        # Stage 10 — optional cross-symbol correlation gate.
        self.correlation_gate = correlation_gate
        # Stage 11 — optional portfolio-level vol-targeting sizer.
        self.vol_target_sizer = vol_target_sizer
        self.account = AccountState()
        self.exchange_params: dict[str, ExchangeParams] = {}
        self._running = False
        self._open_orders: dict[str, list[dict]] = {}
        self._funding_rates: dict[str, float] = {}
        # Last candle timestamp fed to the trend engine / data provider
        # per symbol. Used by _refresh_candles to dedupe so backtest and
        # live tick semantics agree: each new bar's close is consumed
        # EXACTLY ONCE by trend.update / data_provider.append.
        self._last_candle_ts: dict[str, int] = {}
        # ── execution guards (modelled on passivbot executor / reconciler) ──
        # Cap scales with symbol count so busy multi-symbol books don't
        # evict legitimate entries.
        guard_cap = max(128, len(config.symbols) * 16)
        # Throttle re-creating the same order within a short window.
        # Tuple: (timestamp_ms, desired_identity_tuple). The identity
        # tuple is the full (symbol, side, source, reduce_only, price,
        # qty) — keying only on (symbol, price, qty) used to cross-block
        # semantically different orders, e.g. a LONG-close sell and a
        # SHORT-entry sell with coincident price/qty would dedup each
        # other even though one was reducing the long bucket and the
        # other opening short exposure.
        self._recent_creates: deque[tuple[float, tuple]] = deque(maxlen=guard_cap)
        # Persistent client-order-id attribution. Key is the stable
        # ``desired identity`` tuple (symbol, side, source, reduce_only,
        # rounded price, rounded qty); value is (cid, last_seen_ms).
        # Without this, every tick generates a fresh UUID for what is
        # logically the SAME desired entry, so reconcile_orders can
        # never match the exchange's existing open order by cOID and
        # always falls back to fuzzy (price, qty) matching. Caching
        # the cOID across ticks makes the cOID actually load-bearing
        # for live reconcile, not just decorative on a single send.
        self._cid_by_desired: dict[tuple, tuple[str, float]] = {}
        # cOID cache lifetime is INDEPENDENT of the dedup window. The
        # dedup window is "two loop intervals"; tying cOID cache to it
        # meant a 5-minute pause (network hiccup, manual restart)
        # silently invalidated every cached cOID. Then reconcile saw
        # the exchange's open orders as foreign (their clientOrderId
        # no longer matched anything in our cache) and cancelled them.
        # Keep cOIDs in cache for 24h — matches the cid_cache load
        # cutoff in _load_state.
        self._cid_cache_ttl_ms: int = 24 * 60 * 60 * 1_000
        # Pending trend overlay entries: between ``create_order`` ack
        # and the fill arriving via FillEventManager, the trend bucket
        # is still empty locally but the exchange has accepted our
        # market order. Without this ledger, ``_emit_trend_overlay``
        # would see ``trend_long.is_open == False`` on the very next
        # tick and emit ANOTHER market overlay entry — double the
        # intended exposure on a strong-regime tick. Cleared when the
        # fill event lands or the order is rejected/cancelled; TTL
        # expiry MOVES the entry to ``_unknown_overlay`` (does not
        # clear it) so a stuck fill stream cannot silently unlock
        # re-entry.
        self._pending_overlay: dict[tuple[str, Side], float] = {}
        # Unknown-state overlay slots: pending entries whose TTL ran
        # out without a confirming fill or reject. These continue to
        # block ``_emit_trend_overlay`` and trigger a reconcile-time
        # ``_resolve_unknowns`` pass that consults the exchange for
        # ground truth. Round-13 P1.a.
        self._unknown_overlay: dict[tuple[str, Side], float] = {}
        # Durable intent journal — see combo_bot/intent_journal.py.
        # Path is derived from state_file so different deployments
        # (testnet vs real, dry vs live) stay isolated.
        journal_path = (
            Path(config.state_file).with_suffix(".intent_journal.jsonl")
        )
        self.intent_journal = IntentJournal(journal_path)
        # Five minutes is comfortably longer than poll_interval_ms
        # (30s) so a normal fill flow will always clear the pending
        # before this fires, but short enough that a truly stuck
        # exchange / fill stream can't permanently block re-entry.
        self._pending_overlay_ttl_ms: int = 5 * 60 * 1_000
        # Throttle re-cancelling the same order.
        self._recent_cancels: deque[tuple[float, str]] = deque(maxlen=guard_cap)
        # (symbol, side) tuples whose grid-bucket size just changed — defer
        # new entries on that side one cycle. Side-scoped so a long fill
        # doesn't accidentally suppress fresh short entries.
        self._state_change_keys: set[tuple[str, Side]] = set()
        # Max age for recent-order dedup (ms). Must exceed two loop
        # intervals or dedup never catches anything between ticks.
        self._recent_order_window_ms: int = max(
            15_000, int(config.loop_interval_seconds * 1000 * 2),
        )

    async def start(self):
        logger.info("Starting live trader (dry_run=%s)", self.config.dry_run)
        await self._init_exchange()
        await self._load_state()
        # Replay the durable intent journal — this is what closes the
        # crash window between create_order ack and _save_state. Any
        # non-terminal cID gets its attribution restored to
        # FillEventManager and (if TREND non-reduce) its pending-overlay
        # slot. The eventual fill — which may have already happened on
        # the exchange before we restarted — gets routed correctly.
        self._replay_intent_journal()
        # Only stamp bot_start_ms if _load_state didn't restore one.
        # Overwriting it on every restart would invalidate the "real
        # fill from before the restart" guard inside FillEventManager.
        if self.fill_events._bot_start_ms <= 0:
            self.fill_events.set_bot_start(self._now_ms())
        self._running = True

        while self._running:
            try:
                await self._tick()
            except Exception:
                logger.exception("Error in tick loop")
            await asyncio.sleep(self.config.loop_interval_seconds)

    async def stop(self):
        self._running = False
        await self._save_state()
        logger.info("Trader stopped")

    async def _init_exchange(self):
        await self.exchange.load_markets()
        for symbol in self.config.symbols:
            market = self.exchange.market(symbol)
            self.exchange_params[symbol] = ExchangeParams(
                qty_step=float(market.get("precision", {}).get("amount", 0.001)),
                price_step=float(market.get("precision", {}).get("price", 0.01)),
                min_qty=float(market.get("limits", {}).get("amount", {}).get("min", 0.001)),
                min_cost=float(market.get("limits", {}).get("cost", {}).get("min", 5.0)),
                c_mult=float(market.get("contractSize", 1.0)),
                maker_fee=float(market.get("maker", 0.0002)),
                taker_fee=float(market.get("taker", 0.0005)),
            )
            ep = self.exchange_params[symbol]
            self.account.symbols[symbol] = SymbolState(symbol=symbol, c_mult=ep.c_mult)

            if not self.config.dry_run:
                try:
                    await self.exchange.set_leverage(self.config.leverage, symbol)
                    if self.config.margin_mode == "cross":
                        await self.exchange.set_margin_mode("cross", symbol)
                    else:
                        await self.exchange.set_margin_mode("isolated", symbol)
                except Exception:
                    logger.warning("Could not set leverage/margin for %s", symbol)

    async def _tick(self):
        # Pull fills first so source-isolated buckets (especially TREND)
        # are updated before aggregate exchange positions are split into
        # grid-vs-trend in _refresh_account.
        await self._refresh_fills()
        await self._refresh_account()
        await self._refresh_candles()

        self.account.update_equity()
        # Feed the vol-target sizer's rolling equity sample so live
        # actually warms up the same way backtest does. Without this
        # call the sizer stayed at cold-start (scale=1.0) forever in
        # live — vol-targeting was effectively a no-op once turned on,
        # silently disabling the Stage 11 portfolio-vol throttle.
        if self.vol_target_sizer is not None:
            self.vol_target_sizer.record_equity(self.account.equity)
        all_desired_orders: list[Order] = []

        for symbol in self.config.symbols:
            ss = self.account.symbols[symbol]
            ep = self.exchange_params[symbol]
            signal = self.trend.compute(symbol)
            price = ss.last_price

            strat_enter_long, strat_enter_short, strat_exit_long, strat_exit_short = (
                read_strategy_signals(self.data_provider, symbol)
            )
            fr = self._funding_rates.get(symbol, 0.0)
            regime_view = self.regime_arbiter.compute(
                signal,
                funding_rate=fr,
                strategy_exit_long=strat_exit_long,
                strategy_exit_short=strat_exit_short,
                strategy_enter_long=strat_enter_long,
                strategy_enter_short=strat_enter_short,
            )

            we_long = calc_wallet_exposure(
                self.account.balance, abs(ss.position_long.size),
                ss.position_long.entry_price, ep.c_mult,
            ) if ss.position_long.is_open else 0.0

            grid_long = self.grid.compute_orders(
                symbol, Side.LONG, ss.position_long, ss.ema, ss.volatility,
                self.account.balance, we_long, ep, regime_view.long_mode,
                mark_price=price,
                close_markup_multiplier=regime_view.close_aggressiveness,
            )

            we_short = calc_wallet_exposure(
                self.account.balance, abs(ss.position_short.size),
                ss.position_short.entry_price, ep.c_mult,
            ) if ss.position_short.is_open else 0.0

            grid_short = self.grid.compute_orders(
                symbol, Side.SHORT, ss.position_short, ss.ema, ss.volatility,
                self.account.balance, we_short, ep, regime_view.short_mode,
                mark_price=price,
                close_markup_multiplier=regime_view.close_aggressiveness,
            )

            # Strategy layer — gate the engine's proposed orders and inject
            # any forced exits / DCA the user's callbacks decide on.
            now_ms = self._now_ms()
            last_candle = self._latest_candle(symbol, price, now_ms)
            ctx_long = TradeContext(
                symbol=symbol, side=Side.LONG, position=ss.position_long,
                account=self.account, candle=last_candle, signal=signal,
                current_time_ms=now_ms, exchange_params=ep,
            )
            ctx_short = TradeContext(
                symbol=symbol, side=Side.SHORT, position=ss.position_short,
                account=self.account, candle=last_candle, signal=signal,
                current_time_ms=now_ms, exchange_params=ep,
            )
            grid_long = self.strategy_runner.filter_exits(
                self.strategy_runner.filter_entries(grid_long, ctx_long), ctx_long,
            )
            grid_short = self.strategy_runner.filter_exits(
                self.strategy_runner.filter_entries(grid_short, ctx_short), ctx_short,
            )

            # Stage 8 trailing re-entries (passivbot-style). No-op when
            # entry_trailing_threshold_pct / retracement_pct are 0.
            trailing_entries: list[Order] = []
            trail_long = self.grid.compute_trailing_entry(
                symbol, Side.LONG, ss.position_long, ss.trailing_long,
                self.account.balance, we_long, ep, price, regime_view.long_mode,
            )
            if trail_long is not None:
                trailing_entries.extend(
                    self.strategy_runner.filter_entries([trail_long], ctx_long)
                )
            trail_short = self.grid.compute_trailing_entry(
                symbol, Side.SHORT, ss.position_short, ss.trailing_short,
                self.account.balance, we_short, ep, price, regime_view.short_mode,
            )
            if trail_short is not None:
                trailing_entries.extend(
                    self.strategy_runner.filter_entries([trail_short], ctx_short)
                )

            trend_entries = self._emit_trend_overlay(symbol, regime_view, price, ep)
            # Run trend overlay entries through the strategy veto /
            # price / sizing pipeline — see the parallel comment in
            # Backtester. Context's position must be the TREND bucket
            # so confirm_trade_entry sees the right state.
            if trend_entries:
                overlay_side = trend_entries[0].side
                overlay_pos = (
                    ss.trend_long if overlay_side == Side.LONG
                    else ss.trend_short
                )
                ctx_overlay = TradeContext(
                    symbol=symbol, side=overlay_side, position=overlay_pos,
                    account=self.account, candle=last_candle, signal=signal,
                    current_time_ms=now_ms, exchange_params=ep,
                    source=OrderSource.TREND,
                )
                trend_entries = self.strategy_runner.filter_entries(
                    trend_entries, ctx_overlay,
                )
            trend_exits_l = self.merger.generate_trend_exit_orders(symbol, ss.trend_long, Side.LONG, price, ep)
            trend_exits_s = self.merger.generate_trend_exit_orders(symbol, ss.trend_short, Side.SHORT, price, ep)

            strategy_orders: list[Order] = []

            # ── grid-bucket strategy callbacks ────────────────────
            for ctx, pos in ((ctx_long, ss.position_long), (ctx_short, ss.position_short)):
                if not pos.is_open:
                    continue
                fx = self.strategy_runner.check_custom_exit(ctx)
                if fx is not None:
                    strategy_orders.append(fx)
                adj = self.strategy_runner.check_position_adjustment(ctx)
                if adj is not None:
                    strategy_orders.append(adj)

            # ── trend-bucket strategy callbacks ───────────────────
            # Trend positions are also strategy-managed: custom_stoploss,
            # custom_exit, and adjust_trade_position can all override
            # the fixed-% SL/TP from MergerConfig.
            for side, trend_pos in (
                (Side.LONG, ss.trend_long),
                (Side.SHORT, ss.trend_short),
            ):
                if not trend_pos.is_open:
                    continue
                ctx_trend = TradeContext(
                    symbol=symbol, side=side, position=trend_pos,
                    account=self.account, candle=last_candle, signal=signal,
                    current_time_ms=now_ms, exchange_params=ep,
                    source=OrderSource.TREND,
                )
                fx = self.strategy_runner.check_custom_exit(ctx_trend)
                if fx is not None:
                    strategy_orders.append(fx)
                adj = self.strategy_runner.check_position_adjustment(ctx_trend)
                if adj is not None:
                    strategy_orders.append(adj)

            ss.mode_long = regime_view.long_mode
            ss.mode_short = regime_view.short_mode

            all_desired_orders.extend(
                grid_long + grid_short + trailing_entries + trend_entries
                + trend_exits_l + trend_exits_s + strategy_orders
            )

        tick_ms = self._now_ms()
        # Stage 10: feed the correlation tracker once per tick from
        # each symbol's last known close (set by _refresh_candles).
        if self.correlation_gate is not None:
            self.correlation_gate.update_prices(
                (s, self.account.symbols[s].last_price)
                for s in self.config.symbols
                if self.account.symbols[s].last_price > 0
            )

        # Stage 7 protections - drop orders for symbols/sides/sources
        # currently locked. Live doesn't see fills directly (the
        # exchange does), so protections.update() is a no-op here for
        # now; protections that depend on fills require the reconcile
        # path to surface them — TODO once we wire fill events.
        all_desired_orders = self.protections.filter_orders(all_desired_orders, tick_ms)

        # Stage 10 correlation gate — runs after protections, before risk.
        if self.correlation_gate is not None:
            all_desired_orders = self.correlation_gate.filter_orders(
                all_desired_orders, self.account,
            )

        # Stage 11 portfolio-level vol-targeting — same as Backtester.
        if self.vol_target_sizer is not None:
            all_desired_orders = self.vol_target_sizer.filter_orders(all_desired_orders)

        # Stage 5 unstuck — same call as Backtester. Wall-clock time
        # drives both the 24h loss budget and any later staleness checks.
        unstuck_orders = self.risk.compute_unstuck_orders(
            self.account,
            grid_wallet_exposure_limit=self.config.grid.wallet_exposure_limit,
            now_ms=tick_ms,
            exchange_params=self.exchange_params,
        )
        all_desired_orders.extend(unstuck_orders)

        all_desired_orders = self.risk.filter_orders(
            all_desired_orders, self.account, tick_ms,
            exchange_params=self.exchange_params,
        )

        await self._reconcile_orders(all_desired_orders)
        # Round-13 P1.a: try to resolve any unknown-state overlays via
        # the exchange before persisting state. Successful resolves
        # are journaled (terminal) so compaction can evict them.
        await self._resolve_unknowns()
        await self._save_state()

        logger.info(
            "tick | balance=%.2f equity=%.2f dd=%.4f orders=%d",
            self.account.balance, self.account.equity,
            self.account.drawdown, len(all_desired_orders),
        )

    async def _refresh_account(self):
        try:
            balance = await self.exchange.fetch_balance({"type": "future"})
            self.account.balance = self._extract_wallet_balance(balance, "USDT")

            positions = await self.exchange.fetch_positions(self.config.symbols)
            # Track every (symbol, side) the exchange actually returned
            # so we can zero the buckets that DIDN'T appear. Without
            # this pass, a closed position (the exchange stops
            # reporting that side) would leave the local bucket as a
            # ghost — overlay decisions, source_drawdown_pct, and
            # close-grid markups all keep referencing a position that
            # no longer exists.
            seen_sides: set[tuple[str, str]] = set()
            for p in positions:
                symbol = p["symbol"]
                if symbol not in self.account.symbols:
                    continue
                ss = self.account.symbols[symbol]
                side = p.get("side", "")
                size = abs(float(p.get("contracts", 0) or 0))
                entry = float(p.get("entryPrice", 0) or 0)
                # Zero-size rows can appear when an exchange "echoes"
                # a side without an open position. Don't mark the side
                # as seen — it's effectively absent.
                if size > 1e-12 and side in ("long", "short"):
                    seen_sides.add((symbol, side))
                # Limitation: the exchange reports a single aggregate
                # position per side, so we attribute the delta between the
                # reported total and our tracked trend bucket to the grid
                # bucket. This means on a cold start the entire position
                # goes into the grid bucket — trend bookkeeping is rebuilt
                # only by subsequent overlay entries.
                # Compare the previously-tracked GRID-bucket size against
                # the freshly computed one. A drift here means either a
                # fill we didn't know about or a manual position change;
                # either way we quiesce that side for one reconcile cycle.
                #
                # Preserve best_price across the rebuild so trailing
                # stops keep their high-water-mark — wiping best_price
                # to 0 every refresh broke trailing semantics (next
                # update_best_price would re-anchor to entry_price).
                if side == "long":
                    self._rebuild_bucket(
                        symbol=symbol, side=Side.LONG,
                        exchange_size=size, exchange_entry=entry,
                        ss=ss,
                    )
                elif side == "short":
                    self._rebuild_bucket(
                        symbol=symbol, side=Side.SHORT,
                        exchange_size=size, exchange_entry=entry,
                        ss=ss,
                    )
                ss.last_price = float(p.get("markPrice", 0) or ss.last_price)
            # Zero out grid buckets the exchange DIDN'T mention this
            # poll. The trend bucket is fill-driven (FillEventManager),
            # so we leave it alone here — clamp_trend in _rebuild_bucket
            # handles the trend-side divergence path. The grid side is
            # the one that goes "ghost" otherwise: the exchange stops
            # echoing it after a close and our bucket sticks around.
            # The exchange is the source of truth for total position
            # per side. If a side is missing, both buckets on that side
            # must clear — keeping trend_* alive while grid_* clears
            # made the bot think a trend overlay was still open, so the
            # next overlay decision saw `existing.is_open == True` and
            # refused to enter again.
            for sym, ss in self.account.symbols.items():
                # Only mark state-change when we actually CLEARED a live
                # bucket. The old unconditional `add((sym, side))` for
                # every missing side meant a flat account (first tick,
                # no open positions) had both sides quiesced on every
                # _refresh_account → _reconcile_orders dropped ALL
                # non-reduce_only entries → first live order never went
                # out. Locking entries should require evidence that
                # something *changed*, not just absence of a position.
                if (sym, "long") not in seen_sides:
                    cleared = False
                    if ss.position_long.is_open:
                        ss.position_long = Position()
                        cleared = True
                    if ss.trend_long.is_open:
                        ss.trend_long = Position()
                        cleared = True
                    if cleared:
                        logger.info(
                            "[reconcile] %s long: closed on exchange "
                            "— cleared grid + trend buckets",
                            sym,
                        )
                        self._state_change_keys.add((sym, Side.LONG))
                if (sym, "short") not in seen_sides:
                    cleared = False
                    if ss.position_short.is_open:
                        ss.position_short = Position()
                        cleared = True
                    if ss.trend_short.is_open:
                        ss.trend_short = Position()
                        cleared = True
                    if cleared:
                        logger.info(
                            "[reconcile] %s short: closed on exchange "
                            "— cleared grid + trend buckets",
                            sym,
                        )
                        self._state_change_keys.add((sym, Side.SHORT))
        except Exception:
            logger.exception("Failed to refresh account")

        # Funding rates feed into the regime arbiter's overlay veto. Best-effort:
        # if the exchange doesn't support it or the call fails, keep zeros so
        # the overlay never gets vetoed for funding reasons.
        for symbol in self.config.symbols:
            try:
                fr = await self.exchange.fetch_funding_rate(symbol)
                self._funding_rates[symbol] = float(fr.get("fundingRate", 0) or 0)
            except Exception:
                self._funding_rates.setdefault(symbol, 0.0)

    def _rebuild_bucket(
        self,
        *,
        symbol: str,
        side: Side,
        exchange_size: float,
        exchange_entry: float,
        ss: SymbolState,
    ) -> None:
        """Reconstruct grid bucket from an aggregate exchange snapshot.

        The exchange reports a SINGLE aggregate (size, avg_entry_price)
        per side. We track grid + trend buckets independently. Given:

            exchange_total_notional = exchange_size * exchange_entry
            trend_notional          = trend_size * trend_entry
            grid_notional           = grid_size * grid_entry

        and ``exchange_total_notional = grid_notional + trend_notional``,
        we solve for ``grid_entry``:

            grid_entry = (exchange_total_notional - trend_notional) / grid_size

        Old code assigned ``entry_price = exchange_entry`` to the grid
        bucket — that's the AGGREGATE avg entry across both buckets,
        which is wrong whenever the trend bucket entered at a
        meaningfully different price. TP markups then anchored on the
        wrong reference, so live close orders fired at prices that did
        not match the backtest's view of "+1%/+2% from entry".

        Also detects divergence: if ``trend_tracked > exchange_size``,
        our trend bookkeeping has drifted (a fill we never saw, or a
        manual close on the exchange). Log a warning and clamp the
        trend bucket — keeping it would make the grid bucket reconstruct
        as negative size.
        """
        trend_bucket = ss.trend_long if side == Side.LONG else ss.trend_short
        grid_bucket = ss.position_long if side == Side.LONG else ss.position_short
        trend_size = abs(trend_bucket.size)
        trend_entry = trend_bucket.entry_price if trend_bucket.is_open else 0.0

        # Divergence detection: trend bigger than total → our tracking
        # is broken. Clamp trend, warn, defer entries one tick.
        if trend_size > exchange_size + 1e-9:
            logger.warning(
                "[reconcile] %s %s: tracked trend %.6f exceeds exchange "
                "total %.6f — clamping trend bucket; investigate missed "
                "fill or manual close",
                symbol, side.value, trend_size, exchange_size,
            )
            trend_size = exchange_size
            trend_bucket.size = exchange_size
            # Don't blow away entry_price — keep the prior estimate so
            # P&L attribution between buckets stays continuous.
            self._state_change_keys.add((symbol, side))

        grid_size = max(exchange_size - trend_size, 0.0)
        if abs(grid_size - grid_bucket.size) > 1e-10:
            self._state_change_keys.add((symbol, side))

        # Solve for grid bucket entry price. When grid is empty there's
        # nothing to anchor close markups on; entry_price=0 mirrors the
        # closed-position convention.
        if grid_size > 1e-12 and exchange_size > 1e-12:
            total_notional = exchange_size * exchange_entry
            trend_notional = trend_size * trend_entry
            grid_entry = (total_notional - trend_notional) / grid_size
            if grid_entry <= 0:
                # Edge case: trend notional exceeds the aggregate (the
                # exchange's avg drifted between polls). Fall back to
                # aggregate avg rather than emitting a negative entry.
                grid_entry = exchange_entry
        else:
            grid_entry = 0.0

        prev_best = grid_bucket.best_price
        new_grid = Position(
            size=grid_size, entry_price=grid_entry, best_price=prev_best,
        )
        if side == Side.LONG:
            ss.position_long = new_grid
        else:
            ss.position_short = new_grid

    @staticmethod
    def _extract_wallet_balance(balance: dict, currency: str) -> float:
        """Extract futures wallet balance, never free margin when avoidable.

        ``free`` is available margin and shrinks as positions consume
        margin; WE/HSL/sizing need the wallet/equity denominator. Binance
        futures exposes this in ``info.totalWalletBalance`` while ccxt's
        normalized payload often also carries ``USDT.total``.
        """
        info = balance.get("info") or {}
        for key in ("totalWalletBalance", "walletBalance"):
            try:
                value = float(info.get(key, 0) or 0)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0:
                return value

        currency_block = balance.get(currency) or {}
        for key in ("total", "walletBalance", "balance"):
            try:
                value = float(currency_block.get(key, 0) or 0)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0:
                return value
        totals = balance.get("total") or {}
        try:
            total_value = float(totals.get(currency, 0) or 0)
        except (AttributeError, TypeError, ValueError):
            total_value = 0.0
        if total_value > 0:
            return total_value
        try:
            return float(currency_block.get("free", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    async def _refresh_candles(self):
        for symbol in self.config.symbols:
            try:
                ohlcv = await self.exchange.fetch_ohlcv(
                    symbol, self.config.candle_timeframe, limit=100,
                )
                if not ohlcv:
                    continue
                ss = self.account.symbols[symbol]
                last_ts = self._last_candle_ts.get(symbol, -1)
                # Only feed NEW bars to the trend engine and data
                # provider. Without this gate every tick fed the trailing
                # 100 bars again, so trend history grew by 100/tick with
                # ~99 duplicates per bar — RSI / MACD / Bollinger all
                # drifted from their backtest semantics. Bars with
                # timestamp <= last_ts are skipped; the last_ts watermark
                # then advances to the most recent processed bar.
                new_rows = [r for r in ohlcv if int(r[0]) > last_ts]
                for row in new_rows:
                    close_px = float(row[4])
                    self.trend.update(symbol, close_px)
                    self.data_provider.append(
                        symbol,
                        Candle(
                            timestamp=int(row[0]),
                            open=float(row[1]),
                            high=float(row[2]),
                            low=float(row[3]),
                            close=close_px,
                            volume=float(row[5]),
                        ),
                    )
                    if not ss.ema.initialized:
                        ss.ema.init(
                            [
                                self.config.grid.ema_span_0,
                                self.config.grid.ema_span_1,
                            ],
                            close_px,
                        )
                    else:
                        ss.ema.update(close_px)
                    import math
                    row_high = float(row[2])
                    row_low = float(row[3])
                    lr = (
                        math.log(max(row_high, row_low + 1e-12) / row_low)
                        if row_low > 0 else 0.0
                    )
                    if not ss.volatility.initialized:
                        ss.volatility.init(
                            self.config.grid.entry_volatility_ema_span_hours,
                            lr,
                            bar_interval_minutes=self.config.bar_interval_minutes,
                        )
                    else:
                        ss.volatility.update(lr)
                if new_rows:
                    self._last_candle_ts[symbol] = int(new_rows[-1][0])
                # Apply populate_* once after all new rows are appended
                # so the strategy sees the full new window in one shot.
                if new_rows:
                    self._apply_strategy_populates(symbol)

                # Per-tick state that should ALWAYS reflect the latest
                # bar even on a no-new-bar tick (the watermark may not
                # advance every loop iteration if loop interval < bar
                # interval).
                last = ohlcv[-1]
                ss.last_price = float(last[4])
                high = float(last[2])
                low = float(last[3])
                # Stage 8 trailing-entry bundle: seed on first sight
                # of an open grid bucket, invalidate when closed so
                # the next open re-seeds. Trend bucket positions
                # don't drive trailing re-entries (they're managed
                # by the overlay path).
                if ss.position_long.is_open:
                    if not ss.trailing_long.initialized:
                        ss.trailing_long.reset(ss.position_long.entry_price)
                    ss.trailing_long.update_long(high, low)
                else:
                    ss.trailing_long.initialized = False
                if ss.position_short.is_open:
                    if not ss.trailing_short.initialized:
                        ss.trailing_short.reset(ss.position_short.entry_price)
                    ss.trailing_short.update_short(high, low)
                else:
                    ss.trailing_short.initialized = False
                # Ratchet trailing-stop high-water-marks across grid
                # and trend buckets.
                for pos in (ss.position_long, ss.trend_long):
                    if pos.is_open:
                        pos.update_best_price(high, Side.LONG)
                for pos in (ss.position_short, ss.trend_short):
                    if pos.is_open:
                        pos.update_best_price(low, Side.SHORT)
            except Exception:
                logger.exception("Failed to fetch candles for %s", symbol)

    async def _refresh_fills(self) -> None:
        """Poll exchange trade history for new fills and fan them out
        to protections / kelly_sizer / source-PnL bookkeeping. This is
        the live counterpart of Backtester's per-tick fill loop —
        before this method existed those subsystems silently received
        zero fills in live and behaved as if the bot never traded.
        """
        now_ms = self._now_ms()

        def _sink(fills: list[Fill]) -> None:
            if not fills:
                return
            # Process fills in TIMESTAMP order, enriching + applying
            # bucket updates ONE AT A TIME. A naive "enrich all, then
            # apply all" had a subtle ordering bug: when the same
            # fetch_my_trades batch contained a trend entry AND a
            # trend close, the close's fallback PnL ran while the
            # bucket still showed empty (entry hadn't been applied yet)
            # and emitted realized_pnl=0 for what was actually a
            # closing trade. Sequential apply keeps the bucket state
            # consistent with the timeline.
            sorted_fills = sorted(fills, key=lambda f: f.timestamp)
            enriched: list[Fill] = []
            for f in sorted_fills:
                e = self._enrich_fill_pnl(f)
                enriched.append(e)
                # Best-effort journal mark — fills attributed via cID
                # close out the durable intent. We look up the cID via
                # FillEventManager's index in reverse to find which
                # cID this trade belongs to.
                for jcid, meta in self.intent_journal.records.items():
                    if meta.get("kind") in ("filled", "rejected",
                                              "canceled", "resolved"):
                        continue
                    if meta.get("source") == e.source.value and \
                       meta.get("symbol") == e.symbol and \
                       meta.get("side") == e.side.value:
                        try:
                            self.intent_journal.mark_terminal(
                                cid=jcid, kind="filled",
                                now_ms=self._now_ms(),
                            )
                        except Exception:
                            pass
                        break
                if e.source == OrderSource.TREND:
                    self._apply_fill_to_trend_bucket(
                        Order(
                            symbol=e.symbol,
                            side=e.side,
                            price=e.price,
                            qty=e.qty,
                            source=e.source,
                            reduce_only=e.reduce_only,
                        ),
                        e.qty,
                    )
                    # Clear the pending-overlay slot on a non-reduce
                    # trend fill — the in-flight market entry has
                    # landed, the bucket now reflects it. Also clear
                    # the unknown_overlay slot if the resolution
                    # happens to land via a fill rather than the
                    # explicit _resolve_unknowns path.
                    if not e.reduce_only:
                        self._pending_overlay.pop((e.symbol, e.side), None)
                        self._unknown_overlay.pop((e.symbol, e.side), None)
                net = e.realized_pnl - e.fee
                self.account.add_realized_pnl(e.source, net, e.timestamp)
            # Bulk fan-out at the END so protections / Kelly see the
            # whole batch atomically — they don't need ordering between
            # fills within a tick.
            self.protections.update(enriched, self.account, now_ms)
            if self.kelly_sizer is not None:
                self.kelly_sizer.record_fills(enriched)

        for symbol in self.config.symbols:
            try:
                await self.fill_events.poll(symbol, now_ms, _sink)
            except Exception:
                logger.exception("fill-event poll failed for %s", symbol)

    def _apply_strategy_populates(self, symbol: str) -> None:
        """Mirror of Backtester._apply_strategy_populates — call the
        strategy's populate_indicators / populate_entry_trend /
        populate_exit_trend hooks on the cached DataFrame so signal
        columns become available for read_strategy_signals.
        """
        strategy_type = type(self.strategy)
        if (
            strategy_type.populate_indicators is DefaultStrategy.populate_indicators
            and strategy_type.populate_entry_trend is DefaultStrategy.populate_entry_trend
            and strategy_type.populate_exit_trend is DefaultStrategy.populate_exit_trend
        ):
            return  # DefaultStrategy.populate_* are no-ops — skip the work.
        try:
            df = self.data_provider.get_dataframe(symbol)
        except Exception:
            return
        if df is None or len(df) == 0:
            return
        meta = {"pair": symbol}
        try:
            out_ind = self.strategy.populate_indicators(df, meta)
            out_ent = self.strategy.populate_entry_trend(
                out_ind if out_ind is not None else df, meta,
            )
            out_ext = self.strategy.populate_exit_trend(
                out_ent if out_ent is not None else (out_ind or df), meta,
            )
        except Exception:
            logger.exception("Strategy populate_* raised for %s", symbol)
            return
        final = out_ext if out_ext is not None else df
        if final is df:
            return
        for col in ("enter_long", "enter_short", "exit_long", "exit_short",
                    "enter_tag", "exit_tag"):
            if col in final.columns and col not in df.columns:
                df[col] = final[col]

    def _quantize_order_for_send(self, order: Order) -> Order | None:
        """Return the order with qty floored to qty_step, or None if it
        falls below min_qty / min_cost after quantization. Centralising
        the quantization here lets dedup, reconcile, and create_order
        all reason about the same post-quantize qty (the one that
        actually reaches the exchange)."""
        ep = self.exchange_params.get(order.symbol)
        if ep is None or ep.qty_step <= 0:
            return order
        import math
        new_qty = math.floor(order.qty / ep.qty_step) * ep.qty_step
        if new_qty < ep.min_qty:
            return None
        if new_qty * order.price * ep.c_mult < ep.min_cost:
            return None
        if new_qty == order.qty:
            return order
        from dataclasses import replace
        return replace(order, qty=new_qty)

    async def _reconcile_orders(self, desired: list[Order]):
        now_ms = self._now_ms()
        # Quantize / revalidate every desired order to its post-send
        # form FIRST so dedup keys, reconcile matching, and the eventual
        # exchange call all see the same qty. Without this, a sizer
        # producing 0.00723 and 0.00729 on consecutive ticks would
        # quantize to the same 0.007 on send, but dedup-by-raw-qty
        # would miss the second one and send a duplicate.
        quantized: list[Order] = []
        for o in desired:
            q = self._quantize_order_for_send(o)
            if q is not None:
                quantized.append(q)
        desired = quantized
        # ── prune stale dedup + cOID cache BEFORE stamping ────────
        # Order matters: stamping cOIDs first would refresh the ts of
        # any expired-but-recurring identity (the _assign_cid path
        # bumps the watermark whenever it finds a key). The expected
        # semantic is "an entry that hasn't been requested for a while
        # is dead" — so prune first, then let _assign_cid mint a fresh
        # cOID for any identity whose old entry just got dropped.
        cutoff = now_ms - self._recent_order_window_ms
        while self._recent_creates and self._recent_creates[0][0] < cutoff:
            self._recent_creates.popleft()
        while self._recent_cancels and self._recent_cancels[0][0] < cutoff:
            self._recent_cancels.popleft()
        # cOID cache uses its OWN, much longer TTL — see __init__.
        # Survives stop/start cycles and brief outages so that open
        # orders left on the exchange can still be matched by cOID
        # instead of being silently cancelled+recreated.
        cid_cutoff = now_ms - self._cid_cache_ttl_ms
        stale_keys = [
            k for k, (_, ts) in self._cid_by_desired.items()
            if ts < cid_cutoff
        ]
        for k in stale_keys:
            self._cid_by_desired.pop(k, None)
        # Stamp cOID on EVERY desired order from a persistent cache
        # keyed by the order's stable identity. This is what makes the
        # cOID match in _orders_match actually trigger across ticks —
        # the same logical desired entry recurring tick after tick now
        # carries the same cOID, and the exchange's clientOrderId echo
        # exactly matches it.
        desired = [
            replace(o, client_order_id=self._assign_cid(o, now_ms))
            if not o.client_order_id else o
            for o in desired
        ]

        for symbol in self.config.symbols:
            try:
                existing = await self.exchange.fetch_open_orders(symbol)
            except Exception:
                existing = []
            self._open_orders[symbol] = existing

        # Dedup keys are full desired-order identities so a LONG-close
        # sell can't accidentally suppress a SHORT-entry sell at the
        # same price and qty (the pre-fix tuple only had symbol/price/
        # qty and silently cross-blocked them).
        recent_create_keys = {ident for (_, ident) in self._recent_creates}
        recent_cancel_ids = {oid for (_, oid) in self._recent_cancels}

        to_cancel = []
        to_create = []

        for symbol in self.config.symbols:
            existing = self._open_orders.get(symbol, [])
            symbol_desired = [o for o in desired if o.symbol == symbol]

            matched_existing = set()
            matched_desired = set()

            for i, d in enumerate(symbol_desired):
                for j, e in enumerate(existing):
                    if j in matched_existing:
                        continue
                    if self._orders_match(d, e):
                        matched_existing.add(j)
                        matched_desired.add(i)
                        break

            for j, e in enumerate(existing):
                if j not in matched_existing:
                    eid = e.get("id")
                    if eid and eid in recent_cancel_ids:
                        continue  # already cancelled this window
                    to_cancel.append(e)

            for i, d in enumerate(symbol_desired):
                if i not in matched_desired:
                    # Skip new entries on a (symbol, side) whose position
                    # just changed — stale exchange state could cause a
                    # double. Reduce-only exits are always safe to send.
                    if (
                        (symbol, d.side) in self._state_change_keys
                        and not d.reduce_only
                    ):
                        continue
                    # Skip if we recently created this exact entry
                    # (full identity, not just symbol/price/qty).
                    if self._desired_identity(d) in recent_create_keys:
                        continue
                    to_create.append(d)

        # Clear state-change set after reconciliation so next tick can
        # create orders again on those (symbol, side) pairs.
        self._state_change_keys.clear()

        for e in to_cancel[:self.config.max_orders_per_batch]:
            await self._cancel_order(e)

        for d in to_create[:self.config.max_orders_per_batch]:
            # cOID was stamped at the top of _reconcile_orders. Record
            # the FULL desired identity tuple BEFORE the API call so
            # the next tick sees this order as in-flight even if the
            # exchange hasn't yet confirmed.
            self._recent_creates.append((now_ms, self._desired_identity(d)))
            await self._create_order(d)

    def _orders_match(self, desired: Order, existing: dict) -> bool:
        # Prefer exact clientOrderId match — when both sides carry it,
        # ambiguity (two same-price entries, fuzzy qty matching)
        # disappears. ccxt normalises the field to ``clientOrderId``
        # but some exchanges echo it under ``info.clientOrderId``.
        d_cid = desired.client_order_id
        if d_cid:
            e_cid = (
                existing.get("clientOrderId")
                or (existing.get("info") or {}).get("clientOrderId")
                or ""
            )
            if e_cid:
                return str(e_cid) == d_cid

        # Fuzzy fallback: side + price + qty + reduceOnly. The pre-fix
        # fallback only compared (side, price, qty), so a LONG-close
        # sell could match a SHORT-entry sell with coincident price/qty
        # — same ccxt side string, very different intent. We now also
        # require reduceOnly to match. If the exchange doesn't surface
        # reduceOnly OR positionSide we REFUSE to fuzzy-match: the
        # safer outcome is to send the new order (potential duplicate
        # caught by recent_creates dedup) than to silently absorb our
        # desired into someone else's order.
        e_side = existing.get("side", "").lower()
        d_side = desired.exchange_side
        if e_side != d_side:
            return False

        info = existing.get("info") or {}
        e_reduce = existing.get("reduceOnly")
        if e_reduce is None:
            raw = info.get("reduceOnly")
            if raw is not None:
                e_reduce = str(raw).lower() in ("true", "1", "yes")
        if e_reduce is None:
            # Conservative: cOID missing AND reduceOnly missing — we
            # genuinely don't know what this exchange order represents.
            # Don't fuzzy-match.
            return False
        if bool(e_reduce) != bool(desired.reduce_only):
            return False

        # positionSide (hedge mode): when both echo it, require equality.
        e_pos_side = (
            existing.get("positionSide")
            or info.get("positionSide")
            or ""
        ).lower()
        if e_pos_side and e_pos_side not in ("both", desired.side.value):
            return False

        e_price = float(existing.get("price", 0))
        if e_price > 0 and abs(e_price - desired.price) / e_price > self.config.order_match_tolerance_pct:
            return False

        e_amount = float(existing.get("amount", 0))
        if e_amount > 0 and abs(e_amount - desired.qty) / e_amount > 0.05:
            return False

        return True

    async def _cancel_order(self, existing: dict):
        order_id = existing.get("id")
        symbol = existing.get("symbol")
        if not order_id or not symbol:
            return
        if self.config.dry_run:
            logger.info("[DRY] cancel %s %s", symbol, order_id)
            return
        try:
            await self.exchange.cancel_order(order_id, symbol)
            self._recent_cancels.append((self._now_ms(), order_id))
            logger.info("Cancelled %s %s", symbol, order_id)
        except Exception:
            logger.exception("Failed to cancel %s %s", symbol, order_id)

    async def _resolve_unknowns(self) -> None:
        """For each (symbol, side) in unknown_overlay, ask the exchange
        what actually happened. Two outcomes:

        * The cID (or the matching position) is still open → the order
          really IS in flight; demote back to pending.
        * The cID is nowhere in open orders AND the symbol position
          shows the expected size delta → the order filled in a window
          our fetch_my_trades didn't cover. Clear the unknown.

        Anything else (cID absent, no matching position diff) is left
        UNKNOWN and the side stays blocked. Operators see the persistent
        WARNING and intervene.
        """
        if not self._unknown_overlay:
            return
        # Resolve per symbol.
        resolved: list[tuple[str, Side]] = []
        for (symbol, side), ts in list(self._unknown_overlay.items()):
            try:
                opens = await self.exchange.fetch_open_orders(symbol)
            except Exception:
                logger.exception(
                    "[overlay] fetch_open_orders failed during resolve "
                    "for %s — leaving %s in UNKNOWN", symbol, side.value,
                )
                continue
            # Find OUR cIDs for this (symbol, side) that the journal
            # still considers non-terminal.
            ours = {
                cid for cid, rec in self.intent_journal.non_terminal().items()
                if rec.get("symbol") == symbol
                and rec.get("side") == side.value
                and rec.get("source") == OrderSource.TREND.value
                and not rec.get("reduce_only", False)
            }
            still_open = False
            for o in opens:
                ecid = (
                    o.get("clientOrderId")
                    or (o.get("info") or {}).get("clientOrderId")
                    or ""
                )
                if str(ecid) in ours:
                    still_open = True
                    break
            if still_open:
                logger.info(
                    "[overlay] %s %s resolved: order still OPEN — back to pending",
                    symbol, side.value,
                )
                self._pending_overlay[(symbol, side)] = self._now_ms()
                resolved.append((symbol, side))
                continue
            # Not in open orders: either filled or cancelled out of band.
            # The trend bucket will be updated by the next fill_events
            # poll if the fill is real; in the meantime clear the unknown
            # and let the next tick decide based on the actual bucket
            # state. Mark all the candidate journal cIDs as resolved
            # so they don't get replayed again on next restart.
            for cid in ours:
                try:
                    self.intent_journal.mark_terminal(
                        cid=cid, kind="resolved", now_ms=self._now_ms(),
                        reason="unknown_resolve_no_open",
                    )
                except Exception:
                    pass
            logger.info(
                "[overlay] %s %s resolved: no open order; clearing UNKNOWN",
                symbol, side.value,
            )
            resolved.append((symbol, side))
        for key in resolved:
            self._unknown_overlay.pop(key, None)

    def _replay_intent_journal(self) -> None:
        """Resurrect attribution + pending overlay from the durable
        journal after a crash/restart. Idempotent; safe to call after
        a clean shutdown too (journal will be ~empty in that case)."""
        try:
            records = self.intent_journal.replay()
        except Exception:
            logger.exception("[live] intent journal replay failed")
            return
        non_terminal = self.intent_journal.non_terminal()
        if not non_terminal:
            return
        logger.warning(
            "[live] intent journal: %d in-flight cIDs to restore",
            len(non_terminal),
        )
        for cid, rec in non_terminal.items():
            try:
                source = OrderSource(rec.get("source", "grid"))
                side = Side(rec.get("side", "long"))
            except ValueError:
                continue
            reduce_only = bool(rec.get("reduce_only", False))
            symbol = rec.get("symbol", "")
            ex_id = rec.get("exchange_id", "")
            # Restore attribution under both keys.
            self.fill_events.register_outgoing(
                str(ex_id or ""), source, side, reduce_only,
                client_order_id=cid,
            )
            # Restore pending overlay claim for non-reduce TREND.
            if source == OrderSource.TREND and not reduce_only:
                self._pending_overlay[(symbol, side)] = float(
                    rec.get("ts", self._now_ms())
                )

    def _now_ms(self) -> int:
        """Single point of truth for wall-clock ms in the live trader.

        Centralising the call site lets tests monkeypatch one method
        instead of chasing ``time.time()`` references scattered through
        the file, and removes a class of flaky tests that depended on
        the real clock matching test-fixture sentinels (e.g. the
        pending-overlay TTL test had to back-date entries to ``1.0``).
        """
        return int(time.time() * 1000)

    @staticmethod
    def _make_client_order_id() -> str:
        """Generate a short-ish client order ID that fits Binance's
        36-char limit and is unique enough across a multi-symbol bot."""
        return f"cb-{uuid.uuid4().hex[:24]}"

    @staticmethod
    def _desired_identity(order: Order) -> tuple:
        """Stable identity for a desired order across ticks.

        Quantization upstream means consecutive ticks emitting the
        "same" entry produce the same (price, qty), so this tuple is
        the right key for "is this desired the same logical order I
        sent last tick?".

        ``is_market`` is part of the identity: a limit and a market
        entry at the same price/qty are different exchange orders
        (different order_type, different fee tier). Without including
        it, switching a logical desired from limit to market (e.g. a
        regime change promoting a previously-limit overlay to market)
        would reuse the limit's cOID — and that cOID is already bound
        to a now-cancelled limit on the exchange.
        """
        return (
            order.symbol,
            order.side.value,
            order.source.value,
            bool(order.reduce_only),
            bool(order.is_market),
            round(order.price, 6),
            round(order.qty, 8),
        )

    def _assign_cid(self, order: Order, now_ms: int) -> str:
        """Return the cOID this desired order should carry, reusing a
        cached value when the same identity recurs across ticks.

        The cache TTL matches the dedup window so a logical entry that
        survives multiple reconcile passes keeps a stable cOID, while
        entries that disappear (filled, cancelled, or de-prioritised)
        eventually fall out of the cache via the prune step at the
        top of _reconcile_orders.
        """
        if order.client_order_id:
            # An explicit cOID (e.g. set by a strategy hook) wins.
            self._cid_by_desired[self._desired_identity(order)] = (
                order.client_order_id, now_ms,
            )
            return order.client_order_id
        key = self._desired_identity(order)
        existing = self._cid_by_desired.get(key)
        if existing is not None:
            # Refresh the watermark so the cache entry stays alive while
            # the desired order keeps recurring.
            self._cid_by_desired[key] = (existing[0], now_ms)
            return existing[0]
        cid = self._make_client_order_id()
        self._cid_by_desired[key] = (cid, now_ms)
        return cid

    async def _create_order(self, order: Order):
        side = order.exchange_side
        order_type = "market" if order.is_market else "limit"

        # Ensure a cOID even when callers bypass _reconcile_orders (tests,
        # direct calls). Reconcile stamps these too — this is a backstop.
        if not order.client_order_id:
            order = replace(order, client_order_id=self._make_client_order_id())

        params = {"clientOrderId": order.client_order_id}
        if order.reduce_only:
            params["reduceOnly"] = True

        # Quantize qty to the exchange's qty_step BEFORE sending and
        # revalidate every exchange constraint against the post-quantize
        # value. Sizers (Kelly, correlation gate, vol-target) scale qty
        # without re-quantizing, and only the final send_qty matters
        # to the exchange's accept/reject decision.
        ep = self.exchange_params.get(order.symbol)
        send_qty = order.qty
        if ep is not None and ep.qty_step > 0:
            import math
            send_qty = math.floor(order.qty / ep.qty_step) * ep.qty_step
            if send_qty < ep.min_qty:
                logger.warning(
                    "Skipping %s %s — quantized qty %.10f below min_qty %.10f",
                    side, order.symbol, send_qty, ep.min_qty,
                )
                return
            # min_cost must be revalidated against the QUANTIZED qty.
            # The upstream engine quantizes too, but sizers running
            # downstream may shrink qty below the min_cost floor while
            # still above min_qty (different constraint axes).
            cost = send_qty * order.price * ep.c_mult
            if cost < ep.min_cost:
                logger.warning(
                    "Skipping %s %s — cost %.4f below min_cost %.4f at qty %.10f",
                    side, order.symbol, cost, ep.min_cost, send_qty,
                )
                return

        if self.config.dry_run:
            logger.info(
                "[DRY] %s %s %s %.4f @ %.2f (%s%s)",
                order_type, side, order.symbol, send_qty, order.price,
                order.source.value, " reduce" if order.reduce_only else "",
            )
            # Trend bucket bookkeeping: even in dry-run the local state
            # must stay consistent so the next tick's overlay decision
            # doesn't duplicate an entry the "exchange" already has.
            self._apply_fill_to_trend_bucket(order, send_qty)
            return

        # Reserve attribution + pending BEFORE the network call. If the
        # request times out or the connection drops, the exchange may
        # already have accepted the order — and when the eventual fill
        # comes in via fetch_my_trades we still need to attribute it
        # correctly. Pre-registering by clientOrderId means the fill
        # manager can find the source/meta even if we never see an
        # exchange-assigned order id. For a TREND market entry we ALSO
        # set the pending slot up front so the next tick can't race a
        # duplicate overlay through while the network call is in
        # flight. Both reservations are cleared on a clean reject.
        if order.client_order_id:
            self.fill_events.register_outgoing(
                "",  # exchange id unknown yet
                order.source,
                order.side,
                order.reduce_only,
                client_order_id=order.client_order_id,
            )
        pre_reserved_pending = (
            order.source == OrderSource.TREND and not order.reduce_only
        )
        if pre_reserved_pending:
            self._pending_overlay[(order.symbol, order.side)] = (
                self._now_ms()
            )
        # Durable intent journal — written and fsync'd BEFORE the
        # network call so a SIGKILL in the create_order ack window
        # still lets us replay attribution + pending on restart.
        if order.client_order_id:
            try:
                self.intent_journal.submit(
                    cid=order.client_order_id,
                    symbol=order.symbol,
                    side=order.side.value,
                    source=order.source.value,
                    reduce_only=order.reduce_only,
                    is_market=order.is_market,
                    now_ms=self._now_ms(),
                )
            except Exception:
                logger.exception(
                    "[live] intent journal write FAILED for cid=%s — "
                    "refusing to send order without durable intent",
                    order.client_order_id,
                )
                # Roll back the in-memory reservations and abort.
                if pre_reserved_pending:
                    self._pending_overlay.pop(
                        (order.symbol, order.side), None
                    )
                return

        try:
            # Market orders ignore price; pass None to ccxt.
            price = None if order.is_market else order.price
            result = await self.exchange.create_order(
                order.symbol, order_type, side, send_qty, price, params
            )
            status = str(result.get("status", "")).lower()
            oid = result.get("id", "?")
            # "canceled" / "cancelled" cover exchanges that auto-cancel
            # post-only or out-of-band rejected orders without an explicit
            # "rejected" status.
            if status in ("expired", "rejected", "canceled", "cancelled"):
                logger.warning(
                    "Order %s %s %s %.4f @ %s → %s (status=%s)",
                    order_type, side, order.symbol, order.qty,
                    f"{order.price:.2f}" if not order.is_market else "MKT",
                    oid, status,
                )
                # Remove from recent_creates so the next tick can
                # retry. Match on the FULL identity tuple to avoid
                # clearing a different desired (same px/qty/side but
                # different source or reduce_only).
                target = self._desired_identity(order)
                for i in range(len(self._recent_creates) - 1, -1, -1):
                    _, ident = self._recent_creates[i]
                    if ident == target:
                        del self._recent_creates[i]
                        break
                # Rejected/cancelled TREND entries must free the
                # pending-overlay slot — otherwise a transient reject
                # would block the next try until the 5-min TTL.
                if (
                    order.source == OrderSource.TREND
                    and not order.reduce_only
                ):
                    self._pending_overlay.pop(
                        (order.symbol, order.side), None
                    )
                # Mark terminal in journal so replay doesn't try to
                # restore this cID as in-flight.
                if order.client_order_id:
                    try:
                        self.intent_journal.mark_terminal(
                            cid=order.client_order_id,
                            kind=(
                                "rejected" if status == "rejected"
                                else "canceled"
                            ),
                            now_ms=self._now_ms(),
                        )
                    except Exception:
                        logger.exception("intent_journal mark_terminal failed")
            else:
                logger.info(
                    "Created %s %s %s %.4f @ %s → %s",
                    order_type, side, order.symbol, order.qty,
                    f"{order.price:.2f}" if not order.is_market else "MKT",
                    oid,
                )
                # We pre-registered by cOID above. Now patch in the
                # exchange-assigned id so fills that echo only ``order``
                # (not clientOrderId) are still attributable.
                if str(oid) and oid != "?":
                    self.fill_events.register_outgoing(
                        str(oid), order.source, order.side,
                        order.reduce_only,
                        client_order_id=order.client_order_id,
                    )
                # Refresh pending timestamp on confirmed acceptance.
                if pre_reserved_pending:
                    self._pending_overlay[(order.symbol, order.side)] = (
                        self._now_ms()
                    )
                # Journal: transition from submit → open with the
                # exchange-assigned id alongside.
                if order.client_order_id:
                    try:
                        self.intent_journal.open(
                            cid=order.client_order_id,
                            exchange_id=str(oid),
                            now_ms=self._now_ms(),
                        )
                    except Exception:
                        logger.exception("intent_journal open failed")
        except Exception:
            # Cannot confirm whether the exchange accepted or not. The
            # safe assumption is "POSSIBLY accepted" — we keep the
            # cOID attribution and any TREND pending reservation alive
            # so a delayed fill is still routable and the next tick
            # can't double-up a market overlay. Operator alerting
            # comes from the WARNING; the system self-corrects when
            # the next fetch_my_trades / fetch_open_orders poll shows
            # the order's real state.
            logger.warning(
                "[live] create_order EXCEPTION — treating as UNKNOWN state "
                "(may have been accepted by exchange). Order: %s %s %s "
                "%.4f @ %s cID=%s",
                order_type, side, order.symbol, order.qty,
                f"{order.price:.2f}" if not order.is_market else "MKT",
                order.client_order_id or "<none>",
                exc_info=True,
            )

    def _emit_trend_overlay(
        self,
        symbol: str,
        regime: RegimeView,
        price: float,
        exchange: ExchangeParams,
    ) -> list[Order]:
        if regime.trend_overlay is None or regime.trend_qty_scale <= 0:
            return []
        # Pending entries past TTL graduate to UNKNOWN — they don't get
        # silently freed. An unknown entry continues to block emit
        # until ``_resolve_unknowns`` consults the exchange and either
        # confirms the order resolved (cleared the unknown) or leaves
        # it unresolved (continues to block + alerts).
        now_ms = self._now_ms()
        cutoff = now_ms - self._pending_overlay_ttl_ms
        new_pending: dict[tuple[str, Side], float] = {}
        for k, ts in self._pending_overlay.items():
            if ts >= cutoff:
                new_pending[k] = ts
            else:
                self._unknown_overlay[k] = ts
                logger.warning(
                    "[overlay] %s %s pending past TTL — moving to "
                    "UNKNOWN; reconcile will attempt to resolve",
                    k[0], k[1].value,
                )
        self._pending_overlay = new_pending
        ss = self.account.symbols.get(symbol)
        if ss is not None:
            existing = (
                ss.trend_long if regime.trend_overlay == Side.LONG
                else ss.trend_short
            )
            if existing.is_open:
                return []
        # The bucket may still be empty if a market-overlay create_order
        # acked moments ago but the fill_events poll hasn't picked it
        # up yet. Treat the pending slot as "already entered" until
        # the fill lands (which clears the pending entry in _sink).
        if (symbol, regime.trend_overlay) in self._pending_overlay:
            return []
        # Unknown state: pending TTL elapsed but ``_resolve_unknowns``
        # hasn't confirmed a terminal state yet. Continue to block.
        if (symbol, regime.trend_overlay) in self._unknown_overlay:
            return []
        merger_cfg = self.config.merger
        kelly_scale = (
            self.kelly_sizer.fraction(OrderSource.TREND)
            if self.kelly_sizer is not None
            else 1.0
        )
        effective_scale = regime.trend_qty_scale * kelly_scale
        if effective_scale <= 0:
            return []
        budget = self.account.balance * merger_cfg.trend_position_max_pct
        notional = budget * merger_cfg.trend_entry_qty_pct * effective_scale
        qty = notional / max(price * exchange.c_mult, 1e-12)
        qty = max(qty, exchange.min_qty)
        cost = qty * price * exchange.c_mult
        if cost > budget or cost < exchange.min_cost:
            return []
        # Trend overlay entries cross the book immediately — see the
        # parallel comment in Backtester._emit_trend_overlay.
        return [Order(
            symbol=symbol,
            side=regime.trend_overlay,
            price=price,
            qty=qty,
            source=OrderSource.TREND,
            is_market=True,
        )]

    def _enrich_fill_pnl(self, fill: Fill) -> Fill:
        """Fallback realized-PnL computation for reduce-only fills.

        On Binance USDM ccxt surfaces ``info.realizedPnl`` directly and
        we use it as-is. Elsewhere (Bybit, OKX, …) that field can be
        missing or zero; for a reduce_only trade with realized_pnl=0 we
        compute it locally from the bucket's pre-fill entry_price:

            LONG  close: pnl = qty * (fill.price - entry_price) * c_mult
            SHORT close: pnl = qty * (entry_price - fill.price) * c_mult

        Non-reduce fills (opens / adds) carry no realized PnL by
        definition, so they pass through unchanged. We also leave
        already-populated realized_pnl alone — if the exchange told us a
        number we trust it over our local reconstruction.
        """
        if not fill.reduce_only or fill.realized_pnl != 0.0:
            return fill
        ss = self.account.symbols.get(fill.symbol)
        if ss is None:
            return fill
        # Pick the bucket the fill came out of. Order.source distinguishes
        # which bucket got reduced; SymbolState.bucket honors it.
        bucket = ss.bucket(fill.source, fill.side)
        if not bucket.is_open or bucket.entry_price <= 0:
            return fill
        ep = self.exchange_params.get(fill.symbol)
        c_mult = ep.c_mult if ep is not None else 1.0
        if fill.side == Side.LONG:
            pnl = fill.qty * (fill.price - bucket.entry_price) * c_mult
        else:
            pnl = fill.qty * (bucket.entry_price - fill.price) * c_mult
        from dataclasses import replace as _replace
        return _replace(fill, realized_pnl=pnl)

    def _apply_fill_to_trend_bucket(self, order: Order, filled_qty: float) -> None:
        """Update the local trend bucket after a confirmed trend order.

        The exchange reports a single aggregate position per (symbol, side),
        so we must track the trend bucket locally and subtract it when
        rebuilding the grid bucket from the exchange total. Without this,
        a filled trend-market entry would silently flow into the grid
        bucket and the overlay would try to enter again on the next tick.
        """
        if order.source != OrderSource.TREND:
            return
        ss = self.account.symbols.get(order.symbol)
        if ss is None:
            return
        bucket = ss.trend_long if order.side == Side.LONG else ss.trend_short
        if order.reduce_only:
            close_qty = min(filled_qty, abs(bucket.size))
            if abs(bucket.size) <= close_qty + 1e-12:
                bucket.size = 0.0
                bucket.entry_price = 0.0
                bucket.best_price = 0.0
            else:
                delta = close_qty if bucket.size > 0 else -close_qty
                bucket.size -= delta
        else:
            if not bucket.is_open:
                bucket.size = filled_qty
                bucket.entry_price = order.price
                bucket.best_price = order.price
            else:
                new_size = bucket.size + filled_qty
                if abs(new_size) > 1e-12:
                    bucket.entry_price = (
                        bucket.size * bucket.entry_price + filled_qty * order.price
                    ) / new_size
                bucket.size = new_size

    def _latest_candle(self, symbol: str, price: float, now_ms: int) -> Candle:
        """Return the most recent candle for `symbol`, or a degenerate one
        synthesized from the last mark price if the buffer is empty."""
        df = self.data_provider.get_dataframe(symbol)
        if len(df) > 0:
            row = df.iloc[-1]
            return Candle(
                timestamp=int(row["timestamp"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
        return Candle(
            timestamp=now_ms, open=price, high=price, low=price,
            close=price, volume=0.0,
        )

    async def _save_state(self):
        # Persist trend-bucket size + entry price so a restart can
        # subtract the right trend share from the exchange's aggregate
        # position and rebuild grid_size correctly. Without this, the
        # first _refresh_account after restart attributes the entire
        # exchange position to the grid bucket, and any later overlay
        # entry stacks ON TOP — total bucket size diverges from the
        # exchange and P&L attribution is silently wrong until the
        # trend bucket fully closes.
        trend_buckets: dict[str, dict] = {}
        for sym, ss in self.account.symbols.items():
            entry = {}
            if ss.trend_long.is_open:
                entry["long_size"] = ss.trend_long.size
                entry["long_entry_price"] = ss.trend_long.entry_price
            if ss.trend_short.is_open:
                entry["short_size"] = ss.trend_short.size
                entry["short_entry_price"] = ss.trend_short.entry_price
            if entry:
                trend_buckets[sym] = entry

        state = {
            "balance": self.account.balance,
            "equity": self.account.equity,
            "equity_peak": self.account.equity_peak,
            "timestamp": self._now_ms(),
            # ── risk-tier persistence (survives restart) ──
            "risk_tier": self.risk.tier.value,
            "risk_red_latched": self.risk.red_latched,
            "risk_red_cooldown_until": self.risk.red_cooldown_until,
            "risk_dd_ema": self.risk.dd_ema,
            # Persist both EMA bookkeeping flags so the first post-restart
            # assess doesn't overwrite the restored dd_ema with raw
            # drawdown via the "first call" branch of _update_dd_score.
            "risk_dd_initialized": self.risk._dd_initialized,
            "risk_last_assess_minute": self.risk.last_assess_minute,
            # Auto-release wallclock marker. Without persisting this, a
            # restart with red_latched=True would either lose the
            # auto-release timer (release_minutes=0 → never auto-clears)
            # or reset the countdown (red_latched_at_ms=0 → next assess
            # rewrites it to now). Either way the operator-visible
            # behavior diverges from the running bot.
            "risk_red_latched_at_ms": self.risk.red_latched_at_ms,
            # ── trend bucket persistence ──
            "trend_buckets": trend_buckets,
            # ── fill-event watermarks / order attribution ──
            "fill_events": self.fill_events.snapshot(),
            # ── per-source realized P&L ledger ──
            # Without these the following silently reset to 0 across
            # restarts: HslSupervisor's source_drawdown_pct (per-bucket
            # circuit breakers), unstuck loss_24h budget (kept the bot
            # from realizing more losses than the allowance), and
            # KellySizer's edge estimate (over time it'd rebuild but
            # the cold-start window means hours of mis-sized overlay).
            "grid_realized_pnl": self.account.grid_realized_pnl,
            "trend_realized_pnl": self.account.trend_realized_pnl,
            "grid_equity_peak": self.account.grid_equity_peak,
            "trend_equity_peak": self.account.trend_equity_peak,
            # Rolling 24h loss logs feed unstuck allowance — persist
            # them so a restart doesn't grant a fresh full budget. Only
            # entries within the last 24h are useful; loss_24h will
            # prune the rest on first call.
            "grid_loss_log": list(self.account.grid_loss_log),
            "trend_loss_log": list(self.account.trend_loss_log),
            # Persistent desired-identity → cOID cache. Without this, a
            # restart with open orders on the exchange would assign
            # fresh cOIDs to the same logical desired entries; reconcile
            # would NOT cOID-match the existing exchange orders, fall
            # back to fuzzy, and most often cancel-then-recreate them.
            # That round-trips fees and risks brief windows with no
            # open ladder. Serialised as a list of identity rows;
            # tuple keys aren't JSON-encodable. Schema:
            # [symbol, side, source, reduce_only, is_market, price, qty,
            #  cid, ts]
            "cid_cache": [
                [k[0], k[1], k[2], bool(k[3]), bool(k[4]),
                 float(k[5]), float(k[6]), cid, float(ts)]
                for k, (cid, ts) in self._cid_by_desired.items()
            ],
            # Persist pending TREND overlay claims so a process restart
            # between create_order ack and fill-event arrival doesn't
            # lose the "in-flight" marker. Without this, the next
            # post-restart tick sees an empty trend bucket AND an empty
            # pending set → emits another market overlay → doubles
            # exposure on a strong-regime tick.
            "pending_overlay": [
                [sym, side.value, float(ts)]
                for (sym, side), ts in self._pending_overlay.items()
            ],
            "unknown_overlay": [
                [sym, side.value, float(ts)]
                for (sym, side), ts in self._unknown_overlay.items()
            ],
        }
        try:
            Path(self.config.state_file).write_text(json.dumps(state, indent=2))
            # Compact the intent journal AFTER a successful state save:
            # any cIDs that are now terminal (their fills/rejects are
            # captured in state above) can be evicted from the journal.
            try:
                self.intent_journal.compact()
            except Exception:
                logger.exception("intent journal compact failed")
        except Exception:
            pass

    async def _load_state(self):
        try:
            data = json.loads(Path(self.config.state_file).read_text())
            self.account.equity_peak = data.get("equity_peak", 0)
            # Restore risk state so a restart doesn't forget RED.
            # Convert the persisted tier string back into the enum so
            # downstream `.value`/`.name` access keeps working until the
            # next assess() runs and rewrites it anyway.
            raw_tier = data.get("risk_tier")
            if raw_tier:
                try:
                    self.risk.tier = RiskTier(raw_tier)
                except ValueError:
                    logger.warning("[risk] unknown tier %r in state file; ignoring", raw_tier)
            self.risk.red_latched = data.get("risk_red_latched", False)
            self.risk.red_cooldown_until = data.get("risk_red_cooldown_until", 0)
            self.risk.dd_ema = data.get("risk_dd_ema", 0.0)
            # When dd_ema was meaningfully populated on disk, mark the
            # EMA as already-initialized so the next assess doesn't
            # short-circuit through the seeding branch that overwrites
            # dd_ema with raw drawdown.
            self.risk._dd_initialized = data.get(
                "risk_dd_initialized", self.risk._dd_initialized,
            )
            self.risk.last_assess_minute = data.get(
                "risk_last_assess_minute", self.risk.last_assess_minute,
            )
            self.risk.red_latched_at_ms = data.get(
                "risk_red_latched_at_ms", self.risk.red_latched_at_ms,
            )
            self.fill_events.load_snapshot(data.get("fill_events", {}))
            # Restore per-source realized P&L ledger so HSL / unstuck /
            # source-drawdown don't reset to zero on restart.
            self.account.grid_realized_pnl = float(
                data.get("grid_realized_pnl", self.account.grid_realized_pnl)
            )
            self.account.trend_realized_pnl = float(
                data.get("trend_realized_pnl", self.account.trend_realized_pnl)
            )
            self.account.grid_equity_peak = float(
                data.get("grid_equity_peak", self.account.grid_equity_peak)
            )
            self.account.trend_equity_peak = float(
                data.get("trend_equity_peak", self.account.trend_equity_peak)
            )
            grid_log = data.get("grid_loss_log") or []
            if isinstance(grid_log, list):
                self.account.grid_loss_log.clear()
                for entry in grid_log:
                    if isinstance(entry, (list, tuple)) and len(entry) == 2:
                        try:
                            self.account.grid_loss_log.append(
                                (int(entry[0]), float(entry[1]))
                            )
                        except (TypeError, ValueError):
                            continue
            trend_log = data.get("trend_loss_log") or []
            if isinstance(trend_log, list):
                self.account.trend_loss_log.clear()
                for entry in trend_log:
                    if isinstance(entry, (list, tuple)) and len(entry) == 2:
                        try:
                            self.account.trend_loss_log.append(
                                (int(entry[0]), float(entry[1]))
                            )
                        except (TypeError, ValueError):
                            continue
            pending = data.get("pending_overlay") or []
            if isinstance(pending, list):
                self._pending_overlay.clear()
                for row in pending:
                    if not isinstance(row, (list, tuple)) or len(row) != 3:
                        continue
                    try:
                        sym = str(row[0])
                        side = Side(str(row[1]))
                        ts = float(row[2])
                        self._pending_overlay[(sym, side)] = ts
                    except (TypeError, ValueError):
                        continue
            unknown = data.get("unknown_overlay") or []
            if isinstance(unknown, list):
                self._unknown_overlay.clear()
                for row in unknown:
                    if not isinstance(row, (list, tuple)) or len(row) != 3:
                        continue
                    try:
                        sym = str(row[0])
                        side = Side(str(row[1]))
                        ts = float(row[2])
                        self._unknown_overlay[(sym, side)] = ts
                    except (TypeError, ValueError):
                        continue

            cid_cache = data.get("cid_cache") or []
            if isinstance(cid_cache, list):
                self._cid_by_desired.clear()
                # Filter stale entries at load time so a reconcile
                # immediately after restart doesn't "revive" cOIDs that
                # were already evicted by the in-memory pruner. We
                # accept entries up to 24h old (well past the dedup
                # window) so a planned restart with stale state still
                # picks up reasonably-recent open orders.
                now_ms = self._now_ms()
                cache_cutoff_ms = now_ms - 24 * 60 * 60 * 1_000
                for row in cid_cache:
                    if not isinstance(row, (list, tuple)) or len(row) != 9:
                        continue
                    try:
                        ts = float(row[8])
                        if ts < cache_cutoff_ms:
                            continue
                        key = (
                            str(row[0]),       # symbol
                            str(row[1]),       # side.value
                            str(row[2]),       # source.value
                            bool(row[3]),      # reduce_only
                            bool(row[4]),      # is_market
                            round(float(row[5]), 6),
                            round(float(row[6]), 8),
                        )
                        self._cid_by_desired[key] = (str(row[7]), ts)
                    except (TypeError, ValueError):
                        continue
            now_ms = self._now_ms()
            in_cooldown = self.risk.red_cooldown_until > now_ms
            if self.risk.red_latched or in_cooldown:
                logger.warning(
                    "[risk] restored RED state (latched=%s cooldown_remaining_ms=%d) "
                    "— new entries blocked until cooldown expires or latch is reset",
                    self.risk.red_latched,
                    max(self.risk.red_cooldown_until - now_ms, 0),
                )
            # Restore trend bucket sizes & entry prices so the next
            # _refresh_account can attribute the exchange's aggregate
            # position correctly between grid and trend buckets.
            trend_buckets = data.get("trend_buckets", {}) or {}
            for sym, entry in trend_buckets.items():
                ss = self.account.symbols.get(sym)
                if ss is None:
                    # Init the symbol state so the restored data isn't lost
                    # if _init_exchange hasn't been called yet (race-safe).
                    ss = SymbolState(symbol=sym)
                    self.account.symbols[sym] = ss
                if "long_size" in entry:
                    ss.trend_long = Position(
                        size=float(entry["long_size"]),
                        entry_price=float(entry.get("long_entry_price", 0.0)),
                    )
                if "short_size" in entry:
                    ss.trend_short = Position(
                        size=float(entry["short_size"]),
                        entry_price=float(entry.get("short_entry_price", 0.0)),
                    )
                if entry:
                    logger.info(
                        "[restore] %s trend bucket: long=%.6f@%.2f short=%.6f@%.2f",
                        sym, ss.trend_long.size, ss.trend_long.entry_price,
                        ss.trend_short.size, ss.trend_short.entry_price,
                    )
        except Exception:
            pass
