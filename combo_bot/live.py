from __future__ import annotations
import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
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
        # Stage 9 — optional fractional-Kelly trend overlay throttle.
        # The reconcile path doesn't surface fills yet (TODO with the
        # protections wiring), so on live the sizer stays cold-start
        # at fraction 1.0 until that's hooked up.
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
        # Tuple: (timestamp_ms, symbol, price, qty).
        self._recent_creates: deque[tuple[float, str, float, float]] = deque(maxlen=guard_cap)
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
                maker_fee=float(market.get("maker", 0.0002)),
                taker_fee=float(market.get("taker", 0.0005)),
            )
            self.account.symbols[symbol] = SymbolState(symbol=symbol)

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
        await self._refresh_account()
        await self._refresh_candles()

        self.account.update_equity()
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
            now_ms = int(time.time() * 1000)
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
                )
                trend_entries = self.strategy_runner.filter_entries(
                    trend_entries, ctx_overlay,
                )
            trend_exits_l = self.merger.generate_trend_exit_orders(symbol, ss.trend_long, Side.LONG, price, ep)
            trend_exits_s = self.merger.generate_trend_exit_orders(symbol, ss.trend_short, Side.SHORT, price, ep)

            strategy_orders: list[Order] = []
            for ctx, pos in ((ctx_long, ss.position_long), (ctx_short, ss.position_short)):
                if not pos.is_open:
                    continue
                fx = self.strategy_runner.check_custom_exit(ctx)
                if fx is not None:
                    strategy_orders.append(fx)
                adj = self.strategy_runner.check_position_adjustment(ctx)
                if adj is not None:
                    strategy_orders.append(adj)

            ss.mode_long = regime_view.long_mode
            ss.mode_short = regime_view.short_mode

            all_desired_orders.extend(
                grid_long + grid_short + trailing_entries + trend_entries
                + trend_exits_l + trend_exits_s + strategy_orders
            )

        tick_ms = int(time.time() * 1000)
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
        )

        await self._reconcile_orders(all_desired_orders)
        await self._save_state()

        logger.info(
            "tick | balance=%.2f equity=%.2f dd=%.4f orders=%d",
            self.account.balance, self.account.equity,
            self.account.drawdown, len(all_desired_orders),
        )

    async def _refresh_account(self):
        try:
            balance = await self.exchange.fetch_balance({"type": "future"})
            self.account.balance = float(balance.get("USDT", {}).get("free", 0))

            positions = await self.exchange.fetch_positions(self.config.symbols)
            for p in positions:
                symbol = p["symbol"]
                if symbol not in self.account.symbols:
                    continue
                ss = self.account.symbols[symbol]
                side = p.get("side", "")
                size = abs(float(p.get("contracts", 0) or 0))
                entry = float(p.get("entryPrice", 0) or 0)
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
                    tracked_trend = abs(ss.trend_long.size)
                    grid_size = max(size - tracked_trend, 0.0)
                    if abs(grid_size - ss.position_long.size) > 1e-10:
                        self._state_change_keys.add((symbol, Side.LONG))
                    prev_best = ss.position_long.best_price
                    ss.position_long = Position(
                        size=grid_size, entry_price=entry, best_price=prev_best,
                    )
                elif side == "short":
                    tracked_trend = abs(ss.trend_short.size)
                    grid_size = max(size - tracked_trend, 0.0)
                    if abs(grid_size - ss.position_short.size) > 1e-10:
                        self._state_change_keys.add((symbol, Side.SHORT))
                    prev_best = ss.position_short.best_price
                    ss.position_short = Position(
                        size=grid_size, entry_price=entry, best_price=prev_best,
                    )
                ss.last_price = float(p.get("markPrice", 0) or ss.last_price)
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

    async def _refresh_candles(self):
        for symbol in self.config.symbols:
            try:
                ohlcv = await self.exchange.fetch_ohlcv(symbol, "1m", limit=100)
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
                        ss.ema.init([385.0, 620.0], close_px)
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
                        ss.volatility.init(1000.0, lr)
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

    def _apply_strategy_populates(self, symbol: str) -> None:
        """Mirror of Backtester._apply_strategy_populates — call the
        strategy's populate_indicators / populate_entry_trend /
        populate_exit_trend hooks on the cached DataFrame so signal
        columns become available for read_strategy_signals.
        """
        if isinstance(self.strategy, DefaultStrategy):
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
        now_ms = int(time.time() * 1000)
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
        # ── prune stale dedup entries ──────────────────────────────
        cutoff = now_ms - self._recent_order_window_ms
        while self._recent_creates and self._recent_creates[0][0] < cutoff:
            self._recent_creates.popleft()
        while self._recent_cancels and self._recent_cancels[0][0] < cutoff:
            self._recent_cancels.popleft()

        for symbol in self.config.symbols:
            try:
                existing = await self.exchange.fetch_open_orders(symbol)
            except Exception:
                existing = []
            self._open_orders[symbol] = existing

        # Dedup keys must include symbol; otherwise two symbols with
        # coincident grid prices would falsely cross-block each other.
        recent_create_keys = {(s, p, q) for (_, s, p, q) in self._recent_creates}
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
                    # Skip if we recently created this exact entry.
                    key = (d.symbol, round(d.price, 6), round(d.qty, 8))
                    if key in recent_create_keys:
                        continue
                    to_create.append(d)

        # Clear state-change set after reconciliation so next tick can
        # create orders again on those (symbol, side) pairs.
        self._state_change_keys.clear()

        for e in to_cancel[:self.config.max_orders_per_batch]:
            await self._cancel_order(e)

        for d in to_create[:self.config.max_orders_per_batch]:
            # Record BEFORE the API call so the next tick dedup sees it
            # even if the exchange hasn't confirmed yet.
            self._recent_creates.append(
                (now_ms, d.symbol, round(d.price, 6), round(d.qty, 8))
            )
            await self._create_order(d)

    def _orders_match(self, desired: Order, existing: dict) -> bool:
        e_side = existing.get("side", "").lower()
        d_side = desired.exchange_side

        if e_side != d_side:
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
            self._recent_cancels.append((int(time.time() * 1000), order_id))
            logger.info("Cancelled %s %s", symbol, order_id)
        except Exception:
            logger.exception("Failed to cancel %s %s", symbol, order_id)

    async def _create_order(self, order: Order):
        side = order.exchange_side
        order_type = "market" if order.is_market else "limit"

        params = {}
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
                # Remove from recent_creates so the next tick can retry.
                # Match on (symbol, price, qty) — symbol matters: a same
                # px/qty entry on a different symbol must not be wiped.
                target = (order.symbol, round(order.price, 6), round(order.qty, 8))
                for i in range(len(self._recent_creates) - 1, -1, -1):
                    _, s, p, q = self._recent_creates[i]
                    if (s, p, q) == target:
                        del self._recent_creates[i]
                        break
            else:
                logger.info(
                    "Created %s %s %s %.4f @ %s → %s",
                    order_type, side, order.symbol, order.qty,
                    f"{order.price:.2f}" if not order.is_market else "MKT",
                    oid,
                )
        except Exception:
            logger.exception(
                "Failed to create %s %s %s %.4f @ %.2f",
                order_type, side, order.symbol, order.qty, order.price,
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
        ss = self.account.symbols.get(symbol)
        if ss is not None:
            existing = (
                ss.trend_long if regime.trend_overlay == Side.LONG
                else ss.trend_short
            )
            if existing.is_open:
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
            "timestamp": int(time.time() * 1000),
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
            # ── trend bucket persistence ──
            "trend_buckets": trend_buckets,
        }
        try:
            Path(self.config.state_file).write_text(json.dumps(state, indent=2))
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
            now_ms = int(time.time() * 1000)
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
