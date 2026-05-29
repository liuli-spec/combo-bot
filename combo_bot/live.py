from __future__ import annotations
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from combo_bot.types import (
    AccountState, Candle, ExchangeParams, Fill, Order, OrderSource,
    Position, Side, SymbolState, TradingMode,
)
from combo_bot.grid_engine import GridConfig, GridEngine, calc_wallet_exposure
from combo_bot.trend_signal import TrendConfig, TrendEngine
from combo_bot.merger import MergerConfig, DecisionMerger
from combo_bot.risk import RiskConfig, RiskManager
from combo_bot.strategy import DefaultStrategy, IStrategy, StrategyRunner, TradeContext
from combo_bot.data_provider import DataProvider
from combo_bot.regime import RegimeArbiter, RegimeArbiterConfig, read_strategy_signals
from combo_bot.protections import IProtection, ProtectionManager
from combo_bot.sizing import KellySizer
from combo_bot.correlation import CorrelationGate
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
        self.account = AccountState()
        self.exchange_params: dict[str, ExchangeParams] = {}
        self._running = False
        self._open_orders: dict[str, list[dict]] = {}
        self._funding_rates: dict[str, float] = {}

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

            _, _, strat_exit_long, strat_exit_short = read_strategy_signals(
                self.data_provider, symbol,
            )
            fr = self._funding_rates.get(symbol, 0.0)
            regime_view = self.regime_arbiter.compute(
                signal,
                funding_rate=fr,
                strategy_exit_long=strat_exit_long,
                strategy_exit_short=strat_exit_short,
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

        # Stage 5 unstuck — same call as Backtester. Wall-clock time
        # drives both the 24h loss budget and any later staleness checks.
        unstuck_orders = self.risk.compute_unstuck_orders(
            self.account,
            grid_wallet_exposure_limit=self.config.grid.wallet_exposure_limit,
            now_ms=tick_ms,
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
                if side == "long":
                    tracked_trend = abs(ss.trend_long.size)
                    grid_size = max(size - tracked_trend, 0.0)
                    ss.position_long = Position(size=grid_size, entry_price=entry)
                elif side == "short":
                    tracked_trend = abs(ss.trend_short.size)
                    grid_size = max(size - tracked_trend, 0.0)
                    ss.position_short = Position(size=grid_size, entry_price=entry)
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
                for row in ohlcv:
                    self.trend.update(symbol, float(row[4]))
                if ohlcv:
                    last = ohlcv[-1]
                    ss = self.account.symbols[symbol]
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
                            pos.update_best_price(float(last[2]), Side.LONG)
                    for pos in (ss.position_short, ss.trend_short):
                        if pos.is_open:
                            pos.update_best_price(float(last[3]), Side.SHORT)
                    # Feed the strategy's rolling DataFrame view.
                    self.data_provider.append(
                        symbol,
                        Candle(
                            timestamp=int(last[0]),
                            open=float(last[1]),
                            high=float(last[2]),
                            low=float(last[3]),
                            close=float(last[4]),
                            volume=float(last[5]),
                        ),
                    )
                    if not ss.ema.initialized:
                        ss.ema.init([385.0, 620.0], float(last[4]))
                    else:
                        ss.ema.update(float(last[4]))
                    import math
                    high, low = float(last[2]), float(last[3])
                    lr = math.log(max(high, low + 1e-12) / low) if low > 0 else 0.0
                    if not ss.volatility.initialized:
                        ss.volatility.init(1000.0, lr)
                    else:
                        ss.volatility.update(lr)
            except Exception:
                logger.exception("Failed to fetch candles for %s", symbol)

    async def _reconcile_orders(self, desired: list[Order]):
        for symbol in self.config.symbols:
            try:
                existing = await self.exchange.fetch_open_orders(symbol)
            except Exception:
                existing = []
            self._open_orders[symbol] = existing

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
                    to_cancel.append(e)

            for i, d in enumerate(symbol_desired):
                if i not in matched_desired:
                    to_create.append(d)

        for e in to_cancel[:self.config.max_orders_per_batch]:
            await self._cancel_order(e)

        for d in to_create[:self.config.max_orders_per_batch]:
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
            logger.info("Cancelled %s %s", symbol, order_id)
        except Exception:
            logger.exception("Failed to cancel %s %s", symbol, order_id)

    async def _create_order(self, order: Order):
        side = order.exchange_side
        order_type = "market" if order.is_market else "limit"

        params = {}
        if order.reduce_only:
            params["reduceOnly"] = True

        if self.config.dry_run:
            logger.info(
                "[DRY] %s %s %s %.4f @ %.2f (%s%s)",
                order_type, side, order.symbol, order.qty, order.price,
                order.source.value, " reduce" if order.reduce_only else "",
            )
            return

        try:
            # Market orders ignore price; pass None to ccxt.
            price = None if order.is_market else order.price
            result = await self.exchange.create_order(
                order.symbol, order_type, side, order.qty, price, params
            )
            logger.info(
                "Created %s %s %s %.4f @ %s → %s",
                order_type, side, order.symbol, order.qty,
                f"{order.price:.2f}" if not order.is_market else "MKT",
                result.get("id", "?"),
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
        return [Order(
            symbol=symbol,
            side=regime.trend_overlay,
            price=price,
            qty=qty,
            source=OrderSource.TREND,
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
        state = {
            "balance": self.account.balance,
            "equity": self.account.equity,
            "equity_peak": self.account.equity_peak,
            "timestamp": int(time.time() * 1000),
        }
        try:
            Path(self.config.state_file).write_text(json.dumps(state, indent=2))
        except Exception:
            pass

    async def _load_state(self):
        try:
            data = json.loads(Path(self.config.state_file).read_text())
            self.account.equity_peak = data.get("equity_peak", 0)
        except Exception:
            pass
