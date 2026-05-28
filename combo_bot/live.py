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


class LiveTrader:
    def __init__(self, config: LiveConfig, exchange):
        self.config = config
        self.exchange = exchange
        self.grid = GridEngine(config.grid)
        self.trend = TrendEngine(config.trend)
        self.merger = DecisionMerger(config.merger)
        self.risk = RiskManager(config.risk)
        self.account = AccountState()
        self.exchange_params: dict[str, ExchangeParams] = {}
        self._running = False
        self._open_orders: dict[str, list[dict]] = {}

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

            mode_long = self.merger.compute_mode(signal, Side.LONG, ss.position_long)
            mode_short = self.merger.compute_mode(signal, Side.SHORT, ss.position_short)

            we_long = calc_wallet_exposure(
                self.account.balance, abs(ss.position_long.size),
                ss.position_long.entry_price, ep.c_mult,
            ) if ss.position_long.is_open else 0.0

            grid_long = self.grid.compute_orders(
                symbol, Side.LONG, ss.position_long, ss.ema, ss.volatility,
                self.account.balance, we_long, ep, mode_long,
            )
            grid_long = self.merger.filter_grid_orders(grid_long, signal, Side.LONG)

            we_short = calc_wallet_exposure(
                self.account.balance, abs(ss.position_short.size),
                ss.position_short.entry_price, ep.c_mult,
            ) if ss.position_short.is_open else 0.0

            grid_short = self.grid.compute_orders(
                symbol, Side.SHORT, ss.position_short, ss.ema, ss.volatility,
                self.account.balance, we_short, ep, mode_short,
            )
            grid_short = self.merger.filter_grid_orders(grid_short, signal, Side.SHORT)

            trend_entries = self.merger.generate_trend_orders(symbol, signal, price, self.account, ep)
            trend_exits_l = self.merger.generate_trend_exit_orders(symbol, ss.position_long, Side.LONG, price, ep)
            trend_exits_s = self.merger.generate_trend_exit_orders(symbol, ss.position_short, Side.SHORT, price, ep)

            all_desired_orders.extend(grid_long + grid_short + trend_entries + trend_exits_l + trend_exits_s)

        all_desired_orders = self.risk.filter_orders(
            all_desired_orders, self.account, int(time.time() * 1000)
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
                if side == "long":
                    ss.position_long = Position(size=size, entry_price=entry)
                elif side == "short":
                    ss.position_short = Position(size=size, entry_price=entry)
                ss.last_price = float(p.get("markPrice", 0) or ss.last_price)
        except Exception:
            logger.exception("Failed to refresh account")

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

        params = {}
        if order.reduce_only:
            params["reduceOnly"] = True

        if self.config.dry_run:
            logger.info(
                "[DRY] %s %s %.4f @ %.2f (%s%s)",
                side, order.symbol, order.qty, order.price,
                order.source.value, " reduce" if order.reduce_only else "",
            )
            return

        try:
            result = await self.exchange.create_order(
                order.symbol, "limit", side, order.qty, order.price, params
            )
            logger.info(
                "Created %s %s %.4f @ %.2f → %s",
                side, order.symbol, order.qty, order.price,
                result.get("id", "?"),
            )
        except Exception:
            logger.exception(
                "Failed to create %s %s %.4f @ %.2f",
                side, order.symbol, order.qty, order.price,
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
