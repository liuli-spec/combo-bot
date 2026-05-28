from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from combo_bot.types import (
    AccountState, BacktestResult, Candle, ExchangeParams, Fill,
    Order, OrderSource, Position, Side, SymbolState, TradingMode,
)
from combo_bot.grid_engine import GridConfig, GridEngine, ForagerScorer, ForagerWeights, calc_wallet_exposure
from combo_bot.trend_signal import TrendConfig, TrendEngine
from combo_bot.merger import MergerConfig, DecisionMerger
from combo_bot.risk import RiskConfig, RiskManager


@dataclass
class BacktestConfig:
    starting_balance: float = 10000.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005
    slippage_pct: float = 0.0005
    funding_rate_default: float = 0.0001
    funding_interval_hours: int = 8
    liquidation_threshold: float = 0.05
    grid: GridConfig = field(default_factory=GridConfig)
    trend: TrendConfig = field(default_factory=TrendConfig)
    merger: MergerConfig = field(default_factory=MergerConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    forager_weights: ForagerWeights = field(default_factory=ForagerWeights)
    symbols: list[str] = field(default_factory=list)


class Backtester:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.grid = GridEngine(config.grid)
        self.trend = TrendEngine(config.trend)
        self.merger = DecisionMerger(config.merger)
        self.risk = RiskManager(config.risk)
        self.forager = ForagerScorer()

    def run(
        self,
        candle_data: dict[str, list[Candle]],
        funding_rates: dict[str, list[float]] | None = None,
        exchange_params: dict[str, ExchangeParams] | None = None,
    ) -> BacktestResult:
        symbols = list(candle_data.keys())
        n_steps = min(len(v) for v in candle_data.values())

        if exchange_params is None:
            exchange_params = {s: ExchangeParams() for s in symbols}

        account = AccountState(
            balance=self.config.starting_balance,
            equity=self.config.starting_balance,
            equity_peak=self.config.starting_balance,
        )
        for s in symbols:
            account.symbols[s] = SymbolState(symbol=s)

        fills: list[Fill] = []
        equity_log: list[tuple[int, float]] = []
        grid_pnl = 0.0
        trend_pnl = 0.0

        funding_hour_counter = 0

        for step in range(n_steps):
            candles = {s: candle_data[s][step] for s in symbols}
            ts = next(iter(candles.values())).timestamp

            self._update_prices(account, candles)
            self._update_emas(account, candles)
            self._update_volatility(account, candles)

            for s in symbols:
                self.trend.update(s, candles[s].close)

            account.update_equity()

            if self.risk.check_liquidation(account):
                break

            all_orders: list[Order] = []

            for s in symbols:
                ss = account.symbols[s]
                signal = self.trend.compute(s)
                ep = exchange_params[s]
                price = candles[s].close

                mode_long = self.merger.compute_mode(signal, Side.LONG, ss.position_long)
                mode_short = self.merger.compute_mode(signal, Side.SHORT, ss.position_short)
                ss.mode_long = mode_long
                ss.mode_short = mode_short

                we_long = calc_wallet_exposure(
                    account.balance, abs(ss.position_long.size),
                    ss.position_long.entry_price, ep.c_mult
                ) if ss.position_long.is_open else 0.0

                grid_long = self.grid.compute_orders(
                    symbol=s, side=Side.LONG, position=ss.position_long,
                    ema_state=ss.ema, volatility=ss.volatility,
                    balance=account.balance, wallet_exposure=we_long,
                    exchange_params=ep, mode=mode_long,
                )

                we_short = calc_wallet_exposure(
                    account.balance, abs(ss.position_short.size),
                    ss.position_short.entry_price, ep.c_mult
                ) if ss.position_short.is_open else 0.0

                grid_short = self.grid.compute_orders(
                    symbol=s, side=Side.SHORT, position=ss.position_short,
                    ema_state=ss.ema, volatility=ss.volatility,
                    balance=account.balance, wallet_exposure=we_short,
                    exchange_params=ep, mode=mode_short,
                )

                grid_long = self.merger.filter_grid_orders(grid_long, signal, Side.LONG)
                grid_short = self.merger.filter_grid_orders(grid_short, signal, Side.SHORT)

                trend_entries = self.merger.generate_trend_orders(
                    s, signal, price, account, ep
                )

                trend_exits_long = self.merger.generate_trend_exit_orders(
                    s, ss.position_long, Side.LONG, price, ep
                )
                trend_exits_short = self.merger.generate_trend_exit_orders(
                    s, ss.position_short, Side.SHORT, price, ep
                )

                all_orders.extend(grid_long)
                all_orders.extend(grid_short)
                all_orders.extend(trend_entries)
                all_orders.extend(trend_exits_long)
                all_orders.extend(trend_exits_short)

            all_orders = self.risk.filter_orders(all_orders, account, ts)

            step_fills = self._simulate_fills(all_orders, candles, account, exchange_params, ts)
            for f in step_fills:
                fills.append(f)
                account.balance += f.realized_pnl - f.fee
                if f.source == OrderSource.GRID:
                    grid_pnl += f.realized_pnl - f.fee
                else:
                    trend_pnl += f.realized_pnl - f.fee

            hours_elapsed = (step + 1) / 60.0
            if int(hours_elapsed / self.config.funding_interval_hours) > funding_hour_counter:
                funding_hour_counter = int(hours_elapsed / self.config.funding_interval_hours)
                fc = self._apply_funding(account, funding_rates, step, symbols)
                account.funding_cumsum += fc

            account.update_equity()
            equity_log.append((ts, account.equity))

        return self._compile_result(fills, equity_log, account, grid_pnl, trend_pnl)

    def _update_prices(self, account: AccountState, candles: dict[str, Candle]):
        for s, c in candles.items():
            account.symbols[s].last_price = c.close

    def _update_emas(self, account: AccountState, candles: dict[str, Candle]):
        for s, c in candles.items():
            ss = account.symbols[s]
            if not ss.ema.initialized:
                ss.ema.init(
                    [self.config.grid.ema_span_0, self.config.grid.ema_span_1],
                    c.close,
                )
            else:
                ss.ema.update(c.close)

    def _update_volatility(self, account: AccountState, candles: dict[str, Candle]):
        for s, c in candles.items():
            ss = account.symbols[s]
            if c.close > 0 and c.low > 0:
                log_range = math.log(max(c.high, c.low + 1e-12) / c.low) if c.low > 0 else 0.0
            else:
                log_range = 0.0
            if not ss.volatility.initialized:
                ss.volatility.init(self.config.grid.entry_volatility_ema_span_hours, log_range)
            else:
                ss.volatility.update(log_range)

    def _simulate_fills(
        self,
        orders: list[Order],
        candles: dict[str, Candle],
        account: AccountState,
        exchange_params: dict[str, ExchangeParams],
        timestamp: int,
    ) -> list[Fill]:
        step_fills = []
        closes_first = sorted(orders, key=lambda o: (0 if o.reduce_only else 1))

        for order in closes_first:
            c = candles.get(order.symbol)
            if c is None:
                continue
            ep = exchange_params.get(order.symbol, ExchangeParams())
            ss = account.symbols[order.symbol]

            filled = self._check_fill(order, c)
            if not filled:
                continue

            fill_price = order.price
            if order.source == OrderSource.TREND and not order.reduce_only:
                fill_price = c.close * (1.0 + self.config.slippage_pct * (1 if order.side == Side.LONG else -1))

            fee_rate = self.config.maker_fee if order.price != c.close else self.config.taker_fee
            fee = abs(order.qty) * fill_price * ep.c_mult * fee_rate

            pnl = 0.0
            if order.reduce_only:
                pos = ss.position_long if order.side == Side.LONG else ss.position_short
                if not pos.is_open:
                    continue
                close_qty = min(abs(order.qty), abs(pos.size))
                if order.side == Side.LONG:
                    pnl = close_qty * (fill_price - pos.entry_price) * ep.c_mult
                else:
                    pnl = close_qty * (pos.entry_price - fill_price) * ep.c_mult
                self._reduce_position(pos, close_qty)
            else:
                pos = ss.position_long if order.side == Side.LONG else ss.position_short
                self._add_to_position(pos, order.qty, fill_price)

            step_fills.append(Fill(
                timestamp=timestamp,
                symbol=order.symbol,
                side=order.side,
                price=fill_price,
                qty=order.qty,
                fee=fee,
                realized_pnl=pnl,
                source=order.source,
            ))

        return step_fills

    def _check_fill(self, order: Order, candle: Candle) -> bool:
        if order.side == Side.LONG:
            if order.reduce_only:
                return candle.high >= order.price
            return candle.low <= order.price
        else:
            if order.reduce_only:
                return candle.low <= order.price
            return candle.high >= order.price

    def _add_to_position(self, pos: Position, qty: float, price: float):
        if not pos.is_open:
            pos.size = qty
            pos.entry_price = price
        else:
            new_size = pos.size + qty
            if abs(new_size) > 1e-12:
                pos.entry_price = (pos.size * pos.entry_price + qty * price) / new_size
            pos.size = new_size

    def _reduce_position(self, pos: Position, close_qty: float):
        if abs(pos.size) <= close_qty + 1e-12:
            pos.size = 0.0
            pos.entry_price = 0.0
        else:
            pos.size = pos.size - close_qty if pos.size > 0 else pos.size + close_qty

    def _apply_funding(
        self,
        account: AccountState,
        funding_rates: dict[str, list[float]] | None,
        step: int,
        symbols: list[str],
    ) -> float:
        total = 0.0
        for s in symbols:
            ss = account.symbols[s]
            rate = self.config.funding_rate_default
            if funding_rates and s in funding_rates:
                idx = min(step, len(funding_rates[s]) - 1)
                if idx >= 0:
                    rate = funding_rates[s][idx]

            for side, pos in (
                (Side.LONG, ss.position_long),
                (Side.SHORT, ss.position_short),
            ):
                if pos.is_open:
                    notional = abs(pos.size) * ss.last_price
                    cost = notional * rate
                    if side == Side.SHORT:
                        cost = -cost
                    account.balance -= cost
                    total += cost
        return total

    def _compile_result(
        self,
        fills: list[Fill],
        equity_log: list[tuple[int, float]],
        account: AccountState,
        grid_pnl: float,
        trend_pnl: float,
    ) -> BacktestResult:
        eq_arr = np.array(equity_log)
        if len(eq_arr) < 2:
            return BacktestResult(
                fills=fills, equity_curve=eq_arr,
                final_balance=account.balance, total_pnl=0, total_fees=0,
                total_funding=account.funding_cumsum, n_trades=0,
                win_rate=0, adg=0, max_drawdown=0, sharpe_ratio=0,
                sortino_ratio=0, calmar_ratio=0, grid_pnl=grid_pnl,
                trend_pnl=trend_pnl, duration_days=0,
            )

        equities = eq_arr[:, 1]
        timestamps = eq_arr[:, 0]
        duration_ms = timestamps[-1] - timestamps[0]
        duration_days = duration_ms / 86_400_000.0

        total_pnl = sum(f.realized_pnl for f in fills)
        total_fees = sum(f.fee for f in fills)
        winning = sum(1 for f in fills if f.realized_pnl > 0 and f.realized_pnl != 0)
        closing = sum(1 for f in fills if f.realized_pnl != 0)
        win_rate = winning / max(closing, 1)

        daily_eq = _resample_daily(equities, timestamps)
        daily_returns = np.diff(daily_eq) / np.maximum(daily_eq[:-1], 1e-12)

        adg = 0.0
        if duration_days > 0 and equities[-1] > 0 and equities[0] > 0:
            adg = (equities[-1] / equities[0]) ** (1.0 / max(duration_days, 1)) - 1.0

        peak = np.maximum.accumulate(equities)
        drawdowns = (peak - equities) / np.maximum(peak, 1e-12)
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        sharpe = 0.0
        sortino = 0.0
        if len(daily_returns) > 1:
            mean_r = np.mean(daily_returns)
            std_r = np.std(daily_returns)
            if std_r > 1e-12:
                sharpe = mean_r / std_r * np.sqrt(365)
            downside = daily_returns[daily_returns < 0]
            ds_std = np.std(downside) if len(downside) > 0 else 0.0
            if ds_std > 1e-12:
                sortino = mean_r / ds_std * np.sqrt(365)

        calmar = adg * 365 / max(max_dd, 1e-12) if max_dd > 0 else 0.0

        return BacktestResult(
            fills=fills,
            equity_curve=eq_arr,
            final_balance=account.balance,
            total_pnl=total_pnl,
            total_fees=total_fees,
            total_funding=account.funding_cumsum,
            n_trades=len(fills),
            win_rate=win_rate,
            adg=adg,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            grid_pnl=grid_pnl,
            trend_pnl=trend_pnl,
            duration_days=duration_days,
        )


def _resample_daily(equities: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    if len(equities) == 0:
        return np.array([])
    ms_per_day = 86_400_000
    days = (timestamps - timestamps[0]) // ms_per_day
    unique_days = np.unique(days)
    daily = np.empty(len(unique_days))
    for i, d in enumerate(unique_days):
        mask = days == d
        daily[i] = equities[mask][-1]
    return daily
