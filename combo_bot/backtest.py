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
from combo_bot.strategy import DefaultStrategy, IStrategy, StrategyRunner, TradeContext
from combo_bot.data_provider import DataProvider
from combo_bot.regime import RegimeArbiter, RegimeArbiterConfig, read_strategy_signals
from combo_bot.protections import IProtection, ProtectionManager
from combo_bot.sizing import KellySizer
from combo_bot.correlation import CorrelationGate
from combo_bot.vol_target import VolTargetSizer
from combo_bot.types import RegimeView, TrendRegime, TrendSignal


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
    regime: RegimeArbiterConfig = field(default_factory=RegimeArbiterConfig)
    forager_weights: ForagerWeights = field(default_factory=ForagerWeights)
    symbols: list[str] = field(default_factory=list)


class Backtester:
    def __init__(
        self,
        config: BacktestConfig,
        strategy: IStrategy | None = None,
        data_provider_max_rows: int = 1000,
        protections: list[IProtection] | None = None,
        kelly_sizer: KellySizer | None = None,
        correlation_gate: CorrelationGate | None = None,
        vol_target_sizer: VolTargetSizer | None = None,
    ):
        self.config = config
        self.grid = GridEngine(config.grid)
        self.trend = TrendEngine(config.trend)
        self.merger = DecisionMerger(config.merger)
        self.risk = RiskManager(config.risk)
        self.regime_arbiter = RegimeArbiter(config.regime)
        self.forager = ForagerScorer()
        self.strategy: IStrategy = strategy or DefaultStrategy()
        self.strategy_runner = StrategyRunner(self.strategy)
        self.data_provider = DataProvider(max_rows=data_provider_max_rows)
        # Stage 7 — pluggable freqtrade-style protection rules. Default
        # to none so existing tests/configs see no behavior change;
        # users opt in by passing a list of IProtection instances.
        self.protections = ProtectionManager(protections or [])
        # Stage 9 — fractional-Kelly trend overlay sizing. None means
        # constant baseline sizing (preserves legacy behavior); pass a
        # KellySizer to throttle the overlay by realized edge.
        self.kelly_sizer = kelly_sizer
        # Stage 10 — cross-symbol correlation gate. None means no gating;
        # pass a CorrelationGate to scale or drop new entries that
        # would compound an already-crowded factor exposure.
        self.correlation_gate = correlation_gate
        # Stage 11 — portfolio-level vol-targeting sizer. None means no
        # scaling. Each tick the sizer sees the latest equity reading
        # and returns a multiplier applied to every new entry.
        self.vol_target_sizer = vol_target_sizer
        # Skip the per-tick DataFrame construction when the strategy
        # demonstrably doesn't consume signal columns. DefaultStrategy's
        # populate_* are no-ops, so for the common "engine-only" case we
        # avoid rebuilding pandas frames on every tick.
        self._strategy_uses_dataframe = (
            type(self.strategy).populate_entry_trend
            is not DefaultStrategy.populate_entry_trend
            or type(self.strategy).populate_exit_trend
            is not DefaultStrategy.populate_exit_trend
            or type(self.strategy).populate_indicators
            is not DefaultStrategy.populate_indicators
        )

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

        funding_hour_counter = 0

        for step in range(n_steps):
            candles = {s: candle_data[s][step] for s in symbols}
            ts = next(iter(candles.values())).timestamp

            self._update_prices(account, candles)
            self._update_emas(account, candles)
            self._update_volatility(account, candles)

            # Feed the rolling DataFrame view (only if the strategy reads it)
            # alongside the trend engine.
            for s in symbols:
                if self._strategy_uses_dataframe:
                    self.data_provider.append(s, candles[s])
                self.trend.update(s, candles[s].close)

            # Stage 10: feed the correlation tracker once per tick so
            # all pairwise correlations stay current. Cheap (one push
            # per symbol per tick).
            if self.correlation_gate is not None:
                self.correlation_gate.update_prices(
                    (s, candles[s].close) for s in symbols
                )

            account.update_equity()

            if self.risk.check_liquidation(account):
                break

            all_orders: list[Order] = []

            for s in symbols:
                ss = account.symbols[s]
                signal = self.trend.compute(s)
                ep = exchange_params[s]
                candle = candles[s]
                price = candle.close

                # Strategy signal from DataFrame's latest row (freqtrade convention).
                # Skip the DataFrame access entirely if the strategy doesn't write signals.
                if self._strategy_uses_dataframe:
                    _enter_long, _enter_short, strat_exit_long, strat_exit_short = (
                        read_strategy_signals(self.data_provider, s)
                    )
                else:
                    strat_exit_long = strat_exit_short = False

                # Funding rate at this tick (used by arbiter for overlay veto).
                fr = self._funding_rate_for(funding_rates, s, step)

                # Arbiter is the single source of mode / overlay / compression truth.
                regime_view = self.regime_arbiter.compute(
                    signal,
                    funding_rate=fr,
                    strategy_exit_long=strat_exit_long,
                    strategy_exit_short=strat_exit_short,
                )
                ss.mode_long = regime_view.long_mode
                ss.mode_short = regime_view.short_mode

                we_long = calc_wallet_exposure(
                    account.balance, abs(ss.position_long.size),
                    ss.position_long.entry_price, ep.c_mult
                ) if ss.position_long.is_open else 0.0

                grid_long = self.grid.compute_orders(
                    symbol=s, side=Side.LONG, position=ss.position_long,
                    ema_state=ss.ema, volatility=ss.volatility,
                    balance=account.balance, wallet_exposure=we_long,
                    exchange_params=ep, mode=regime_view.long_mode, mark_price=price,
                    close_markup_multiplier=regime_view.close_aggressiveness,
                )

                we_short = calc_wallet_exposure(
                    account.balance, abs(ss.position_short.size),
                    ss.position_short.entry_price, ep.c_mult
                ) if ss.position_short.is_open else 0.0

                grid_short = self.grid.compute_orders(
                    symbol=s, side=Side.SHORT, position=ss.position_short,
                    ema_state=ss.ema, volatility=ss.volatility,
                    balance=account.balance, wallet_exposure=we_short,
                    exchange_params=ep, mode=regime_view.short_mode, mark_price=price,
                    close_markup_multiplier=regime_view.close_aggressiveness,
                )

                # Strategy-layer hooks — applied AFTER the engine proposes orders
                # but BEFORE risk gating. The runner can veto, reprice, or
                # resize individual orders, and can also inject a forced exit.
                ctx_long = TradeContext(
                    symbol=s, side=Side.LONG, position=ss.position_long,
                    account=account, candle=candle, signal=signal,
                    current_time_ms=ts, exchange_params=ep,
                )
                ctx_short = TradeContext(
                    symbol=s, side=Side.SHORT, position=ss.position_short,
                    account=account, candle=candle, signal=signal,
                    current_time_ms=ts, exchange_params=ep,
                )
                grid_long = self.strategy_runner.filter_exits(
                    self.strategy_runner.filter_entries(grid_long, ctx_long),
                    ctx_long,
                )
                grid_short = self.strategy_runner.filter_exits(
                    self.strategy_runner.filter_entries(grid_short, ctx_short),
                    ctx_short,
                )

                # Stage 8 trailing re-entries (passivbot-style two-stage
                # trigger). No-op when entry_trailing_threshold_pct or
                # retracement_pct is 0 — config opts in.
                trailing_entries: list[Order] = []
                trail_long = self.grid.compute_trailing_entry(
                    s, Side.LONG, ss.position_long, ss.trailing_long,
                    account.balance, we_long, ep, price, regime_view.long_mode,
                )
                if trail_long is not None:
                    trailing_entries.extend(
                        self.strategy_runner.filter_entries([trail_long], ctx_long)
                    )
                trail_short = self.grid.compute_trailing_entry(
                    s, Side.SHORT, ss.position_short, ss.trailing_short,
                    account.balance, we_short, ep, price, regime_view.short_mode,
                )
                if trail_short is not None:
                    trailing_entries.extend(
                        self.strategy_runner.filter_entries([trail_short], ctx_short)
                    )

                # Trend overlay — direct emission driven by the arbiter.
                trend_entries = self._emit_trend_overlay(
                    s, regime_view, price, account, ep,
                )

                # Trend SL/TP only acts on the trend bucket — grid TP/SL
                # is managed by the grid engine's close ladder.
                trend_exits_long = self.merger.generate_trend_exit_orders(
                    s, ss.trend_long, Side.LONG, price, ep
                )
                trend_exits_short = self.merger.generate_trend_exit_orders(
                    s, ss.trend_short, Side.SHORT, price, ep
                )

                # Strategy-triggered forced exits and position adjustments
                # (custom_exit / custom_stoploss / adjust_trade_position).
                strategy_orders: list[Order] = []
                if ss.position_long.is_open:
                    fx = self.strategy_runner.check_custom_exit(ctx_long)
                    if fx is not None:
                        strategy_orders.append(fx)
                    adj = self.strategy_runner.check_position_adjustment(ctx_long)
                    if adj is not None:
                        strategy_orders.append(adj)
                if ss.position_short.is_open:
                    fx = self.strategy_runner.check_custom_exit(ctx_short)
                    if fx is not None:
                        strategy_orders.append(fx)
                    adj = self.strategy_runner.check_position_adjustment(ctx_short)
                    if adj is not None:
                        strategy_orders.append(adj)

                all_orders.extend(grid_long)
                all_orders.extend(grid_short)
                all_orders.extend(trailing_entries)
                all_orders.extend(trend_entries)
                all_orders.extend(trend_exits_long)
                all_orders.extend(trend_exits_short)
                all_orders.extend(strategy_orders)

            # Stage 7: protections filter — drop new entries for any
            # (symbol, side, source) currently locked by a protection
            # rule. Runs before unstuck/risk so an active lock
            # immediately stops new exposure even if risk is happy.
            # Reduce-only orders always pass.
            all_orders = self.protections.filter_orders(all_orders, ts)

            # Stage 10: correlation gate — scale or drop entries that
            # would compound a same-factor exposure. Runs AFTER
            # protections (so locks still pre-empt) but BEFORE unstuck/
            # risk (so the qty seen by the rest of the pipeline is the
            # gated size).
            if self.correlation_gate is not None:
                all_orders = self.correlation_gate.filter_orders(
                    all_orders, account,
                )

            # Stage 11: portfolio-level vol-targeting. Scales all
            # new entries by target_vol / realized_vol so the bot's
            # ex-ante risk stays stable across regimes. Runs LAST in
            # the sizing chain so it sees orders already throttled by
            # Kelly and correlation.
            if self.vol_target_sizer is not None:
                all_orders = self.vol_target_sizer.filter_orders(all_orders)

            # Stage 5: passivbot-style unstuck — emit small controlled
            # reduce-only limit orders for any bucket whose wallet
            # exposure crosses the unstuck threshold. Runs alongside
            # (not instead of) the regular close ladder; the fill
            # simulator deduplicates by limit price.
            unstuck_orders = self.risk.compute_unstuck_orders(
                account,
                grid_wallet_exposure_limit=self.config.grid.wallet_exposure_limit,
                now_ms=ts,
            )
            all_orders.extend(unstuck_orders)

            all_orders = self.risk.filter_orders(all_orders, account, ts)

            step_fills = self._simulate_fills(all_orders, candles, account, exchange_params, ts)
            for f in step_fills:
                fills.append(f)
                account.balance += f.realized_pnl - f.fee
                # Route P&L into the bucket the fill operated on. RISK
                # source routes to the grid bucket (matches the
                # fill-routing rule in SymbolState.bucket). Timestamp
                # threads through so the rolling-24h loss budget can be
                # measured against bar time, not wall-clock.
                account.add_realized_pnl(f.source, f.realized_pnl - f.fee, ts)

            # Feed this tick's fills back into protections so the next
            # tick sees up-to-date loss counts.
            self.protections.update(step_fills, account, ts)
            # Stage 9: feed closing fills into the Kelly sizer so the
            # next tick's overlay can use the updated edge estimate.
            if self.kelly_sizer is not None:
                self.kelly_sizer.record_fills(step_fills)

            hours_elapsed = (step + 1) / 60.0
            if int(hours_elapsed / self.config.funding_interval_hours) > funding_hour_counter:
                funding_hour_counter = int(hours_elapsed / self.config.funding_interval_hours)
                fc = self._apply_funding(account, funding_rates, step, symbols)
                account.funding_cumsum += fc

            account.update_equity()
            equity_log.append((ts, account.equity))

        return self._compile_result(
            fills, equity_log, account,
            account.grid_realized_pnl, account.trend_realized_pnl,
        )

    def _update_prices(self, account: AccountState, candles: dict[str, Candle]):
        for s, c in candles.items():
            ss = account.symbols[s]
            ss.last_price = c.close
            # Ratchet the high-water-mark used by trailing stops across
            # both grid and trend buckets so source-isolated trailing
            # stops still tighten with favorable moves.
            for pos in (ss.position_long, ss.trend_long):
                if pos.is_open:
                    pos.update_best_price(c.high, Side.LONG)
            for pos in (ss.position_short, ss.trend_short):
                if pos.is_open:
                    pos.update_best_price(c.low, Side.SHORT)
            # Stage 8 trailing-entry price bundles. Only feed candle
            # extremes while the position is open; bundles get re-seeded
            # on each position open in _add_to_position.
            if ss.position_long.is_open:
                ss.trailing_long.update_long(c.high, c.low)
            if ss.position_short.is_open:
                ss.trailing_short.update_short(c.high, c.low)

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

    def _funding_rate_for(
        self,
        funding_rates: dict[str, list[float]] | None,
        symbol: str,
        step: int,
    ) -> float:
        if funding_rates and symbol in funding_rates:
            idx = min(step, len(funding_rates[symbol]) - 1)
            if idx >= 0:
                return funding_rates[symbol][idx]
        return self.config.funding_rate_default

    def _emit_trend_overlay(
        self,
        symbol: str,
        regime: RegimeView,
        price: float,
        account: AccountState,
        exchange: ExchangeParams,
    ) -> list[Order]:
        """Convert the arbiter's overlay decision into an entry order.

        Sizing uses merger.trend_position_max_pct as the budget ceiling and
        merger.trend_entry_qty_pct as the first-entry fraction, scaled by
        the arbiter's conviction-derived qty scale. Stage 5 will replace
        this with a TrendOverlay class that pyramids on ATR moves.
        """
        if regime.trend_overlay is None or regime.trend_qty_scale <= 0:
            return []

        ss = account.symbols.get(symbol)
        if ss is not None:
            existing = (
                ss.trend_long if regime.trend_overlay == Side.LONG
                else ss.trend_short
            )
            # Don't double up the first trend entry. Pyramiding is Stage 5+.
            # Note: we check the trend bucket only — a co-existing grid
            # position on the same side is fine and is the point of overlay.
            if existing.is_open:
                return []

        merger_cfg = self.config.merger
        # Stage 9: fractional Kelly throttle. Multiplies the arbiter's
        # qty scale by an edge-driven [0, 1] factor. Below min_samples
        # the sizer returns 1.0 (cold start, no throttle). When the
        # rolling mean edge turns negative the sizer returns 0.0 and
        # the overlay self-suspends.
        kelly_scale = (
            self.kelly_sizer.fraction(OrderSource.TREND)
            if self.kelly_sizer is not None
            else 1.0
        )
        effective_scale = regime.trend_qty_scale * kelly_scale
        if effective_scale <= 0:
            return []

        budget = account.balance * merger_cfg.trend_position_max_pct
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

            # Market orders cross the book at close ± slippage with taker fee.
            # Trend entries also cross aggressively (legacy behavior).
            # Grid limits fill at their stated price with maker fee.
            if order.is_market:
                slip_dir = 1 if (order.side == Side.LONG and not order.reduce_only) or (order.side == Side.SHORT and order.reduce_only) else -1
                fill_price = c.close * (1.0 + self.config.slippage_pct * slip_dir)
                fee_rate = self.config.taker_fee
            elif order.source == OrderSource.TREND and not order.reduce_only:
                fill_price = c.close * (1.0 + self.config.slippage_pct * (1 if order.side == Side.LONG else -1))
                fee_rate = self.config.taker_fee
            else:
                fill_price = order.price
                fee_rate = self.config.maker_fee
            fee = abs(order.qty) * fill_price * ep.c_mult * fee_rate

            pnl = 0.0
            # Route fills to the bucket designated by order.source so grid
            # and trend P&L stay isolated. RISK falls back to grid for
            # backward compat with strategy custom_exit.
            pos = ss.bucket(order.source, order.side)
            if order.reduce_only:
                if not pos.is_open:
                    continue
                close_qty = min(abs(order.qty), abs(pos.size))
                if order.side == Side.LONG:
                    pnl = close_qty * (fill_price - pos.entry_price) * ep.c_mult
                else:
                    pnl = close_qty * (pos.entry_price - fill_price) * ep.c_mult
                self._reduce_position(pos, close_qty)
            else:
                # Seed the trailing bundle on a fresh open of the grid
                # bucket — this is the only entry path that drives the
                # passivbot-style trailing re-entry, so we don't reset
                # for trend-bucket fills.
                fresh_grid_open = (
                    not pos.is_open
                    and order.source != OrderSource.TREND
                )
                self._add_to_position(pos, order.qty, fill_price)
                if fresh_grid_open:
                    bundle = (
                        ss.trailing_long if order.side == Side.LONG
                        else ss.trailing_short
                    )
                    bundle.reset(fill_price)

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
        if order.is_market:
            return True
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
            pos.best_price = price
        else:
            new_size = pos.size + qty
            if abs(new_size) > 1e-12:
                pos.entry_price = (pos.size * pos.entry_price + qty * price) / new_size
            pos.size = new_size

    def _reduce_position(self, pos: Position, close_qty: float):
        if abs(pos.size) <= close_qty + 1e-12:
            pos.size = 0.0
            pos.entry_price = 0.0
            pos.best_price = 0.0
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

            # Funding hits every open bucket — perp exchanges charge the
            # full position regardless of which strategy opened it.
            for side, pos in (
                (Side.LONG, ss.position_long),
                (Side.LONG, ss.trend_long),
                (Side.SHORT, ss.position_short),
                (Side.SHORT, ss.trend_short),
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
