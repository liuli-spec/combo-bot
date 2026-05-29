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
    # Bar cadence assumed for the input candles. Defaults to 1m (legacy
    # behaviour); set to 60.0 for hourly bars, etc. Wired into:
    #   * funding-application loop (every funding_interval_hours of wall
    #     clock, not every funding_interval_hours * 60 bars);
    #   * VolatilityState.init so EMA span matches the cadence;
    # If you backtest 1h candles with the default 1.0 you get funding
    # applied ~60× too rarely and a volatility EMA ~60× too slow.
    bar_interval_minutes: float = 1.0
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
        # Orders computed at tick N, filled at tick N+1 (one-bar delay
        # eliminates look-ahead bias — matches Rust backtest semantics).
        pending_orders: list[Order] = []

        for step in range(n_steps):
            candles = {s: candle_data[s][step] for s in symbols}
            ts = next(iter(candles.values())).timestamp

            # ── update state with current bar ──────────────────────
            self._update_prices(account, candles)
            self._update_emas(account, candles)
            self._update_volatility(account, candles)
            for s in symbols:
                if self._strategy_uses_dataframe:
                    self.data_provider.append(s, candles[s])
                    # Apply the strategy's populate_* callbacks so the
                    # signal columns (enter_long / enter_short / etc.)
                    # actually exist on the cached DataFrame that
                    # read_strategy_signals reads from. Without this
                    # call the populate hooks were dead code — the
                    # strategy's entry/exit logic never reached the
                    # regime arbiter, so user-defined signals silently
                    # never fired. Strategies are expected to mutate
                    # the DataFrame in place (freqtrade convention);
                    # callbacks that return a new DataFrame won't
                    # propagate back to the cache.
                    self._apply_strategy_populates(s)
                self.trend.update(s, candles[s].close)
            if self.correlation_gate is not None:
                self.correlation_gate.update_prices(
                    (s, candles[s].close) for s in symbols
                )

            # ── fill pending orders from PREVIOUS tick ─────────────
            step_fills = self._simulate_fills(
                pending_orders, candles, account, exchange_params, ts,
            )
            for f in step_fills:
                fills.append(f)
                account.balance += f.realized_pnl - f.fee
                account.add_realized_pnl(f.source, f.realized_pnl - f.fee, ts)
            self.protections.update(step_fills, account, ts)
            if self.kelly_sizer is not None:
                self.kelly_sizer.record_fills(step_fills)

            # ── funding ───────────────────────────────────────────
            # Wall-clock hours, not bars-since-start. Without scaling by
            # bar_interval, 1h backtests would only trigger funding every
            # 480h instead of every 8h.
            hours_elapsed = (step + 1) * self.config.bar_interval_minutes / 60.0
            if int(hours_elapsed / self.config.funding_interval_hours) > funding_hour_counter:
                funding_hour_counter = int(hours_elapsed / self.config.funding_interval_hours)
                fc = self._apply_funding(account, funding_rates, step, symbols)
                account.funding_cumsum += fc

            account.update_equity()
            equity_log.append((ts, account.equity))

            # Stage 11: feed the vol-target sizer so its rolling
            # equity-return history stays current every tick.
            if self.vol_target_sizer is not None:
                self.vol_target_sizer.record_equity(account.equity)

            if self.risk.check_liquidation(account):
                break

            # ── compute orders for NEXT tick ──────────────────────
            all_orders: list[Order] = []

            for s in symbols:
                ss = account.symbols[s]
                signal = self.trend.compute(s)
                ep = exchange_params[s]
                candle = candles[s]
                price = candle.close

                if self._strategy_uses_dataframe:
                    strat_enter_long, strat_enter_short, strat_exit_long, strat_exit_short = (
                        read_strategy_signals(self.data_provider, s)
                    )
                else:
                    strat_enter_long = strat_enter_short = False
                    strat_exit_long = strat_exit_short = False
                fr = self._funding_rate_for(funding_rates, s, step)
                regime_view = self.regime_arbiter.compute(
                    signal,
                    funding_rate=fr,
                    strategy_exit_long=strat_exit_long,
                    strategy_exit_short=strat_exit_short,
                    strategy_enter_long=strat_enter_long,
                    strategy_enter_short=strat_enter_short,
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
                ctx_long = TradeContext(
                    symbol=s, side=Side.LONG, position=ss.position_long,
                    account=account, candle=candle, signal=signal,
                    current_time_ms=ts, exchange_params=ep,
                )
                trailing_entries: list[Order] = []
                ctx_short = TradeContext(
                    symbol=s, side=Side.SHORT, position=ss.position_short,
                    account=account, candle=candle, signal=signal,
                    current_time_ms=ts, exchange_params=ep,
                )
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
                grid_long = self.strategy_runner.filter_exits(
                    self.strategy_runner.filter_entries(grid_long, ctx_long),
                    ctx_long,
                )
                grid_short = self.strategy_runner.filter_exits(
                    self.strategy_runner.filter_entries(grid_short, ctx_short),
                    ctx_short,
                )

                trend_entries = self._emit_trend_overlay(
                    s, regime_view, price, account, ep,
                )
                # Run trend overlay entries through the strategy's veto
                # / price / sizing pipeline using a context whose
                # position is the TREND bucket (not the grid bucket).
                # Without this, confirm_trade_entry, custom_entry_price,
                # custom_stake_amount, and adjust_entry_price all had no
                # effect on trend overlay — a real safety gap because
                # the overlay is the most aggressive entry path the bot
                # emits.
                if trend_entries:
                    overlay_side = trend_entries[0].side
                    overlay_pos = (
                        ss.trend_long if overlay_side == Side.LONG
                        else ss.trend_short
                    )
                    ctx_overlay = TradeContext(
                        symbol=s, side=overlay_side, position=overlay_pos,
                        account=account, candle=candle, signal=signal,
                        current_time_ms=ts, exchange_params=ep,
                    )
                    trend_entries = self.strategy_runner.filter_entries(
                        trend_entries, ctx_overlay,
                    )
                trend_exits_long = self.merger.generate_trend_exit_orders(
                    s, ss.trend_long, Side.LONG, price, ep,
                )
                trend_exits_short = self.merger.generate_trend_exit_orders(
                    s, ss.trend_short, Side.SHORT, price, ep,
                )

                strategy_orders: list[Order] = []
                if ss.position_long.is_open:
                    fx = self.strategy_runner.check_custom_exit(ctx_long)
                    if fx is not None:
                        strategy_orders.append(fx)
                    # ctx.position is the grid bucket; the runner emits
                    # source=GRID so the fill lands in the same bucket
                    # the strategy is reasoning about.
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

            all_orders = self.protections.filter_orders(all_orders, ts)
            if self.correlation_gate is not None:
                all_orders = self.correlation_gate.filter_orders(
                    all_orders, account,
                )
            if self.vol_target_sizer is not None:
                all_orders = self.vol_target_sizer.filter_orders(all_orders)
            unstuck_orders = self.risk.compute_unstuck_orders(
                account,
                grid_wallet_exposure_limit=self.config.grid.wallet_exposure_limit,
                now_ms=ts,
                exchange_params=exchange_params,
            )
            all_orders.extend(unstuck_orders)
            all_orders = self.risk.filter_orders(all_orders, account, ts)

            pending_orders = all_orders

        return self._compile_result(
            fills, equity_log, account,
            account.grid_realized_pnl, account.trend_realized_pnl,
        )

    def _apply_strategy_populates(self, symbol: str) -> None:
        """Run the strategy's populate_indicators/entry/exit hooks on the
        cached DataFrame so signal columns become readable by
        :func:`combo_bot.regime.read_strategy_signals`.

        Strategies should mutate the DataFrame in place per freqtrade
        convention; if the hook returns a NEW DataFrame, we copy any
        signal columns it added back into the cache so reads still see
        them.
        """
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
            # A misbehaving user strategy mustn't crash the backtest loop.
            return
        final = out_ext if out_ext is not None else df
        if final is df:
            return
        # Copy any signal columns the strategy added on a copy of the
        # frame back into the cached frame so downstream reads see them.
        for col in ("enter_long", "enter_short", "exit_long", "exit_short",
                    "enter_tag", "exit_tag"):
            if col in final.columns and col not in df.columns:
                df[col] = final[col]

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
                ss.volatility.init(
                    self.config.grid.entry_volatility_ema_span_hours,
                    log_range,
                    bar_interval_minutes=self.config.bar_interval_minutes,
                )
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

        # Trend overlay entries cross the book immediately — the whole
        # point of overlay activation is that we want exposure NOW in a
        # strong regime, not when a limit price might (or might not) be
        # touched. is_market=True makes the fill simulator and the live
        # executor agree on the same semantics. (Previously is_market
        # was False but a special-case in the fill simulator filled at
        # close*slip anyway, so backtest and live diverged.)
        return [Order(
            symbol=symbol,
            side=regime.trend_overlay,
            price=price,
            qty=qty,
            source=OrderSource.TREND,
            is_market=True,
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

            # Backstop: enforce the same exchange constraints the live
            # executor's _quantize_order_for_send applies, so any upstream
            # generator that forgot to quantize / validate (e.g. a sizer
            # that scaled qty below qty_step, an unstuck path with no
            # ep, a strategy adjust returning a sub-step delta) is
            # rejected by the simulator too. Without this guard, backtest
            # quietly fills orders that live would never accept, and the
            # two paths report different fill streams.
            if ep.qty_step > 0:
                # Match live: floor to qty_step before validating.
                import math
                qstep_qty = math.floor(order.qty / ep.qty_step) * ep.qty_step
                if qstep_qty < ep.min_qty:
                    continue
                if qstep_qty * order.price * ep.c_mult < ep.min_cost:
                    continue
                if qstep_qty != order.qty:
                    # Quantize the order in place for the rest of the
                    # fill pipeline so fee / pnl / position math all
                    # use the same qty the live executor would send.
                    from dataclasses import replace
                    order = replace(order, qty=qstep_qty)

            # Market orders cross the book at close ± slippage with taker fee.
            # Grid limits fill at their stated price with maker fee.
            # (Trend overlay entries are now is_market=True at the source,
            # so they take the market branch — the old elif special-case
            # that filled them at close*slip while is_market was False
            # has been removed.)
            if order.is_market:
                slip_dir = 1 if (order.side == Side.LONG and not order.reduce_only) or (order.side == Side.SHORT and order.reduce_only) else -1
                fill_price = c.close * (1.0 + self.config.slippage_pct * slip_dir)
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
        # Net PnL (after fees) determines win/loss, not gross.
        winning = sum(1 for f in fills if f.realized_pnl > f.fee and f.realized_pnl != 0)
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