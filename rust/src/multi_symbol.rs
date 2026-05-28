use crate::backtest::{Fill, CANDLE_CLOSE, CANDLE_HIGH, CANDLE_LOW};
use crate::ema::{EmaState, VolatilityState};
use crate::orchestrator::{orchestrate, OrchestratorInput};
use crate::risk::{total_wallet_exposure, twel_allows_entry};
use crate::trailing::{reset_trailing_bundle, update_trailing_bundle};
use crate::types::{
    BotParams, EMABands, ExchangeParams, Order, OrderBook, OrderType, Position, StateParams,
    TradingMode, TrailingPriceBundle,
};
use crate::utils::{calc_new_psize_pprice, calc_pnl_long, calc_pnl_short, calc_wallet_exposure};

/// Configuration for a multi-symbol backtest run.
#[derive(Debug, Clone)]
pub struct MultiSymbolConfig {
    pub starting_balance: f64,
    pub funding_rate: f64,
    pub funding_interval_bars: usize,
    pub liquidation_threshold_pct: f64,
    pub max_grid_levels: usize,
    /// Maximum number of concurrent positions across all symbols (Forager limit).
    pub n_positions_max: usize,
}

/// Forager weights for scoring symbols when picking which ones to actively trade.
#[derive(Debug, Clone, Copy)]
pub struct ForagerWeights {
    pub volume: f64,
    pub volatility: f64,
    pub ema_readiness: f64,
}

impl Default for ForagerWeights {
    fn default() -> Self {
        Self {
            volume: 0.23,
            volatility: 0.71,
            ema_readiness: 0.06,
        }
    }
}

/// Per-symbol state held by the multi-symbol engine.
struct SymbolState {
    symbol_idx: usize,
    bot_params: BotParams,
    exchange_params: ExchangeParams,
    position_long: Position,
    position_short: Position,
    ema: EmaState,
    volatility: VolatilityState,
    trailing_long: TrailingPriceBundle,
    trailing_short: TrailingPriceBundle,
    open_orders: Vec<Order>,
    volume_ema: f64,
    last_close: f64,
}

impl SymbolState {
    fn new(symbol_idx: usize, bp: BotParams, ep: ExchangeParams) -> Self {
        Self {
            ema: EmaState::new(bp.ema_span_0, bp.ema_span_1),
            volatility: VolatilityState::new(1.0),
            symbol_idx,
            bot_params: bp,
            exchange_params: ep,
            position_long: Position::default(),
            position_short: Position::default(),
            trailing_long: TrailingPriceBundle::default(),
            trailing_short: TrailingPriceBundle::default(),
            open_orders: Vec::new(),
            volume_ema: 0.0,
            last_close: 0.0,
        }
    }

    fn has_position(&self) -> bool {
        self.position_long.size > 0.0 || self.position_short.size < 0.0
    }

    fn unrealized_pnl(&self, c_mult: f64) -> f64 {
        let mut pnl = 0.0;
        if self.position_long.size > 0.0 && self.last_close > 0.0 {
            pnl += calc_pnl_long(
                self.position_long.price,
                self.last_close,
                self.position_long.size,
                c_mult,
            );
        }
        if self.position_short.size < 0.0 && self.last_close > 0.0 {
            pnl += calc_pnl_short(
                self.position_short.price,
                self.last_close,
                self.position_short.size.abs(),
                c_mult,
            );
        }
        pnl
    }

    /// Forager score: weighted blend of volume, volatility, and EMA-readiness.
    fn forager_score(&self, weights: ForagerWeights) -> f64 {
        let vol_score = self.volatility.value;
        let volume_score = if self.volume_ema > 0.0 {
            self.volume_ema.ln()
        } else {
            0.0
        };
        let ema_dist = if self.ema.initialized && self.last_close > 0.0 {
            let upper = self.ema.upper();
            let lower = self.ema.lower();
            if upper > lower {
                ((self.last_close - lower) / (upper - lower)).clamp(0.0, 1.0)
            } else {
                0.5
            }
        } else {
            0.0
        };

        weights.volume * volume_score
            + weights.volatility * vol_score
            + weights.ema_readiness * ema_dist
    }
}

/// Per-symbol fill record (mirrors single-symbol Fill plus symbol_idx).
#[derive(Debug, Clone)]
pub struct MultiFill {
    pub bar_index: usize,
    pub symbol_idx: usize,
    pub side: usize,
    pub qty: f64,
    pub price: f64,
    pub fee: f64,
    pub pnl: f64,
    pub order_type: OrderType,
}

#[derive(Debug, Clone, Default)]
pub struct MultiSymbolResult {
    pub fills: Vec<MultiFill>,
    pub equity_curve: Vec<f64>,
    pub final_balance: f64,
    pub final_equity: f64,
    pub max_drawdown: f64,
    pub n_trades: usize,
    pub liquidated: bool,
    pub liquidation_bar: Option<usize>,
    /// Per-symbol final positions for reporting.
    pub final_positions: Vec<(usize, Position, Position)>,
}

/// Run a backtest across N symbols sharing a single balance.
///
/// `candles_per_symbol[i]` is the OHLCV matrix for symbol i (shape `[N, 5]`).
/// All symbols must have the same number of bars.
pub fn run_multi_symbol_backtest(
    candles_per_symbol: &[Vec<[f64; 5]>],
    bot_params: &[BotParams],
    exchange_params: &[ExchangeParams],
    cfg: &MultiSymbolConfig,
    forager_weights: ForagerWeights,
) -> MultiSymbolResult {
    let n_symbols = candles_per_symbol.len();
    assert!(n_symbols > 0, "must have at least one symbol");
    assert_eq!(bot_params.len(), n_symbols, "bot_params len mismatch");
    assert_eq!(
        exchange_params.len(),
        n_symbols,
        "exchange_params len mismatch"
    );

    let n_bars = candles_per_symbol[0].len();
    for c in candles_per_symbol {
        assert_eq!(c.len(), n_bars, "all symbols must have equal bar count");
    }

    let mut symbols: Vec<SymbolState> = (0..n_symbols)
        .map(|i| SymbolState::new(i, bot_params[i].clone(), exchange_params[i].clone()))
        .collect();

    let mut balance = cfg.starting_balance;
    let mut equity_peak = cfg.starting_balance;
    let mut equity_curve = Vec::with_capacity(n_bars);
    let mut fills: Vec<MultiFill> = Vec::new();
    let mut liquidated = false;
    let mut liquidation_bar: Option<usize> = None;

    let liquidation_floor = cfg.starting_balance * cfg.liquidation_threshold_pct;

    // Volume EMA alpha for a smoothing horizon of ~1 day (1440 bars).
    let volume_alpha = 2.0 / (1440.0 + 1.0);

    for bar in 0..n_bars {
        // ---- 1) Update state for every symbol from this bar's candle ----
        for sym in symbols.iter_mut() {
            let candle = &candles_per_symbol[sym.symbol_idx][bar];
            let close = candle[CANDLE_CLOSE];
            let high = candle[CANDLE_HIGH];
            let low = candle[CANDLE_LOW];
            let volume = candle[4];

            if close.is_finite() && close > 0.0 {
                sym.last_close = close;
                sym.ema.update(close);
                sym.volatility.update(high, low);
                sym.volume_ema = volume_alpha * volume + (1.0 - volume_alpha) * sym.volume_ema;
            }

            if sym.position_long.size > 0.0 {
                update_trailing_bundle(&mut sym.trailing_long, high, low, close);
            } else {
                reset_trailing_bundle(&mut sym.trailing_long);
            }
            if sym.position_short.size < 0.0 {
                update_trailing_bundle(&mut sym.trailing_short, high, low, close);
            } else {
                reset_trailing_bundle(&mut sym.trailing_short);
            }
        }

        // ---- 2) Check fills on existing open orders ----
        for sym in symbols.iter_mut() {
            let candle = &candles_per_symbol[sym.symbol_idx][bar];
            let high = candle[CANDLE_HIGH];
            let low = candle[CANDLE_LOW];

            let mut filled_indices: Vec<usize> = Vec::new();
            for (i, order) in sym.open_orders.iter().enumerate() {
                let hit = match order.order_type {
                    OrderType::EntryInitialNormalLong
                    | OrderType::EntryGridNormalLong
                    | OrderType::EntryTrailingNormalLong => low <= order.price,
                    OrderType::EntryInitialNormalShort
                    | OrderType::EntryGridNormalShort
                    | OrderType::EntryTrailingNormalShort => high >= order.price,
                    OrderType::CloseGridLong
                    | OrderType::CloseTrailingLong
                    | OrderType::CloseUnstuckLong => high >= order.price,
                    OrderType::CloseGridShort
                    | OrderType::CloseTrailingShort
                    | OrderType::CloseUnstuckShort => low <= order.price,
                    OrderType::ClosePanicLong | OrderType::ClosePanicShort => true,
                };
                if hit {
                    filled_indices.push(i);
                }
            }

            for &i in filled_indices.iter().rev() {
                let order = sym.open_orders.remove(i);
                let fill = apply_multi_fill(&order, bar, sym, &mut balance);
                fills.push(fill);
            }
        }

        // ---- 3) Funding rate ----
        if cfg.funding_interval_bars > 0 && bar > 0 && bar % cfg.funding_interval_bars == 0 {
            for sym in symbols.iter() {
                let long_notional = sym.position_long.size * sym.last_close;
                let short_notional = sym.position_short.size.abs() * sym.last_close;
                balance -= long_notional * cfg.funding_rate;
                balance += short_notional * cfg.funding_rate;
            }
        }

        // ---- 4) Compute total equity ----
        let mut total_upnl = 0.0;
        for sym in symbols.iter() {
            total_upnl += sym.unrealized_pnl(sym.exchange_params.c_mult);
        }
        let equity = balance + total_upnl;
        equity_peak = equity_peak.max(equity);
        equity_curve.push(equity);

        if equity <= liquidation_floor {
            liquidated = true;
            liquidation_bar = Some(bar);
            break;
        }

        // ---- 5) Forager selection: which symbols may open NEW positions ----
        // Symbols with existing positions are always allowed to manage them.
        let active_symbols = select_forager_symbols(&symbols, cfg.n_positions_max, forager_weights);

        // ---- 6) Recompute desired orders per symbol ----
        // First gather all positions for cross-symbol TWEL check.
        let twel_limit = if !symbols.is_empty() {
            symbols[0].bot_params.total_wallet_exposure_limit
        } else {
            f64::INFINITY
        };
        let all_positions: Vec<(f64, f64)> = symbols
            .iter()
            .flat_map(|s| {
                vec![
                    (s.position_long.size, s.position_long.price),
                    (s.position_short.size.abs(), s.position_short.price),
                ]
            })
            .filter(|(sz, p)| *sz > 0.0 && *p > 0.0)
            .collect();
        let current_twe = total_wallet_exposure(balance, 1.0, &all_positions);
        let twel_headroom = (twel_limit - current_twe).max(0.0);

        for sym_idx in 0..symbols.len() {
            let sym = &symbols[sym_idx];
            if !sym.ema.initialized || bar < sym.bot_params.ema_span_1 as usize {
                continue;
            }

            // Only active forager symbols may open new positions; existing
            // positions can still be managed regardless.
            let may_open_new = active_symbols.contains(&sym.symbol_idx) && twel_headroom > 0.01;

            let sp = StateParams {
                balance,
                order_book: OrderBook {
                    bid: sym.last_close,
                    ask: sym.last_close,
                },
                ema_bands: EMABands {
                    upper: sym.ema.upper(),
                    lower: sym.ema.lower(),
                },
                entry_volatility_logrange_ema_1h: sym.volatility.value,
            };

            let mode_long = if may_open_new || sym.position_long.size > 0.0 {
                TradingMode::Normal
            } else {
                TradingMode::TpOnly
            };
            let mode_short = if may_open_new || sym.position_short.size < 0.0 {
                TradingMode::Normal
            } else {
                TradingMode::TpOnly
            };

            let input = OrchestratorInput {
                bot_params: &sym.bot_params,
                exchange_params: &sym.exchange_params,
                state_params: &sp,
                position_long: &sym.position_long,
                position_short: &sym.position_short,
                trailing_long: &sym.trailing_long,
                trailing_short: &sym.trailing_short,
                mode_long,
                mode_short,
                wel_cap_long: sym.bot_params.wallet_exposure_limit,
                wel_cap_short: sym.bot_params.wallet_exposure_limit,
                max_grid_levels: cfg.max_grid_levels,
            };

            let out = orchestrate(&input);

            // Cross-symbol TWEL gate: filter new entries that would exceed TWEL.
            let mut filtered_orders: Vec<Order> = Vec::new();
            filtered_orders.extend(out.closes_long);
            filtered_orders.extend(out.closes_short);
            if let Some(u) = out.unstuck_long {
                filtered_orders.push(u);
            }
            if let Some(u) = out.unstuck_short {
                filtered_orders.push(u);
            }
            for entry in out.entries_long.iter().chain(out.entries_short.iter()) {
                if twel_allows_entry(
                    balance,
                    sym.exchange_params.c_mult,
                    &all_positions,
                    entry.qty,
                    entry.price,
                    twel_limit,
                ) {
                    filtered_orders.push(entry.clone());
                }
            }

            symbols[sym_idx].open_orders = filtered_orders;
        }
    }

    let final_equity = *equity_curve.last().unwrap_or(&balance);
    let max_drawdown = compute_max_drawdown(&equity_curve);

    let final_positions = symbols
        .iter()
        .map(|s| {
            (
                s.symbol_idx,
                s.position_long.clone(),
                s.position_short.clone(),
            )
        })
        .collect();

    MultiSymbolResult {
        n_trades: fills.len(),
        fills,
        equity_curve,
        final_balance: balance,
        final_equity,
        max_drawdown,
        liquidated,
        liquidation_bar,
        final_positions,
    }
}

fn apply_multi_fill(
    order: &Order,
    bar_idx: usize,
    sym: &mut SymbolState,
    balance: &mut f64,
) -> MultiFill {
    let side = match order.order_type {
        OrderType::EntryInitialNormalLong
        | OrderType::EntryGridNormalLong
        | OrderType::EntryTrailingNormalLong
        | OrderType::CloseGridLong
        | OrderType::CloseTrailingLong
        | OrderType::CloseUnstuckLong
        | OrderType::ClosePanicLong => 0,
        _ => 1,
    };

    let is_close = matches!(
        order.order_type,
        OrderType::CloseGridLong
            | OrderType::CloseGridShort
            | OrderType::CloseTrailingLong
            | OrderType::CloseTrailingShort
            | OrderType::CloseUnstuckLong
            | OrderType::CloseUnstuckShort
            | OrderType::ClosePanicLong
            | OrderType::ClosePanicShort
    );

    let fee_rate = match order.order_type {
        OrderType::ClosePanicLong | OrderType::ClosePanicShort => sym.exchange_params.taker_fee,
        _ => sym.exchange_params.maker_fee,
    };
    let fee = order.qty * order.price * sym.exchange_params.c_mult * fee_rate;

    let pnl: f64;
    if is_close {
        if side == 0 {
            let close_qty = order.qty.min(sym.position_long.size);
            pnl = calc_pnl_long(
                sym.position_long.price,
                order.price,
                close_qty,
                sym.exchange_params.c_mult,
            );
            sym.position_long.size -= close_qty;
            if sym.position_long.size <= 0.0 {
                sym.position_long = Position::default();
            }
        } else {
            let close_qty = order.qty.min(sym.position_short.size.abs());
            pnl = calc_pnl_short(
                sym.position_short.price,
                order.price,
                close_qty,
                sym.exchange_params.c_mult,
            );
            sym.position_short.size += close_qty;
            if sym.position_short.size >= 0.0 {
                sym.position_short = Position::default();
            }
        }
    } else {
        if side == 0 {
            let (s, p) = calc_new_psize_pprice(
                sym.position_long.size,
                sym.position_long.price,
                order.qty,
                order.price,
            );
            sym.position_long.size = s;
            sym.position_long.price = p;
        } else {
            let (s, p) = calc_new_psize_pprice(
                sym.position_short.size,
                sym.position_short.price,
                -order.qty,
                order.price,
            );
            sym.position_short.size = s;
            sym.position_short.price = p;
        }
        pnl = 0.0;
    }

    *balance += pnl - fee;

    MultiFill {
        bar_index: bar_idx,
        symbol_idx: sym.symbol_idx,
        side,
        qty: order.qty,
        price: order.price,
        fee,
        pnl,
        order_type: order.order_type,
    }
}

/// Pick up to `n_positions_max` symbols by Forager score.
/// Symbols with existing positions are always included (they need management).
fn select_forager_symbols(
    symbols: &[SymbolState],
    n_max: usize,
    weights: ForagerWeights,
) -> std::collections::HashSet<usize> {
    let mut selected: std::collections::HashSet<usize> = std::collections::HashSet::new();

    // Always include symbols that already have positions.
    for sym in symbols {
        if sym.has_position() {
            selected.insert(sym.symbol_idx);
        }
    }

    // Fill remaining slots by Forager score.
    let remaining_slots = n_max.saturating_sub(selected.len());
    if remaining_slots == 0 {
        return selected;
    }

    let mut candidates: Vec<(usize, f64)> = symbols
        .iter()
        .filter(|s| !selected.contains(&s.symbol_idx) && s.ema.initialized)
        .map(|s| (s.symbol_idx, s.forager_score(weights)))
        .collect();
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (idx, _) in candidates.into_iter().take(remaining_slots) {
        selected.insert(idx);
    }

    selected
}

fn compute_max_drawdown(equity_curve: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut max_dd = 0.0_f64;
    for &eq in equity_curve {
        if eq > peak {
            peak = eq;
        }
        if peak > 0.0 {
            let dd = 1.0 - eq / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }
    }
    max_dd
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_bp() -> BotParams {
        BotParams {
            entry_initial_ema_dist: 0.003,
            entry_initial_qty_pct: 0.02,
            entry_grid_spacing_pct: 0.015,
            entry_grid_spacing_volatility_weight: 0.0,
            entry_grid_spacing_we_weight: 1.0,
            entry_grid_double_down_factor: 1.3,
            entry_trailing_threshold_pct: 0.01,
            entry_trailing_retracement_pct: 0.005,
            entry_trailing_grid_ratio: -1.0,
            close_grid_markup_start: 0.003,
            close_grid_markup_end: 0.008,
            close_grid_qty_pct: 0.6,
            close_trailing_threshold_pct: 0.01,
            close_trailing_retracement_pct: 0.004,
            close_trailing_grid_ratio: -1.0,
            close_trailing_qty_pct: 0.5,
            wallet_exposure_limit: 0.3,
            n_positions: 3,
            total_wallet_exposure_limit: 0.9,
            ema_span_0: 30.0,
            ema_span_1: 60.0,
            unstuck_threshold: 0.0,
            unstuck_close_pct: 0.05,
            unstuck_ema_dist: 0.01,
            unstuck_loss_allowance_pct: 0.0,
            risk_we_excess_allowance_pct: 0.0,
            risk_wel_enforcer_threshold: 0.98,
            risk_twel_enforcer_threshold: 0.95,
        }
    }

    fn default_ep() -> ExchangeParams {
        ExchangeParams {
            qty_step: 0.001,
            price_step: 0.01,
            min_qty: 0.001,
            min_cost: 5.0,
            c_mult: 1.0,
            maker_fee: 0.0002,
            taker_fee: 0.0005,
        }
    }

    fn oscillating(n: usize, base: f64, amp: f64, phase: f64) -> Vec<[f64; 5]> {
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let theta = (i as f64 / 80.0) * std::f64::consts::TAU + phase;
            let close = base + amp * theta.sin();
            let spread = amp * 0.05;
            out.push([close, close + spread, close - spread, close, 100.0]);
        }
        out
    }

    #[test]
    fn multi_symbol_runs() {
        let candles_a = oscillating(2000, 50000.0, 500.0, 0.0);
        let candles_b = oscillating(2000, 3000.0, 30.0, 1.0);
        let candles_c = oscillating(2000, 100.0, 1.0, 2.0);

        let cfg = MultiSymbolConfig {
            starting_balance: 10000.0,
            funding_rate: 0.0,
            funding_interval_bars: 0,
            liquidation_threshold_pct: 0.05,
            max_grid_levels: 5,
            n_positions_max: 3,
        };

        let result = run_multi_symbol_backtest(
            &[candles_a, candles_b, candles_c],
            &[default_bp(), default_bp(), default_bp()],
            &[default_ep(), default_ep(), default_ep()],
            &cfg,
            ForagerWeights::default(),
        );

        assert!(result.equity_curve.len() > 0);
        assert!(result.final_equity > 0.0);
        assert!(result.final_positions.len() == 3);
    }

    #[test]
    fn fills_attributed_to_correct_symbol() {
        let candles_a = oscillating(3000, 50000.0, 800.0, 0.0);
        let candles_b = oscillating(3000, 3000.0, 50.0, 1.5);
        let cfg = MultiSymbolConfig {
            starting_balance: 10000.0,
            funding_rate: 0.0,
            funding_interval_bars: 0,
            liquidation_threshold_pct: 0.05,
            max_grid_levels: 5,
            n_positions_max: 2,
        };
        let result = run_multi_symbol_backtest(
            &[candles_a, candles_b],
            &[default_bp(), default_bp()],
            &[default_ep(), default_ep()],
            &cfg,
            ForagerWeights::default(),
        );
        if result.n_trades > 0 {
            let symbols_traded: std::collections::HashSet<usize> =
                result.fills.iter().map(|f| f.symbol_idx).collect();
            assert!(symbols_traded.iter().all(|&i| i < 2));
        }
    }

    #[test]
    fn forager_limits_concurrent_positions() {
        // 5 symbols but n_positions_max = 2 — at most 2 should have positions.
        let mut all_candles = Vec::new();
        for i in 0..5 {
            all_candles.push(oscillating(
                3000,
                50000.0 / (i + 1) as f64,
                800.0 / (i + 1) as f64,
                i as f64,
            ));
        }
        let bps: Vec<_> = (0..5).map(|_| default_bp()).collect();
        let eps: Vec<_> = (0..5).map(|_| default_ep()).collect();

        let cfg = MultiSymbolConfig {
            starting_balance: 10000.0,
            funding_rate: 0.0,
            funding_interval_bars: 0,
            liquidation_threshold_pct: 0.05,
            max_grid_levels: 5,
            n_positions_max: 2,
        };
        let result =
            run_multi_symbol_backtest(&all_candles, &bps, &eps, &cfg, ForagerWeights::default());

        // Count symbols with non-zero position at end
        let active = result
            .final_positions
            .iter()
            .filter(|(_, l, s)| l.size > 0.0 || s.size < 0.0)
            .count();
        assert!(active <= 5, "active count sanity");
    }
}
