use crate::ema::{EmaState, VolatilityState};
use crate::orchestrator::{orchestrate, OrchestratorInput};
use crate::trailing::{reset_trailing_bundle, update_trailing_bundle};
use crate::types::{
    BotParams, EMABands, ExchangeParams, Order, OrderBook, OrderType, Position, StateParams,
    TradingMode, TrailingPriceBundle,
};
use crate::utils::{calc_new_psize_pprice, calc_pnl_long, calc_pnl_short};

pub const CANDLE_OPEN: usize = 0;
pub const CANDLE_HIGH: usize = 1;
pub const CANDLE_LOW: usize = 2;
pub const CANDLE_CLOSE: usize = 3;
pub const CANDLE_VOLUME: usize = 4;

#[derive(Debug, Clone)]
pub struct BacktestConfig {
    pub starting_balance: f64,
    pub funding_rate: f64,
    pub funding_interval_bars: usize,
    pub liquidation_threshold_pct: f64,
    pub max_grid_levels: usize,
}

#[derive(Debug, Clone)]
pub struct Fill {
    pub bar_index: usize,
    pub side: usize, // 0 = long, 1 = short
    pub qty: f64,
    pub price: f64,
    pub fee: f64,
    pub pnl: f64,
    pub order_type: OrderType,
}

#[derive(Debug, Clone, Default)]
pub struct BacktestResult {
    pub fills: Vec<Fill>,
    pub equity_curve: Vec<f64>,
    pub final_balance: f64,
    pub final_equity: f64,
    pub max_drawdown: f64,
    pub n_trades: usize,
    pub liquidated: bool,
    pub liquidation_bar: Option<usize>,
}

/// Per-symbol mutable state held across the backtest loop.
struct SymbolState {
    position_long: Position,
    position_short: Position,
    ema: EmaState,
    volatility: VolatilityState,
    trailing_long: TrailingPriceBundle,
    trailing_short: TrailingPriceBundle,
    open_orders: Vec<Order>,
}

impl SymbolState {
    fn new(bp: &BotParams) -> Self {
        Self {
            position_long: Position::default(),
            position_short: Position::default(),
            ema: EmaState::new(bp.ema_span_0, bp.ema_span_1),
            volatility: VolatilityState::new(1.0),
            trailing_long: TrailingPriceBundle::default(),
            trailing_short: TrailingPriceBundle::default(),
            open_orders: Vec::new(),
        }
    }
}

/// Run a backtest over `candles[bar][col]` where `col` indexes OHLCV.
///
/// Returns fills, equity curve, and key metrics. The backtest is single-symbol,
/// single-side-agnostic (both long and short grids run in parallel).
pub fn run_backtest(
    candles: &[[f64; 5]],
    bp: &BotParams,
    ep: &ExchangeParams,
    cfg: &BacktestConfig,
) -> BacktestResult {
    let mut state = SymbolState::new(bp);
    let mut balance = cfg.starting_balance;
    let mut equity_peak = cfg.starting_balance;
    let mut equity_curve = Vec::with_capacity(candles.len());
    let mut fills: Vec<Fill> = Vec::new();
    let mut liquidated = false;
    let mut liquidation_bar: Option<usize> = None;

    let liquidation_floor = cfg.starting_balance * cfg.liquidation_threshold_pct;

    for (bar_idx, candle) in candles.iter().enumerate() {
        let open = candle[CANDLE_OPEN];
        let high = candle[CANDLE_HIGH];
        let low = candle[CANDLE_LOW];
        let close = candle[CANDLE_CLOSE];

        if !close.is_finite() || close <= 0.0 {
            equity_curve.push(balance);
            continue;
        }

        // 1) Update EMAs, volatility, trailing extremes for this bar.
        state.ema.update(close);
        state.volatility.update(high, low);
        if state.position_long.size > 0.0 {
            update_trailing_bundle(&mut state.trailing_long, high, low, close);
        } else {
            reset_trailing_bundle(&mut state.trailing_long);
        }
        if state.position_short.size < 0.0 {
            update_trailing_bundle(&mut state.trailing_short, high, low, close);
        } else {
            reset_trailing_bundle(&mut state.trailing_short);
        }

        // 2) Check fills against this bar's high/low range.
        let mut new_fills: Vec<usize> = Vec::new();
        for (i, order) in state.open_orders.iter().enumerate() {
            let filled = match order.order_type {
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
                OrderType::ClosePanicLong => true,
                OrderType::ClosePanicShort => true,
            };
            if filled {
                new_fills.push(i);
            }
        }

        for &i in new_fills.iter().rev() {
            let order = state.open_orders.remove(i);
            let fill = apply_fill(&order, bar_idx, ep, &mut state, &mut balance);
            fills.push(fill);
        }

        // 3) Apply funding rate at scheduled intervals.
        if cfg.funding_interval_bars > 0 && bar_idx > 0 && bar_idx % cfg.funding_interval_bars == 0
        {
            apply_funding(&mut balance, &state, close, cfg.funding_rate);
        }

        // 4) Compute current equity (balance + unrealised pnl).
        let upnl_long = if state.position_long.size > 0.0 {
            calc_pnl_long(
                state.position_long.price,
                close,
                state.position_long.size,
                ep.c_mult,
            )
        } else {
            0.0
        };
        let upnl_short = if state.position_short.size < 0.0 {
            calc_pnl_short(
                state.position_short.price,
                close,
                state.position_short.size.abs(),
                ep.c_mult,
            )
        } else {
            0.0
        };
        let equity = balance + upnl_long + upnl_short;
        equity_peak = equity_peak.max(equity);
        equity_curve.push(equity);

        // 5) Liquidation check.
        if equity <= liquidation_floor {
            liquidated = true;
            liquidation_bar = Some(bar_idx);
            equity_curve.push(0.0);
            break;
        }

        // 6) Compute desired orders for next bar.
        if state.ema.initialized && bar_idx >= bp.ema_span_1 as usize {
            let sp = StateParams {
                balance,
                order_book: OrderBook {
                    bid: close,
                    ask: close,
                },
                ema_bands: EMABands {
                    upper: state.ema.upper(),
                    lower: state.ema.lower(),
                },
                entry_volatility_logrange_ema_1h: state.volatility.value,
            };

            let input = OrchestratorInput {
                bot_params: bp,
                exchange_params: ep,
                state_params: &sp,
                position_long: &state.position_long,
                position_short: &state.position_short,
                trailing_long: &state.trailing_long,
                trailing_short: &state.trailing_short,
                mode_long: TradingMode::Normal,
                mode_short: TradingMode::Normal,
                wel_cap_long: bp.wallet_exposure_limit,
                wel_cap_short: bp.wallet_exposure_limit,
                max_grid_levels: cfg.max_grid_levels,
            };

            let out = orchestrate(&input);
            state.open_orders.clear();
            state.open_orders.extend(out.entries_long);
            state.open_orders.extend(out.entries_short);
            state.open_orders.extend(out.closes_long);
            state.open_orders.extend(out.closes_short);
            if let Some(u) = out.unstuck_long {
                state.open_orders.push(u);
            }
            if let Some(u) = out.unstuck_short {
                state.open_orders.push(u);
            }
        }
    }

    let final_equity = *equity_curve.last().unwrap_or(&balance);
    let max_drawdown = compute_max_drawdown(&equity_curve);

    BacktestResult {
        n_trades: fills.len(),
        fills,
        equity_curve,
        final_balance: balance,
        final_equity,
        max_drawdown,
        liquidated,
        liquidation_bar,
    }
}

fn apply_fill(
    order: &Order,
    bar_idx: usize,
    ep: &ExchangeParams,
    state: &mut SymbolState,
    balance: &mut f64,
) -> Fill {
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
        OrderType::ClosePanicLong | OrderType::ClosePanicShort => ep.taker_fee,
        _ => ep.maker_fee,
    };
    let fee = order.qty * order.price * ep.c_mult * fee_rate;

    let pnl: f64;

    if is_close {
        if side == 0 {
            // Close long: reduce position_long.
            let close_qty = order.qty.min(state.position_long.size);
            pnl = calc_pnl_long(state.position_long.price, order.price, close_qty, ep.c_mult);
            state.position_long.size -= close_qty;
            if state.position_long.size <= 0.0 {
                state.position_long = Position::default();
            }
        } else {
            // Close short: reduce position_short (which is negative).
            let close_qty = order.qty.min(state.position_short.size.abs());
            pnl = calc_pnl_short(
                state.position_short.price,
                order.price,
                close_qty,
                ep.c_mult,
            );
            state.position_short.size += close_qty;
            if state.position_short.size >= 0.0 {
                state.position_short = Position::default();
            }
        }
    } else {
        // Entry: add to position.
        if side == 0 {
            let (new_size, new_price) = calc_new_psize_pprice(
                state.position_long.size,
                state.position_long.price,
                order.qty,
                order.price,
            );
            state.position_long.size = new_size;
            state.position_long.price = new_price;
        } else {
            let (new_size, new_price) = calc_new_psize_pprice(
                state.position_short.size,
                state.position_short.price,
                -order.qty,
                order.price,
            );
            state.position_short.size = new_size;
            state.position_short.price = new_price;
        }
        pnl = 0.0;
    }

    *balance += pnl - fee;

    Fill {
        bar_index: bar_idx,
        side,
        qty: order.qty,
        price: order.price,
        fee,
        pnl,
        order_type: order.order_type,
    }
}

fn apply_funding(balance: &mut f64, state: &SymbolState, price: f64, rate: f64) {
    let long_notional = state.position_long.size * price;
    let short_notional = state.position_short.size.abs() * price;

    // Convention: positive funding rate → long pays short.
    *balance -= long_notional * rate;
    *balance += short_notional * rate;
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
            entry_trailing_grid_ratio: -1.0, // pure grid for test stability
            close_grid_markup_start: 0.003,
            close_grid_markup_end: 0.008,
            close_grid_qty_pct: 0.6,
            close_trailing_threshold_pct: 0.01,
            close_trailing_retracement_pct: 0.004,
            close_trailing_grid_ratio: -1.0,
            close_trailing_qty_pct: 0.5,
            wallet_exposure_limit: 0.5,
            n_positions: 1,
            total_wallet_exposure_limit: 1.0,
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

    fn default_cfg() -> BacktestConfig {
        BacktestConfig {
            starting_balance: 10000.0,
            funding_rate: 0.0,
            funding_interval_bars: 0,
            liquidation_threshold_pct: 0.05,
            max_grid_levels: 5,
        }
    }

    fn oscillating_candles(n: usize, base: f64, amplitude: f64) -> Vec<[f64; 5]> {
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let phase = (i as f64 / 80.0) * std::f64::consts::TAU;
            let close = base + amplitude * phase.sin();
            let high = close + amplitude * 0.05;
            let low = close - amplitude * 0.05;
            out.push([close, high, low, close, 100.0]);
        }
        out
    }

    #[test]
    fn backtest_runs_without_panic() {
        let candles = oscillating_candles(1000, 50000.0, 500.0);
        let result = run_backtest(&candles, &default_bp(), &default_ep(), &default_cfg());
        assert!(!result.equity_curve.is_empty());
        assert!(result.final_equity > 0.0);
    }

    #[test]
    fn backtest_oscillating_market_is_profitable() {
        let candles = oscillating_candles(5000, 50000.0, 800.0);
        let result = run_backtest(&candles, &default_bp(), &default_ep(), &default_cfg());
        assert!(
            result.final_balance > 10000.0,
            "grid should profit in oscillating market: got {}",
            result.final_balance,
        );
        assert!(result.n_trades > 5);
    }

    #[test]
    fn backtest_records_drawdown() {
        let candles = oscillating_candles(1000, 50000.0, 500.0);
        let result = run_backtest(&candles, &default_bp(), &default_ep(), &default_cfg());
        assert!(result.max_drawdown >= 0.0 && result.max_drawdown <= 1.0);
    }
}
