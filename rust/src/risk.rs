use crate::types::{BotParams, ExchangeParams, Order, OrderType, Position, StateParams};
use crate::utils::{calc_pnl_long, calc_pnl_short, calc_wallet_exposure, cost_to_qty, round_up};

use crate::entries::calc_min_entry_qty;

// --- TWEL Enforcer ---
// Ported from Passivbot risk.rs: gate_entries_by_twel
// When total wallet exposure exceeds the limit, block new entries

pub fn total_wallet_exposure(
    balance: f64,
    c_mult: f64,
    positions: &[(f64, f64)], // (size, price) pairs
) -> f64 {
    if balance <= 0.0 {
        return 0.0;
    }
    positions
        .iter()
        .filter(|(size, price)| size.abs() > 1e-12 && *price > 0.0 && price.is_finite())
        .map(|(size, price)| calc_wallet_exposure(c_mult, balance, size.abs(), *price))
        .sum()
}

pub fn twel_allows_entry(
    balance: f64,
    c_mult: f64,
    current_positions: &[(f64, f64)],
    new_qty: f64,
    new_price: f64,
    twel_limit: f64,
) -> bool {
    if twel_limit <= 0.0 {
        return false;
    }
    let current_twe = total_wallet_exposure(balance, c_mult, current_positions);
    let entry_exposure = calc_wallet_exposure(c_mult, balance, new_qty, new_price);
    current_twe + entry_exposure <= twel_limit
}

// --- WEL Enforcer ---
// Ported from Passivbot closes.rs: calc_wel_auto_reduce
// When per-symbol wallet exposure exceeds threshold, auto-reduce position

pub struct WelReduceResult {
    pub close_qty: f64,
    pub close_price: f64,
    pub order_type: OrderType,
}

pub fn calc_wel_auto_reduce_long(
    exchange_params: &ExchangeParams,
    bot_params: &BotParams,
    position: &Position,
    wallet_exposure: f64,
    balance: f64,
    market_price: f64,
) -> Option<WelReduceResult> {
    if bot_params.risk_wel_enforcer_threshold <= 0.0 {
        return None;
    }
    let allowed = wallet_exposure_limit_with_allowance(bot_params);
    if allowed <= 0.0 {
        return None;
    }
    let target = allowed * bot_params.risk_wel_enforcer_threshold;
    if wallet_exposure <= target {
        return None;
    }
    let psize_abs = position.size.abs();
    if psize_abs <= f64::EPSILON || position.price <= 0.0 || balance <= 0.0 {
        return None;
    }
    let target_psize = (target * balance) / (position.price * exchange_params.c_mult);
    let reduce_qty = (psize_abs - target_psize).max(0.0);
    let close_qty = round_up(reduce_qty, exchange_params.qty_step)
        .max(calc_min_entry_qty(market_price, exchange_params))
        .min(psize_abs);
    if close_qty <= 0.0 {
        return None;
    }
    Some(WelReduceResult {
        close_qty,
        close_price: market_price,
        order_type: OrderType::CloseGridLong,
    })
}

pub fn calc_wel_auto_reduce_short(
    exchange_params: &ExchangeParams,
    bot_params: &BotParams,
    position: &Position,
    wallet_exposure: f64,
    balance: f64,
    market_price: f64,
) -> Option<WelReduceResult> {
    if bot_params.risk_wel_enforcer_threshold <= 0.0 {
        return None;
    }
    let allowed = wallet_exposure_limit_with_allowance(bot_params);
    if allowed <= 0.0 {
        return None;
    }
    let target = allowed * bot_params.risk_wel_enforcer_threshold;
    if wallet_exposure <= target {
        return None;
    }
    let psize_abs = position.size.abs();
    if psize_abs <= f64::EPSILON || position.price <= 0.0 || balance <= 0.0 {
        return None;
    }
    let target_psize = (target * balance) / (position.price * exchange_params.c_mult);
    let reduce_qty = (psize_abs - target_psize).max(0.0);
    let close_qty = round_up(reduce_qty, exchange_params.qty_step)
        .max(calc_min_entry_qty(market_price, exchange_params))
        .min(psize_abs);
    if close_qty <= 0.0 {
        return None;
    }
    Some(WelReduceResult {
        close_qty,
        close_price: market_price,
        order_type: OrderType::CloseGridShort,
    })
}

fn wallet_exposure_limit_with_allowance(bp: &BotParams) -> f64 {
    let base = bp.wallet_exposure_limit;
    if base <= 0.0 {
        base
    } else {
        base * (1.0 + bp.risk_we_excess_allowance_pct.max(0.0))
    }
}

// --- Unstucking ---
// Ported from Passivbot risk.rs: calc_unstucking_action
// When a position's WE is stuck at a high level, close a small portion at a loss

pub struct UnstuckResult {
    pub close_qty: f64,
    pub close_price: f64,
    pub projected_loss: f64,
}

pub fn calc_unstuck_close_long(
    exchange_params: &ExchangeParams,
    bot_params: &BotParams,
    position: &Position,
    wallet_exposure: f64,
    balance: f64,
    ema_band_lower: f64,
    loss_allowance_balance: f64,
) -> Option<UnstuckResult> {
    let wel = wallet_exposure_limit_with_allowance(bot_params);
    if wel <= 0.0 || bot_params.unstuck_threshold <= 0.0 {
        return None;
    }
    let we_ratio = wallet_exposure / wel;
    if we_ratio < bot_params.unstuck_threshold {
        return None;
    }
    let psize_abs = position.size.abs();
    if psize_abs <= f64::EPSILON || position.price <= 0.0 {
        return None;
    }

    let close_price = round_up(
        ema_band_lower * (1.0 + bot_params.unstuck_ema_dist),
        exchange_params.price_step,
    );
    if close_price <= 0.0 || close_price >= position.price {
        return None;
    }

    let close_qty = round_up(
        psize_abs * bot_params.unstuck_close_pct,
        exchange_params.qty_step,
    )
    .max(calc_min_entry_qty(close_price, exchange_params))
    .min(psize_abs);

    let projected_loss = calc_pnl_long(
        position.price,
        close_price,
        close_qty,
        exchange_params.c_mult,
    );
    if projected_loss.abs() > loss_allowance_balance {
        return None;
    }

    Some(UnstuckResult {
        close_qty,
        close_price,
        projected_loss,
    })
}

pub fn calc_unstuck_close_short(
    exchange_params: &ExchangeParams,
    bot_params: &BotParams,
    position: &Position,
    wallet_exposure: f64,
    balance: f64,
    ema_band_upper: f64,
    loss_allowance_balance: f64,
) -> Option<UnstuckResult> {
    let wel = wallet_exposure_limit_with_allowance(bot_params);
    if wel <= 0.0 || bot_params.unstuck_threshold <= 0.0 {
        return None;
    }
    let we_ratio = wallet_exposure / wel;
    if we_ratio < bot_params.unstuck_threshold {
        return None;
    }
    let psize_abs = position.size.abs();
    if psize_abs <= f64::EPSILON || position.price <= 0.0 {
        return None;
    }

    let close_price = crate::utils::round_dn(
        ema_band_upper * (1.0 - bot_params.unstuck_ema_dist.abs()),
        exchange_params.price_step,
    );
    if close_price <= 0.0 || close_price <= position.price {
        return None;
    }

    let close_qty = round_up(
        psize_abs * bot_params.unstuck_close_pct,
        exchange_params.qty_step,
    )
    .max(calc_min_entry_qty(close_price, exchange_params))
    .min(psize_abs);

    let projected_loss = calc_pnl_short(
        position.price,
        close_price,
        close_qty,
        exchange_params.c_mult,
    );
    if projected_loss.abs() > loss_allowance_balance {
        return None;
    }

    Some(UnstuckResult {
        close_qty,
        close_price,
        projected_loss,
    })
}

// --- Equity Hard Stop Loss (HSL) ---
// Ported from Passivbot equity_hard_stop_loss.rs
// 4-tier drawdown system with EMA smoothing

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HardStopTier {
    #[default]
    Green,
    Yellow,
    Orange,
    Red,
}

#[derive(Debug, Clone, Copy)]
pub struct HslConfig {
    pub red_threshold: f64,
    pub ema_span_minutes: f64,
    pub yellow_ratio: f64, // fraction of red_threshold for yellow (e.g. 0.5)
    pub orange_ratio: f64, // fraction of red_threshold for orange (e.g. 0.75)
}

#[derive(Debug, Clone)]
pub struct HslState {
    pub peak_equity: f64,
    pub drawdown_ema: f64,
    pub tier: HardStopTier,
    pub red_latched: bool,
    pub initialized: bool,
    pub last_minute: u64,
}

impl Default for HslState {
    fn default() -> Self {
        Self {
            peak_equity: 0.0,
            drawdown_ema: 0.0,
            tier: HardStopTier::Green,
            red_latched: false,
            initialized: false,
            last_minute: 0,
        }
    }
}

pub struct HslStep {
    pub drawdown_raw: f64,
    pub drawdown_score: f64,
    pub tier: HardStopTier,
    pub changed: bool,
}

pub fn hsl_update(state: &mut HslState, config: &HslConfig, equity: f64, minute: u64) -> HslStep {
    if !state.initialized {
        state.peak_equity = equity;
        state.drawdown_ema = 0.0;
        state.initialized = true;
        state.last_minute = minute;
        return HslStep {
            drawdown_raw: 0.0,
            drawdown_score: 0.0,
            tier: HardStopTier::Green,
            changed: false,
        };
    }

    state.peak_equity = state.peak_equity.max(equity);
    let drawdown_raw = if state.peak_equity > 0.0 {
        1.0 - equity / state.peak_equity
    } else {
        0.0
    };

    let elapsed = (minute.saturating_sub(state.last_minute)).max(1) as f64;
    let alpha = 1.0 - (-elapsed / config.ema_span_minutes.max(1.0)).exp();
    state.drawdown_ema = alpha * drawdown_raw + (1.0 - alpha) * state.drawdown_ema;
    state.last_minute = minute;

    let score = state.drawdown_ema;
    let prev_tier = state.tier;

    let yellow_threshold = config.red_threshold * config.yellow_ratio;
    let orange_threshold = config.red_threshold * config.orange_ratio;

    state.tier = if score >= config.red_threshold {
        state.red_latched = true;
        HardStopTier::Red
    } else if score >= orange_threshold {
        HardStopTier::Orange
    } else if score >= yellow_threshold {
        HardStopTier::Yellow
    } else {
        HardStopTier::Green
    };

    HslStep {
        drawdown_raw,
        drawdown_score: score,
        tier: state.tier,
        changed: state.tier != prev_tier,
    }
}

// --- Loss Gate ---
// Block close orders that would realize losses beyond a threshold

pub fn loss_gate_allows(
    realized_pnl_cumsum_peak: f64,
    realized_pnl_cumsum_current: f64,
    projected_pnl: f64,
    balance_peak: f64,
    max_realized_loss_pct: f64,
) -> bool {
    if max_realized_loss_pct >= 1.0 {
        return true;
    }
    if projected_pnl >= 0.0 {
        return true;
    }
    if balance_peak <= 0.0 {
        return false;
    }
    let floor = realized_pnl_cumsum_peak - balance_peak * max_realized_loss_pct;
    let projected_cumsum = realized_pnl_cumsum_current + projected_pnl;
    projected_cumsum >= floor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hsl_transitions_through_tiers() {
        let config = HslConfig {
            red_threshold: 0.10,
            ema_span_minutes: 1.0,
            yellow_ratio: 0.5,
            orange_ratio: 0.75,
        };
        let mut state = HslState::default();

        let step = hsl_update(&mut state, &config, 10000.0, 0);
        assert_eq!(step.tier, HardStopTier::Green);

        for i in 1..=100 {
            hsl_update(&mut state, &config, 10000.0 - (i as f64 * 12.0), i);
        }
        assert!(state.tier == HardStopTier::Red || state.tier == HardStopTier::Orange);
    }

    #[test]
    fn loss_gate_blocks_when_exceeded() {
        assert!(loss_gate_allows(100.0, 95.0, -3.0, 1000.0, 0.05));
        assert!(!loss_gate_allows(100.0, 55.0, -10.0, 1000.0, 0.05));
    }

    #[test]
    fn loss_gate_always_allows_profits() {
        assert!(loss_gate_allows(0.0, -100.0, 5.0, 1000.0, 0.01));
    }

    #[test]
    fn twel_blocks_when_exceeded() {
        let positions = vec![(1.0, 50000.0)]; // 1 BTC at 50k = 50k exposure
        assert!(!twel_allows_entry(
            10000.0, 1.0, &positions, 0.1, 50000.0, 3.0
        ));
    }

    #[test]
    fn twel_allows_when_under_limit() {
        let positions = vec![(0.01, 50000.0)]; // 0.01 BTC = 500 exposure on 10k balance = 0.05 WE
        assert!(twel_allows_entry(
            10000.0, 1.0, &positions, 0.01, 50000.0, 3.0
        ));
    }
}
