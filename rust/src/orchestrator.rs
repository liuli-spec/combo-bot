use crate::closes::{calc_closes_long, calc_closes_short};
use crate::entries::calc_entries_long;
use crate::entries::calc_entries_short;
use crate::risk::{calc_unstuck_close_long, calc_unstuck_close_short};
use crate::types::{
    BotParams, ExchangeParams, Order, OrderType, Position, StateParams, TradingMode,
    TrailingPriceBundle,
};
use crate::utils::calc_wallet_exposure;

/// Output of a single orchestration tick.
#[derive(Debug, Clone, Default)]
pub struct OrchestratorOutput {
    pub entries_long: Vec<Order>,
    pub entries_short: Vec<Order>,
    pub closes_long: Vec<Order>,
    pub closes_short: Vec<Order>,
    pub unstuck_long: Option<Order>,
    pub unstuck_short: Option<Order>,
}

impl OrchestratorOutput {
    pub fn all_orders(&self) -> Vec<&Order> {
        let mut out: Vec<&Order> = Vec::new();
        out.extend(self.closes_long.iter());
        out.extend(self.closes_short.iter());
        if let Some(o) = &self.unstuck_long {
            out.push(o);
        }
        if let Some(o) = &self.unstuck_short {
            out.push(o);
        }
        out.extend(self.entries_long.iter());
        out.extend(self.entries_short.iter());
        out
    }
}

pub struct OrchestratorInput<'a> {
    pub bot_params: &'a BotParams,
    pub exchange_params: &'a ExchangeParams,
    pub state_params: &'a StateParams,
    pub position_long: &'a Position,
    pub position_short: &'a Position,
    pub trailing_long: &'a TrailingPriceBundle,
    pub trailing_short: &'a TrailingPriceBundle,
    pub mode_long: TradingMode,
    pub mode_short: TradingMode,
    pub wel_cap_long: f64,
    pub wel_cap_short: f64,
    pub max_grid_levels: usize,
}

/// Compute every order the engine wants to be live for one symbol.
///
/// This is the per-tick decision point: given the current market state, position,
/// and trailing extremes, produce the complete set of desired open orders
/// (entries + closes + risk-driven actions) on both sides.
pub fn orchestrate(input: &OrchestratorInput) -> OrchestratorOutput {
    let mut out = OrchestratorOutput::default();

    // ---------- LONG side ----------
    match input.mode_long {
        TradingMode::Panic => {
            if input.position_long.size > 0.0 {
                out.closes_long.push(Order {
                    qty: input.position_long.size,
                    price: input.state_params.order_book.bid,
                    order_type: OrderType::ClosePanicLong,
                });
            }
        }
        TradingMode::GracefulStop | TradingMode::TpOnly => {
            if input.position_long.size > 0.0 {
                out.closes_long = calc_closes_long(
                    input.exchange_params,
                    input.state_params,
                    input.bot_params,
                    input.position_long,
                    input.trailing_long,
                );
            }
        }
        TradingMode::Normal => {
            if input.position_long.size > 0.0 {
                out.closes_long = calc_closes_long(
                    input.exchange_params,
                    input.state_params,
                    input.bot_params,
                    input.position_long,
                    input.trailing_long,
                );
            }
            out.entries_long = calc_entries_long(
                input.exchange_params,
                input.state_params,
                input.bot_params,
                input.position_long,
                input.trailing_long,
                input.wel_cap_long,
                input.max_grid_levels,
            );

            // Unstuck check: if WE is at high water-mark and an EMA-bound close
            // would only cost a small fraction of balance, fire it.
            if input.position_long.size > 0.0 {
                let we = calc_wallet_exposure(
                    input.exchange_params.c_mult,
                    input.state_params.balance,
                    input.position_long.size,
                    input.position_long.price,
                );
                let loss_allowance = input.state_params.balance
                    * input.bot_params.unstuck_loss_allowance_pct;
                if let Some(u) = calc_unstuck_close_long(
                    input.exchange_params,
                    input.bot_params,
                    input.position_long,
                    we,
                    input.state_params.balance,
                    input.state_params.ema_bands.lower,
                    loss_allowance,
                ) {
                    out.unstuck_long = Some(Order {
                        qty: u.close_qty,
                        price: u.close_price,
                        order_type: OrderType::CloseUnstuckLong,
                    });
                }
            }
        }
    }

    // ---------- SHORT side ----------
    match input.mode_short {
        TradingMode::Panic => {
            if input.position_short.size < 0.0 {
                out.closes_short.push(Order {
                    qty: input.position_short.size.abs(),
                    price: input.state_params.order_book.ask,
                    order_type: OrderType::ClosePanicShort,
                });
            }
        }
        TradingMode::GracefulStop | TradingMode::TpOnly => {
            if input.position_short.size < 0.0 {
                out.closes_short = calc_closes_short(
                    input.exchange_params,
                    input.state_params,
                    input.bot_params,
                    input.position_short,
                    input.trailing_short,
                );
            }
        }
        TradingMode::Normal => {
            if input.position_short.size < 0.0 {
                out.closes_short = calc_closes_short(
                    input.exchange_params,
                    input.state_params,
                    input.bot_params,
                    input.position_short,
                    input.trailing_short,
                );
            }
            out.entries_short = calc_entries_short(
                input.exchange_params,
                input.state_params,
                input.bot_params,
                input.position_short,
                input.trailing_short,
                input.wel_cap_short,
                input.max_grid_levels,
            );

            if input.position_short.size < 0.0 {
                let we = calc_wallet_exposure(
                    input.exchange_params.c_mult,
                    input.state_params.balance,
                    input.position_short.size,
                    input.position_short.price,
                );
                let loss_allowance = input.state_params.balance
                    * input.bot_params.unstuck_loss_allowance_pct;
                if let Some(u) = calc_unstuck_close_short(
                    input.exchange_params,
                    input.bot_params,
                    input.position_short,
                    we,
                    input.state_params.balance,
                    input.state_params.ema_bands.upper,
                    loss_allowance,
                ) {
                    out.unstuck_short = Some(Order {
                        qty: u.close_qty,
                        price: u.close_price,
                        order_type: OrderType::CloseUnstuckShort,
                    });
                }
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EMABands, OrderBook};

    fn default_bp() -> BotParams {
        BotParams {
            entry_initial_ema_dist: 0.005,
            entry_initial_qty_pct: 0.02,
            entry_grid_spacing_pct: 0.02,
            entry_grid_spacing_volatility_weight: 0.0,
            entry_grid_spacing_we_weight: 1.0,
            entry_grid_double_down_factor: 1.5,
            entry_trailing_threshold_pct: 0.01,
            entry_trailing_retracement_pct: 0.005,
            entry_trailing_grid_ratio: 0.0,
            close_grid_markup_start: 0.005,
            close_grid_markup_end: 0.015,
            close_grid_qty_pct: 0.5,
            close_trailing_threshold_pct: 0.01,
            close_trailing_retracement_pct: 0.004,
            close_trailing_grid_ratio: 0.0,
            close_trailing_qty_pct: 0.5,
            wallet_exposure_limit: 1.0,
            n_positions: 5,
            total_wallet_exposure_limit: 3.0,
            ema_span_0: 385.0,
            ema_span_1: 620.0,
            unstuck_threshold: 0.8,
            unstuck_close_pct: 0.05,
            unstuck_ema_dist: 0.01,
            unstuck_loss_allowance_pct: 0.001,
            risk_we_excess_allowance_pct: 0.0,
            risk_wel_enforcer_threshold: 0.98,
            risk_twel_enforcer_threshold: 0.95,
        }
    }

    #[test]
    fn normal_mode_produces_entries_and_closes_when_position_open() {
        let bp = default_bp();
        let ep = ExchangeParams {
            qty_step: 0.001, price_step: 0.01, min_qty: 0.001, min_cost: 5.0,
            c_mult: 1.0, maker_fee: 0.0002, taker_fee: 0.0005,
        };
        let sp = StateParams {
            balance: 10000.0,
            order_book: OrderBook { bid: 49900.0, ask: 50000.0 },
            ema_bands: EMABands { upper: 50500.0, lower: 49500.0 },
            entry_volatility_logrange_ema_1h: 0.0,
        };
        let pos_long = Position { size: 0.01, price: 50000.0 };
        let pos_short = Position::default();

        let input = OrchestratorInput {
            bot_params: &bp, exchange_params: &ep, state_params: &sp,
            position_long: &pos_long, position_short: &pos_short,
            trailing_long: &TrailingPriceBundle::default(),
            trailing_short: &TrailingPriceBundle::default(),
            mode_long: TradingMode::Normal, mode_short: TradingMode::Normal,
            wel_cap_long: 1.0, wel_cap_short: 1.0, max_grid_levels: 5,
        };

        let out = orchestrate(&input);
        assert!(!out.entries_long.is_empty(), "should propose long entries");
        assert!(!out.closes_long.is_empty(), "should propose long closes");
    }

    #[test]
    fn panic_mode_market_closes_full_position() {
        let bp = default_bp();
        let ep = ExchangeParams {
            qty_step: 0.001, price_step: 0.01, min_qty: 0.001, min_cost: 5.0,
            c_mult: 1.0, maker_fee: 0.0002, taker_fee: 0.0005,
        };
        let sp = StateParams {
            balance: 10000.0,
            order_book: OrderBook { bid: 49900.0, ask: 50000.0 },
            ema_bands: EMABands { upper: 50500.0, lower: 49500.0 },
            entry_volatility_logrange_ema_1h: 0.0,
        };
        let pos_long = Position { size: 0.1, price: 50000.0 };
        let pos_short = Position::default();

        let input = OrchestratorInput {
            bot_params: &bp, exchange_params: &ep, state_params: &sp,
            position_long: &pos_long, position_short: &pos_short,
            trailing_long: &TrailingPriceBundle::default(),
            trailing_short: &TrailingPriceBundle::default(),
            mode_long: TradingMode::Panic, mode_short: TradingMode::Normal,
            wel_cap_long: 1.0, wel_cap_short: 1.0, max_grid_levels: 5,
        };

        let out = orchestrate(&input);
        assert_eq!(out.closes_long.len(), 1);
        assert!(matches!(out.closes_long[0].order_type, OrderType::ClosePanicLong));
        assert_eq!(out.closes_long[0].qty, pos_long.size);
    }

    #[test]
    fn tp_only_mode_no_new_entries() {
        let bp = default_bp();
        let ep = ExchangeParams {
            qty_step: 0.001, price_step: 0.01, min_qty: 0.001, min_cost: 5.0,
            c_mult: 1.0, maker_fee: 0.0002, taker_fee: 0.0005,
        };
        let sp = StateParams {
            balance: 10000.0,
            order_book: OrderBook { bid: 49900.0, ask: 50000.0 },
            ema_bands: EMABands { upper: 50500.0, lower: 49500.0 },
            entry_volatility_logrange_ema_1h: 0.0,
        };
        let pos_long = Position { size: 0.01, price: 50000.0 };

        let input = OrchestratorInput {
            bot_params: &bp, exchange_params: &ep, state_params: &sp,
            position_long: &pos_long, position_short: &Position::default(),
            trailing_long: &TrailingPriceBundle::default(),
            trailing_short: &TrailingPriceBundle::default(),
            mode_long: TradingMode::TpOnly, mode_short: TradingMode::Normal,
            wel_cap_long: 1.0, wel_cap_short: 1.0, max_grid_levels: 5,
        };

        let out = orchestrate(&input);
        assert!(out.entries_long.is_empty());
        assert!(!out.closes_long.is_empty());
    }
}
