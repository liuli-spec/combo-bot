use crate::entries::{calc_min_entry_qty, wallet_exposure_limit_with_allowance};
use crate::types::{
    BotParams, ExchangeParams, Order, OrderType, Position, StateParams, TrailingPriceBundle,
};
use crate::utils::{cost_to_qty, round_, round_dn, round_up};

pub fn calc_close_qty(
    ep: &ExchangeParams,
    bp: &BotParams,
    position: &Position,
    close_qty_pct: f64,
    balance: f64,
    close_price: f64,
) -> f64 {
    let full_psize = cost_to_qty(
        balance * wallet_exposure_limit_with_allowance(bp),
        position.price,
        ep.c_mult,
    );
    let psize_abs = position.size.abs();
    let leftover = (psize_abs - full_psize).max(0.0);
    let min_qty = calc_min_entry_qty(close_price, ep);

    let target = round_up(full_psize * close_qty_pct + leftover, ep.qty_step).max(min_qty);
    let qty = round_(psize_abs, ep.qty_step).min(target);

    // Close the dust: if remaining would be below min_qty, close everything.
    if qty > 0.0 && qty < psize_abs && (psize_abs - qty) < min_qty {
        psize_abs
    } else {
        qty
    }
}

pub fn calc_grid_close_long(
    ep: &ExchangeParams,
    sp: &StateParams,
    bp: &BotParams,
    position: &Position,
) -> Option<Order> {
    if position.size <= 0.0 || position.price <= 0.0 {
        return None;
    }

    let close_price = round_up(position.price * (1.0 + bp.close_grid_markup_start), ep.price_step)
        .max(round_up(sp.order_book.ask, ep.price_step));

    let qty = calc_close_qty(ep, bp, position, bp.close_grid_qty_pct, sp.balance, close_price);
    if qty <= 0.0 {
        return None;
    }

    Some(Order {
        qty,
        price: close_price,
        order_type: OrderType::CloseGridLong,
    })
}

pub fn calc_grid_close_short(
    ep: &ExchangeParams,
    sp: &StateParams,
    bp: &BotParams,
    position: &Position,
) -> Option<Order> {
    if position.size >= 0.0 || position.price <= 0.0 {
        return None;
    }

    let close_price = round_dn(position.price * (1.0 - bp.close_grid_markup_start), ep.price_step)
        .min(round_dn(sp.order_book.bid, ep.price_step));
    if close_price <= 0.0 {
        return None;
    }

    let qty = calc_close_qty(ep, bp, position, bp.close_grid_qty_pct, sp.balance, close_price);
    if qty <= 0.0 {
        return None;
    }

    Some(Order {
        qty,
        price: close_price,
        order_type: OrderType::CloseGridShort,
    })
}

pub fn calc_trailing_close_long(
    ep: &ExchangeParams,
    sp: &StateParams,
    bp: &BotParams,
    position: &Position,
    trailing: &TrailingPriceBundle,
) -> Option<Order> {
    if position.size <= 0.0 || position.price <= 0.0 {
        return None;
    }
    if trailing.max_since_open <= f64::MIN {
        return None;
    }

    // Activation: price has run up at least threshold_pct above entry.
    let activation_price = position.price * (1.0 + bp.close_trailing_threshold_pct);
    if trailing.max_since_open < activation_price {
        return None;
    }

    // Trigger: price has retraced from max by at least retracement_pct.
    let retracement = if trailing.max_since_open > 0.0 {
        1.0 - trailing.min_since_max / trailing.max_since_open
    } else {
        0.0
    };
    if retracement < bp.close_trailing_retracement_pct {
        return None;
    }

    let close_price = round_up(sp.order_book.ask, ep.price_step);
    let qty = calc_close_qty(
        ep,
        bp,
        position,
        bp.close_trailing_qty_pct,
        sp.balance,
        close_price,
    );
    if qty <= 0.0 {
        return None;
    }

    Some(Order {
        qty,
        price: close_price,
        order_type: OrderType::CloseTrailingLong,
    })
}

pub fn calc_trailing_close_short(
    ep: &ExchangeParams,
    sp: &StateParams,
    bp: &BotParams,
    position: &Position,
    trailing: &TrailingPriceBundle,
) -> Option<Order> {
    if position.size >= 0.0 || position.price <= 0.0 {
        return None;
    }
    if trailing.min_since_open >= f64::MAX {
        return None;
    }

    let activation_price = position.price * (1.0 - bp.close_trailing_threshold_pct);
    if trailing.min_since_open > activation_price {
        return None;
    }

    let bounce = if trailing.min_since_open > 0.0 {
        trailing.max_since_min / trailing.min_since_open - 1.0
    } else {
        0.0
    };
    if bounce < bp.close_trailing_retracement_pct {
        return None;
    }

    let close_price = round_dn(sp.order_book.bid, ep.price_step);
    if close_price <= 0.0 {
        return None;
    }
    let qty = calc_close_qty(
        ep,
        bp,
        position,
        bp.close_trailing_qty_pct,
        sp.balance,
        close_price,
    );
    if qty <= 0.0 {
        return None;
    }

    Some(Order {
        qty,
        price: close_price,
        order_type: OrderType::CloseTrailingShort,
    })
}

pub fn calc_closes_long(
    ep: &ExchangeParams,
    sp: &StateParams,
    bp: &BotParams,
    position: &Position,
    trailing: &TrailingPriceBundle,
) -> Vec<Order> {
    let mut orders = Vec::new();
    if position.size <= 0.0 {
        return orders;
    }

    let psize_abs = position.size.abs();
    let n_levels = (1.0 / bp.close_grid_qty_pct.max(0.01)).ceil() as usize;
    let mut remaining = psize_abs;
    let markup_step = (bp.close_grid_markup_end - bp.close_grid_markup_start).max(0.0)
        / (n_levels.max(1) as f64);

    for i in 0..n_levels {
        let markup = bp.close_grid_markup_start + markup_step * i as f64;
        let price = round_up(position.price * (1.0 + markup), ep.price_step);
        if price <= 0.0 {
            break;
        }

        let level_qty = if i == n_levels - 1 {
            remaining
        } else {
            round_up(psize_abs * bp.close_grid_qty_pct, ep.qty_step).min(remaining)
        };

        if level_qty < calc_min_entry_qty(price, ep) {
            continue;
        }

        orders.push(Order {
            qty: level_qty,
            price,
            order_type: OrderType::CloseGridLong,
        });

        remaining -= level_qty;
        if remaining < calc_min_entry_qty(price, ep) {
            break;
        }
    }

    if let Some(trail) = calc_trailing_close_long(ep, sp, bp, position, trailing) {
        orders.push(trail);
    }

    orders
}

pub fn calc_closes_short(
    ep: &ExchangeParams,
    sp: &StateParams,
    bp: &BotParams,
    position: &Position,
    trailing: &TrailingPriceBundle,
) -> Vec<Order> {
    let mut orders = Vec::new();
    if position.size >= 0.0 {
        return orders;
    }

    let psize_abs = position.size.abs();
    let n_levels = (1.0 / bp.close_grid_qty_pct.max(0.01)).ceil() as usize;
    let mut remaining = psize_abs;
    let markup_step = (bp.close_grid_markup_end - bp.close_grid_markup_start).max(0.0)
        / (n_levels.max(1) as f64);

    for i in 0..n_levels {
        let markup = bp.close_grid_markup_start + markup_step * i as f64;
        let price = round_dn(position.price * (1.0 - markup), ep.price_step);
        if price <= 0.0 {
            break;
        }

        let level_qty = if i == n_levels - 1 {
            remaining
        } else {
            round_up(psize_abs * bp.close_grid_qty_pct, ep.qty_step).min(remaining)
        };

        if level_qty < calc_min_entry_qty(price, ep) {
            continue;
        }

        orders.push(Order {
            qty: level_qty,
            price,
            order_type: OrderType::CloseGridShort,
        });

        remaining -= level_qty;
        if remaining < calc_min_entry_qty(price, ep) {
            break;
        }
    }

    if let Some(trail) = calc_trailing_close_short(ep, sp, bp, position, trailing) {
        orders.push(trail);
    }

    orders
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EMABands, OrderBook};

    fn ep() -> ExchangeParams {
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

    fn bp() -> BotParams {
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

    fn sp() -> StateParams {
        StateParams {
            balance: 10000.0,
            order_book: OrderBook { bid: 50000.0, ask: 50100.0 },
            ema_bands: EMABands { upper: 50500.0, lower: 49500.0 },
            entry_volatility_logrange_ema_1h: 0.0,
        }
    }

    #[test]
    fn close_long_above_entry() {
        let pos = Position { size: 0.1, price: 50000.0 };
        let order = calc_grid_close_long(&ep(), &sp(), &bp(), &pos).unwrap();
        assert!(order.price > pos.price);
    }

    #[test]
    fn close_short_below_entry() {
        let pos = Position { size: -0.1, price: 50000.0 };
        let order = calc_grid_close_short(&ep(), &sp(), &bp(), &pos).unwrap();
        assert!(order.price < pos.price);
    }

    #[test]
    fn close_levels_distributed_across_markup_range() {
        let pos = Position { size: 0.1, price: 50000.0 };
        let orders = calc_closes_long(&ep(), &sp(), &bp(), &pos, &TrailingPriceBundle::default());
        assert!(orders.len() >= 2);
        for i in 1..orders.len() {
            if matches!(orders[i].order_type, OrderType::CloseGridLong)
                && matches!(orders[i - 1].order_type, OrderType::CloseGridLong)
            {
                assert!(orders[i].price >= orders[i - 1].price);
            }
        }
    }

    #[test]
    fn close_qtys_sum_to_position() {
        let pos = Position { size: 0.1, price: 50000.0 };
        let orders = calc_closes_long(&ep(), &sp(), &bp(), &pos, &TrailingPriceBundle::default());
        let grid_total: f64 = orders
            .iter()
            .filter(|o| matches!(o.order_type, OrderType::CloseGridLong))
            .map(|o| o.qty)
            .sum();
        assert!((grid_total - pos.size).abs() < 0.005);
    }

    #[test]
    fn trailing_close_activates_after_run_up() {
        let pos = Position { size: 0.1, price: 50000.0 };
        let trailing = TrailingPriceBundle {
            min_since_open: 49000.0,
            max_since_min: 51500.0,
            max_since_open: 51500.0,
            min_since_max: 51000.0, // retraced from 51500 → 51000 = 0.98% drop
        };
        let mut b = bp();
        b.close_trailing_threshold_pct = 0.01;
        b.close_trailing_retracement_pct = 0.005;
        let order = calc_trailing_close_long(&ep(), &sp(), &b, &pos, &trailing);
        assert!(order.is_some());
        assert!(matches!(order.unwrap().order_type, OrderType::CloseTrailingLong));
    }

    #[test]
    fn no_close_for_empty_position() {
        let pos = Position::default();
        assert!(calc_grid_close_long(&ep(), &sp(), &bp(), &pos).is_none());
        assert!(calc_grid_close_short(&ep(), &sp(), &bp(), &pos).is_none());
    }
}
