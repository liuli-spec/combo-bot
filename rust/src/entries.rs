use crate::types::{
    BotParams, ExchangeParams, Order, OrderType, Position, StateParams, TrailingPriceBundle,
};
use crate::utils::{
    calc_ema_price_ask, calc_ema_price_bid, calc_new_psize_pprice, calc_wallet_exposure,
    cost_to_qty, round_, round_dn, round_up,
};

#[inline]
pub fn wallet_exposure_limit_with_allowance(bp: &BotParams) -> f64 {
    let base = bp.wallet_exposure_limit;
    if base <= 0.0 {
        return base;
    }
    base * (1.0 + bp.risk_we_excess_allowance_pct.max(0.0))
}

#[inline]
pub fn calc_min_entry_qty(entry_price: f64, ep: &ExchangeParams) -> f64 {
    ep.min_qty
        .max(round_up(cost_to_qty(ep.min_cost, entry_price, ep.c_mult), ep.qty_step))
}

#[inline]
pub fn calc_initial_entry_qty(
    ep: &ExchangeParams,
    bp: &BotParams,
    balance: f64,
    entry_price: f64,
) -> f64 {
    let wel = wallet_exposure_limit_with_allowance(bp);
    let target_qty = cost_to_qty(balance * wel * bp.entry_initial_qty_pct, entry_price, ep.c_mult);
    calc_min_entry_qty(entry_price, ep).max(round_(target_qty, ep.qty_step))
}

pub fn calc_reentry_qty(
    entry_price: f64,
    balance: f64,
    position_size: f64,
    double_down_factor: f64,
    ep: &ExchangeParams,
    bp: &BotParams,
    wel_cap: f64,
) -> f64 {
    let effective_wel = wel_cap.min(wallet_exposure_limit_with_allowance(bp));
    let by_dd = position_size.abs() * double_down_factor;
    let by_initial = cost_to_qty(balance, entry_price, ep.c_mult)
        * effective_wel
        * bp.entry_initial_qty_pct;
    let raw = by_dd.max(by_initial);
    calc_min_entry_qty(entry_price, ep).max(round_(raw, ep.qty_step))
}

fn calc_reentry_price_bid(
    position_price: f64,
    wallet_exposure: f64,
    order_book_bid: f64,
    ep: &ExchangeParams,
    bp: &BotParams,
    volatility_1h: f64,
    wel_cap: f64,
) -> f64 {
    let effective_wel = wel_cap.min(wallet_exposure_limit_with_allowance(bp));
    let we_multiplier = if effective_wel > 0.0 {
        (wallet_exposure / effective_wel) * bp.entry_grid_spacing_we_weight
    } else {
        0.0
    };
    let log_multiplier = volatility_1h * bp.entry_grid_spacing_volatility_weight;
    let spacing_multiplier = (1.0 + we_multiplier + log_multiplier).max(0.0);

    let reentry_price = round_dn(
        position_price * (1.0 - bp.entry_grid_spacing_pct * spacing_multiplier),
        ep.price_step,
    )
    .min(order_book_bid);

    if reentry_price <= ep.price_step {
        0.0
    } else {
        reentry_price
    }
}

fn calc_reentry_price_ask(
    position_price: f64,
    wallet_exposure: f64,
    order_book_ask: f64,
    ep: &ExchangeParams,
    bp: &BotParams,
    volatility_1h: f64,
    wel_cap: f64,
) -> f64 {
    let effective_wel = wel_cap.min(wallet_exposure_limit_with_allowance(bp));
    let we_multiplier = if effective_wel > 0.0 {
        (wallet_exposure / effective_wel) * bp.entry_grid_spacing_we_weight
    } else {
        0.0
    };
    let log_multiplier = volatility_1h * bp.entry_grid_spacing_volatility_weight;
    let spacing_multiplier = (1.0 + we_multiplier + log_multiplier).max(0.0);

    let reentry_price = round_up(
        position_price * (1.0 + bp.entry_grid_spacing_pct * spacing_multiplier),
        ep.price_step,
    )
    .max(order_book_ask);

    if reentry_price <= ep.price_step {
        0.0
    } else {
        reentry_price
    }
}

pub fn calc_cropped_reentry_qty(
    ep: &ExchangeParams,
    bp: &BotParams,
    position: &Position,
    wallet_exposure: f64,
    balance: f64,
    entry_qty: f64,
    entry_price: f64,
    wel_cap: f64,
) -> (f64, f64) {
    let effective_wel = wel_cap.min(wallet_exposure_limit_with_allowance(bp));
    let psize_abs = position.size.abs();
    let qty_abs = entry_qty.abs();

    let cost_after = (psize_abs + qty_abs) * entry_price * ep.c_mult;
    let we_if_filled = if balance > 0.0 { cost_after / balance } else { 0.0 };

    let min_qty = calc_min_entry_qty(entry_price, ep);

    if we_if_filled > effective_wel * 1.01 {
        // Linear interpolation: how much qty fits exactly at the limit?
        let we_now = wallet_exposure;
        if we_if_filled - we_now <= f64::EPSILON {
            return (we_if_filled, min_qty);
        }
        let scale = (effective_wel - we_now) / (we_if_filled - we_now);
        let cropped = (scale * qty_abs).max(0.0);
        let final_qty = round_(cropped, ep.qty_step).max(min_qty);
        (we_if_filled, final_qty)
    } else {
        (we_if_filled, qty_abs.max(min_qty))
    }
}

pub fn calc_grid_entry_long(
    ep: &ExchangeParams,
    sp: &StateParams,
    bp: &BotParams,
    position: &Position,
    wel_cap: f64,
) -> Option<Order> {
    if wallet_exposure_limit_with_allowance(bp) == 0.0 || sp.balance <= 0.0 {
        return None;
    }

    let initial_price = calc_ema_price_bid(
        ep.price_step,
        sp.order_book.bid,
        sp.ema_bands.lower,
        bp.entry_initial_ema_dist,
    );
    if initial_price <= ep.price_step {
        return None;
    }

    let initial_qty = calc_initial_entry_qty(ep, bp, sp.balance, initial_price);

    if position.size == 0.0 {
        return Some(Order {
            qty: initial_qty,
            price: initial_price,
            order_type: OrderType::EntryInitialNormalLong,
        });
    }

    if position.size < initial_qty * 0.8 {
        let needed = round_dn(initial_qty - position.size, ep.qty_step)
            .max(calc_min_entry_qty(initial_price, ep));
        return Some(Order {
            qty: needed,
            price: initial_price,
            order_type: OrderType::EntryInitialNormalLong,
        });
    }

    let wallet_exposure = calc_wallet_exposure(ep.c_mult, sp.balance, position.size, position.price);
    let effective_wel = wel_cap.min(wallet_exposure_limit_with_allowance(bp));
    if wallet_exposure >= effective_wel * 0.999 {
        return None;
    }

    let reentry_price = calc_reentry_price_bid(
        position.price,
        wallet_exposure,
        sp.order_book.bid,
        ep,
        bp,
        sp.entry_volatility_logrange_ema_1h,
        effective_wel,
    );
    if reentry_price <= 0.0 {
        return None;
    }

    let reentry_qty = calc_reentry_qty(
        reentry_price,
        sp.balance,
        position.size,
        bp.entry_grid_double_down_factor,
        ep,
        bp,
        effective_wel,
    )
    .max(initial_qty);

    let (_we_if, cropped) = calc_cropped_reentry_qty(
        ep,
        bp,
        position,
        wallet_exposure,
        sp.balance,
        reentry_qty,
        reentry_price,
        effective_wel,
    );

    if cropped < calc_min_entry_qty(reentry_price, ep) {
        return None;
    }

    Some(Order {
        qty: cropped,
        price: reentry_price,
        order_type: OrderType::EntryGridNormalLong,
    })
}

pub fn calc_grid_entry_short(
    ep: &ExchangeParams,
    sp: &StateParams,
    bp: &BotParams,
    position: &Position,
    wel_cap: f64,
) -> Option<Order> {
    if wallet_exposure_limit_with_allowance(bp) == 0.0 || sp.balance <= 0.0 {
        return None;
    }

    let initial_price = calc_ema_price_ask(
        ep.price_step,
        sp.order_book.ask,
        sp.ema_bands.upper,
        bp.entry_initial_ema_dist,
    );
    if initial_price <= ep.price_step {
        return None;
    }

    let initial_qty = calc_initial_entry_qty(ep, bp, sp.balance, initial_price);

    if position.size == 0.0 {
        return Some(Order {
            qty: initial_qty,
            price: initial_price,
            order_type: OrderType::EntryInitialNormalShort,
        });
    }

    let psize_abs = position.size.abs();
    if psize_abs < initial_qty * 0.8 {
        let needed = round_dn(initial_qty - psize_abs, ep.qty_step)
            .max(calc_min_entry_qty(initial_price, ep));
        return Some(Order {
            qty: needed,
            price: initial_price,
            order_type: OrderType::EntryInitialNormalShort,
        });
    }

    let wallet_exposure = calc_wallet_exposure(ep.c_mult, sp.balance, position.size, position.price);
    let effective_wel = wel_cap.min(wallet_exposure_limit_with_allowance(bp));
    if wallet_exposure >= effective_wel * 0.999 {
        return None;
    }

    let reentry_price = calc_reentry_price_ask(
        position.price,
        wallet_exposure,
        sp.order_book.ask,
        ep,
        bp,
        sp.entry_volatility_logrange_ema_1h,
        effective_wel,
    );
    if reentry_price <= 0.0 {
        return None;
    }

    let reentry_qty = calc_reentry_qty(
        reentry_price,
        sp.balance,
        position.size,
        bp.entry_grid_double_down_factor,
        ep,
        bp,
        effective_wel,
    )
    .max(initial_qty);

    let (_we_if, cropped) = calc_cropped_reentry_qty(
        ep,
        bp,
        position,
        wallet_exposure,
        sp.balance,
        reentry_qty,
        reentry_price,
        effective_wel,
    );

    if cropped < calc_min_entry_qty(reentry_price, ep) {
        return None;
    }

    Some(Order {
        qty: cropped,
        price: reentry_price,
        order_type: OrderType::EntryGridNormalShort,
    })
}

pub fn calc_trailing_entry_long(
    ep: &ExchangeParams,
    sp: &StateParams,
    bp: &BotParams,
    position: &Position,
    trailing: &TrailingPriceBundle,
    wel_cap: f64,
) -> Option<Order> {
    if position.size == 0.0 || position.price <= 0.0 {
        return None;
    }
    if trailing.min_since_open >= f64::MAX || trailing.max_since_min <= f64::MIN {
        return None;
    }

    let wallet_exposure = calc_wallet_exposure(ep.c_mult, sp.balance, position.size, position.price);
    let effective_wel = wel_cap.min(wallet_exposure_limit_with_allowance(bp));
    if wallet_exposure >= effective_wel * 0.999 {
        return None;
    }

    // Threshold: how far below position_price the min must have dipped to activate trailing.
    let threshold_price = position.price * (1.0 - bp.entry_trailing_threshold_pct);
    if trailing.min_since_open > threshold_price {
        return None;
    }

    // Retracement: price has bounced from min by at least retracement_pct.
    let bounce_ratio = if trailing.min_since_open > 0.0 {
        trailing.max_since_min / trailing.min_since_open - 1.0
    } else {
        0.0
    };
    if bounce_ratio < bp.entry_trailing_retracement_pct {
        return None;
    }

    let entry_price = round_dn(sp.order_book.bid, ep.price_step);
    if entry_price <= 0.0 {
        return None;
    }

    let entry_qty = calc_reentry_qty(
        entry_price,
        sp.balance,
        position.size,
        bp.entry_grid_double_down_factor,
        ep,
        bp,
        effective_wel,
    );
    let (_we_if, cropped) = calc_cropped_reentry_qty(
        ep,
        bp,
        position,
        wallet_exposure,
        sp.balance,
        entry_qty,
        entry_price,
        effective_wel,
    );
    if cropped < calc_min_entry_qty(entry_price, ep) {
        return None;
    }

    Some(Order {
        qty: cropped,
        price: entry_price,
        order_type: OrderType::EntryTrailingNormalLong,
    })
}

pub fn calc_trailing_entry_short(
    ep: &ExchangeParams,
    sp: &StateParams,
    bp: &BotParams,
    position: &Position,
    trailing: &TrailingPriceBundle,
    wel_cap: f64,
) -> Option<Order> {
    if position.size == 0.0 || position.price <= 0.0 {
        return None;
    }
    if trailing.max_since_open <= f64::MIN || trailing.min_since_max >= f64::MAX {
        return None;
    }

    let wallet_exposure = calc_wallet_exposure(ep.c_mult, sp.balance, position.size, position.price);
    let effective_wel = wel_cap.min(wallet_exposure_limit_with_allowance(bp));
    if wallet_exposure >= effective_wel * 0.999 {
        return None;
    }

    let threshold_price = position.price * (1.0 + bp.entry_trailing_threshold_pct);
    if trailing.max_since_open < threshold_price {
        return None;
    }

    let drop_ratio = if trailing.max_since_open > 0.0 {
        1.0 - trailing.min_since_max / trailing.max_since_open
    } else {
        0.0
    };
    if drop_ratio < bp.entry_trailing_retracement_pct {
        return None;
    }

    let entry_price = round_up(sp.order_book.ask, ep.price_step);
    if entry_price <= 0.0 {
        return None;
    }

    let entry_qty = calc_reentry_qty(
        entry_price,
        sp.balance,
        position.size,
        bp.entry_grid_double_down_factor,
        ep,
        bp,
        effective_wel,
    );
    let (_we_if, cropped) = calc_cropped_reentry_qty(
        ep,
        bp,
        position,
        wallet_exposure,
        sp.balance,
        entry_qty,
        entry_price,
        effective_wel,
    );
    if cropped < calc_min_entry_qty(entry_price, ep) {
        return None;
    }

    Some(Order {
        qty: cropped,
        price: entry_price,
        order_type: OrderType::EntryTrailingNormalShort,
    })
}

pub fn calc_next_entry_long(
    ep: &ExchangeParams,
    sp: &StateParams,
    bp: &BotParams,
    position: &Position,
    trailing: &TrailingPriceBundle,
    wel_cap: f64,
) -> Option<Order> {
    let grid = calc_grid_entry_long(ep, sp, bp, position, wel_cap);
    let trail = calc_trailing_entry_long(ep, sp, bp, position, trailing, wel_cap);

    let ratio = bp.entry_trailing_grid_ratio;
    match (grid, trail) {
        (None, None) => None,
        (Some(g), None) => Some(g),
        (None, Some(t)) => Some(t),
        (Some(g), Some(t)) => {
            // ratio in [-1.0, 1.0]: -1 = pure grid, 1 = pure trailing
            if ratio >= 0.99 {
                Some(t)
            } else if ratio <= -0.99 {
                Some(g)
            } else if ratio >= 0.0 {
                // Prefer trailing when its retracement has triggered.
                Some(t)
            } else {
                Some(g)
            }
        }
    }
}

pub fn calc_next_entry_short(
    ep: &ExchangeParams,
    sp: &StateParams,
    bp: &BotParams,
    position: &Position,
    trailing: &TrailingPriceBundle,
    wel_cap: f64,
) -> Option<Order> {
    let grid = calc_grid_entry_short(ep, sp, bp, position, wel_cap);
    let trail = calc_trailing_entry_short(ep, sp, bp, position, trailing, wel_cap);

    let ratio = bp.entry_trailing_grid_ratio;
    match (grid, trail) {
        (None, None) => None,
        (Some(g), None) => Some(g),
        (None, Some(t)) => Some(t),
        (Some(g), Some(t)) => {
            if ratio >= 0.99 {
                Some(t)
            } else if ratio <= -0.99 {
                Some(g)
            } else if ratio >= 0.0 {
                Some(t)
            } else {
                Some(g)
            }
        }
    }
}

pub fn calc_entries_long(
    ep: &ExchangeParams,
    sp: &StateParams,
    bp: &BotParams,
    position: &Position,
    trailing: &TrailingPriceBundle,
    wel_cap: f64,
    max_levels: usize,
) -> Vec<Order> {
    let mut orders = Vec::new();
    let mut pos = position.clone();

    for _ in 0..max_levels {
        let entry = calc_next_entry_long(ep, sp, bp, &pos, trailing, wel_cap);
        match entry {
            Some(o) if o.qty > 0.0 => {
                let (new_size, new_price) =
                    calc_new_psize_pprice(pos.size, pos.price, o.qty, o.price);
                pos.size = new_size;
                pos.price = new_price;
                orders.push(o);
            }
            _ => break,
        }
    }

    orders
}

pub fn calc_entries_short(
    ep: &ExchangeParams,
    sp: &StateParams,
    bp: &BotParams,
    position: &Position,
    trailing: &TrailingPriceBundle,
    wel_cap: f64,
    max_levels: usize,
) -> Vec<Order> {
    let mut orders = Vec::new();
    let mut pos = position.clone();

    for _ in 0..max_levels {
        let entry = calc_next_entry_short(ep, sp, bp, &pos, trailing, wel_cap);
        match entry {
            Some(o) if o.qty > 0.0 => {
                let (new_size, new_price) =
                    calc_new_psize_pprice(pos.size, pos.price, -o.qty, o.price);
                pos.size = new_size;
                pos.price = new_price;
                orders.push(o);
            }
            _ => break,
        }
    }

    orders
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

    fn default_sp() -> StateParams {
        StateParams {
            balance: 10000.0,
            order_book: OrderBook { bid: 49900.0, ask: 50000.0 },
            ema_bands: EMABands { upper: 50500.0, lower: 49500.0 },
            entry_volatility_logrange_ema_1h: 0.0,
        }
    }

    #[test]
    fn initial_long_entry_when_no_position() {
        let order = calc_grid_entry_long(
            &default_ep(), &default_sp(), &default_bp(), &Position::default(), 1.0,
        );
        let o = order.unwrap();
        assert!(matches!(o.order_type, OrderType::EntryInitialNormalLong));
        assert!(o.qty > 0.0);
        assert!(o.price > 0.0);
        assert!(o.price <= 49900.0);
    }

    #[test]
    fn initial_short_entry_when_no_position() {
        let order = calc_grid_entry_short(
            &default_ep(), &default_sp(), &default_bp(), &Position::default(), 1.0,
        );
        let o = order.unwrap();
        assert!(matches!(o.order_type, OrderType::EntryInitialNormalShort));
        assert!(o.qty > 0.0);
        assert!(o.price >= 50000.0);
    }

    #[test]
    fn no_entry_when_wel_zero() {
        let mut bp = default_bp();
        bp.wallet_exposure_limit = 0.0;
        let order = calc_grid_entry_long(
            &default_ep(), &default_sp(), &bp, &Position::default(), 1.0,
        );
        assert!(order.is_none());
    }

    #[test]
    fn grid_reentry_lower_than_position_price() {
        let pos = Position { size: 0.01, price: 50000.0 };
        let order = calc_grid_entry_long(&default_ep(), &default_sp(), &default_bp(), &pos, 1.0).unwrap();
        assert!(
            matches!(order.order_type, OrderType::EntryGridNormalLong)
                || matches!(order.order_type, OrderType::EntryInitialNormalLong)
        );
        assert!(order.price < pos.price);
    }

    #[test]
    fn no_reentry_when_at_wel_limit() {
        let pos = Position { size: 0.2, price: 50000.0 }; // 10k notional at 10k balance = WE 1.0
        let order = calc_grid_entry_long(&default_ep(), &default_sp(), &default_bp(), &pos, 1.0);
        assert!(order.is_none());
    }

    #[test]
    fn double_down_increases_qty() {
        let pos = Position { size: 0.01, price: 50000.0 };
        let qty = calc_reentry_qty(49000.0, 10000.0, pos.size, 1.5, &default_ep(), &default_bp(), 1.0);
        assert!(qty > pos.size);
    }

    #[test]
    fn multiple_entries_accumulate() {
        let orders = calc_entries_long(
            &default_ep(), &default_sp(), &default_bp(),
            &Position::default(), &TrailingPriceBundle::default(), 1.0, 5,
        );
        assert!(!orders.is_empty());
        for i in 1..orders.len() {
            assert!(orders[i].price < orders[i - 1].price, "entries must descend");
        }
    }

    #[test]
    fn trailing_entry_requires_retracement() {
        let pos = Position { size: 0.01, price: 50000.0 };
        let trailing = TrailingPriceBundle::default();
        let order = calc_trailing_entry_long(
            &default_ep(), &default_sp(), &default_bp(), &pos, &trailing, 1.0,
        );
        assert!(order.is_none(), "no trailing without retracement data");
    }

    #[test]
    fn trailing_entry_triggers_on_bounce() {
        let mut bp = default_bp();
        bp.entry_trailing_threshold_pct = 0.005;
        bp.entry_trailing_retracement_pct = 0.002;
        let pos = Position { size: 0.01, price: 50000.0 };
        let trailing = TrailingPriceBundle {
            min_since_open: 49000.0,
            max_since_min: 49500.0,
            max_since_open: 50100.0,
            min_since_max: 49000.0,
        };
        let order = calc_trailing_entry_long(
            &default_ep(), &default_sp(), &bp, &pos, &trailing, 1.0,
        );
        assert!(order.is_some());
        assert!(matches!(order.unwrap().order_type, OrderType::EntryTrailingNormalLong));
    }
}
