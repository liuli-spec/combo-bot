/// Round `value` to the nearest multiple of `step`.
///
/// Returns 0.0 when `step` is non-positive or non-finite, or when `value` is
/// non-finite, to avoid propagating NaN / Inf into order quantities or prices.
#[inline]
pub fn round_(value: f64, step: f64) -> f64 {
    if !step.is_finite() || step <= 0.0 || !value.is_finite() {
        return 0.0;
    }
    let r = (value / step).round() * step;
    // Clamp the tiny floating-point residuals that `round` can leave behind.
    (r * 1e10).round() / 1e10
}

/// Round `value` **down** to the nearest multiple of `step`.
#[inline]
pub fn round_dn(value: f64, step: f64) -> f64 {
    if !step.is_finite() || step <= 0.0 || !value.is_finite() {
        return 0.0;
    }
    let r = (value / step).floor() * step;
    (r * 1e10).round() / 1e10
}

/// Round `value` **up** to the nearest multiple of `step`.
#[inline]
pub fn round_up(value: f64, step: f64) -> f64 {
    if !step.is_finite() || step <= 0.0 || !value.is_finite() {
        return 0.0;
    }
    let r = (value / step).ceil() * step;
    (r * 1e10).round() / 1e10
}

/// Convert a notional `cost` into a quantity given `price` and contract
/// multiplier `c_mult`.
///
/// `qty = cost / (price * c_mult)`
///
/// Returns 0.0 when either `price` or `c_mult` is zero / non-finite.
#[inline]
pub fn cost_to_qty(cost: f64, price: f64, c_mult: f64) -> f64 {
    let denom = price * c_mult;
    if !denom.is_finite() || denom == 0.0 {
        return 0.0;
    }
    cost / denom
}

/// Convert a `qty` (possibly signed) into notional cost.
///
/// `cost = |qty| * price * c_mult`
#[inline]
pub fn qty_to_cost(qty: f64, price: f64, c_mult: f64) -> f64 {
    qty.abs() * price * c_mult
}

/// Wallet exposure: the ratio of position notional to account balance.
///
/// `WE = |position_size| * position_price * c_mult / balance`
///
/// Returns 0.0 when `balance` is zero or any input is non-finite.
#[inline]
pub fn calc_wallet_exposure(
    c_mult: f64,
    balance: f64,
    position_size: f64,
    position_price: f64,
) -> f64 {
    if !balance.is_finite() || balance == 0.0 {
        return 0.0;
    }
    let cost = qty_to_cost(position_size, position_price, c_mult);
    if !cost.is_finite() {
        return 0.0;
    }
    cost / balance
}

/// Compute the new position size and volume-weighted average entry price after
/// adding `qty` at `price` to an existing position.
///
/// Handles the cases where the existing position is empty (size == 0) and where
/// the addition would flip the position sign (clamped to the new fill only).
#[inline]
pub fn calc_new_psize_pprice(
    position_size: f64,
    position_price: f64,
    qty: f64,
    price: f64,
) -> (f64, f64) {
    if qty == 0.0 {
        return (position_size, position_price);
    }
    let new_size = position_size + qty;

    // Position fully closed or flipped — reset to the new fill.
    if new_size == 0.0 {
        return (0.0, 0.0);
    }
    // If the sign changed the position flipped; treat as fresh entry.
    if (position_size > 0.0 && new_size < 0.0) || (position_size < 0.0 && new_size > 0.0) {
        return (qty, price);
    }

    // First fill into a flat position.
    if position_size == 0.0 || position_price <= 0.0 {
        return (qty, price);
    }

    // Volume-weighted average price.
    let abs_old = position_size.abs();
    let abs_new = qty.abs();
    let total = abs_old + abs_new;
    if total == 0.0 {
        return (new_size, price);
    }
    let new_price = (position_price * abs_old + price * abs_new) / total;

    (new_size, new_price)
}

/// Realised PnL for closing `qty` of a **long** position.
///
/// `pnl = qty * (close_price - entry_price) * c_mult`
///
/// `qty` should be negative (closing a long), but we use its absolute value
/// internally so the caller does not need to worry about sign.
#[inline]
pub fn calc_pnl_long(entry_price: f64, close_price: f64, qty: f64, c_mult: f64) -> f64 {
    qty.abs() * (close_price - entry_price) * c_mult
}

/// Realised PnL for closing `qty` of a **short** position.
///
/// `pnl = qty_abs * (entry_price - close_price) * c_mult`
#[inline]
pub fn calc_pnl_short(entry_price: f64, close_price: f64, qty: f64, c_mult: f64) -> f64 {
    qty.abs() * (entry_price - close_price) * c_mult
}

/// Compute the EMA-derived **bid** price for a long initial entry.
///
/// The price is the minimum of the current best bid and the EMA lower band
/// adjusted by `initial_ema_dist`, then rounded down to `price_step`.
#[inline]
pub fn calc_ema_price_bid(
    price_step: f64,
    order_book_bid: f64,
    ema_band_lower: f64,
    initial_ema_dist: f64,
) -> f64 {
    let ema_price = ema_band_lower * (1.0 - initial_ema_dist);
    let raw = order_book_bid.min(ema_price);
    round_dn(raw, price_step)
}

/// Compute the EMA-derived **ask** price for a short initial entry.
///
/// The price is the maximum of the current best ask and the EMA upper band
/// adjusted by `initial_ema_dist`, then rounded up to `price_step`.
#[inline]
pub fn calc_ema_price_ask(
    price_step: f64,
    order_book_ask: f64,
    ema_band_upper: f64,
    initial_ema_dist: f64,
) -> f64 {
    let ema_price = ema_band_upper * (1.0 + initial_ema_dist);
    let raw = order_book_ask.max(ema_price);
    round_up(raw, price_step)
}

/// Piece-wise linear interpolation.
///
/// Given sorted control points `xs` (ascending) with corresponding values
/// `ys`, return the interpolated value at `target`.
///
/// - If `target <= xs[0]` → `ys[0]` (clamp left).
/// - If `target >= xs[last]` → `ys[last]` (clamp right).
/// - If `xs` is empty → 0.0.
/// - `xs` and `ys` must have the same length; if they differ the shorter
///   length is used.
#[inline]
pub fn interpolate(target: f64, xs: &[f64], ys: &[f64]) -> f64 {
    let len = xs.len().min(ys.len());
    if len == 0 {
        return 0.0;
    }
    if len == 1 || target <= xs[0] {
        return ys[0];
    }
    if target >= xs[len - 1] {
        return ys[len - 1];
    }

    // Find the segment [xs[i], xs[i+1]] that brackets `target`.
    // Linear scan is fine for the small control-point arrays used in practice.
    for i in 0..len - 1 {
        if target <= xs[i + 1] {
            let dx = xs[i + 1] - xs[i];
            if dx == 0.0 {
                return ys[i];
            }
            let t = (target - xs[i]) / dx;
            return ys[i] + t * (ys[i + 1] - ys[i]);
        }
    }

    // Fallback (should be unreachable).
    ys[len - 1]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    // -- rounding -----------------------------------------------------------

    #[test]
    fn test_round_nearest() {
        assert!((round_(1.23456, 0.01) - 1.23).abs() < EPS);
        assert!((round_(1.235, 0.01) - 1.24).abs() < EPS);
        assert!((round_(100.0, 0.5) - 100.0).abs() < EPS);
        assert!((round_(100.3, 0.5) - 100.5).abs() < EPS);
    }

    #[test]
    fn test_round_dn() {
        assert!((round_dn(1.239, 0.01) - 1.23).abs() < EPS);
        assert!((round_dn(100.7, 0.5) - 100.5).abs() < EPS);
    }

    #[test]
    fn test_round_up() {
        assert!((round_up(1.231, 0.01) - 1.24).abs() < EPS);
        assert!((round_up(100.1, 0.5) - 100.5).abs() < EPS);
    }

    #[test]
    fn test_round_edge_cases() {
        assert_eq!(round_(f64::NAN, 0.01), 0.0);
        assert_eq!(round_(1.0, 0.0), 0.0);
        assert_eq!(round_(1.0, -0.01), 0.0);
        assert_eq!(round_(1.0, f64::INFINITY), 0.0);
        assert_eq!(round_dn(f64::NAN, 1.0), 0.0);
        assert_eq!(round_up(f64::NAN, 1.0), 0.0);
    }

    // -- cost / qty ---------------------------------------------------------

    #[test]
    fn test_cost_to_qty() {
        let qty = cost_to_qty(1000.0, 50000.0, 1.0);
        assert!((qty - 0.02).abs() < EPS);
    }

    #[test]
    fn test_cost_to_qty_zero_price() {
        assert_eq!(cost_to_qty(1000.0, 0.0, 1.0), 0.0);
    }

    #[test]
    fn test_qty_to_cost() {
        let cost = qty_to_cost(0.5, 30000.0, 1.0);
        assert!((cost - 15000.0).abs() < EPS);
    }

    #[test]
    fn test_qty_to_cost_negative() {
        let cost = qty_to_cost(-0.5, 30000.0, 1.0);
        assert!((cost - 15000.0).abs() < EPS);
    }

    // -- wallet exposure ----------------------------------------------------

    #[test]
    fn test_calc_wallet_exposure() {
        let we = calc_wallet_exposure(1.0, 10000.0, 0.5, 30000.0);
        assert!((we - 1.5).abs() < EPS);
    }

    #[test]
    fn test_calc_wallet_exposure_zero_balance() {
        assert_eq!(calc_wallet_exposure(1.0, 0.0, 1.0, 100.0), 0.0);
    }

    // -- new psize / pprice -------------------------------------------------

    #[test]
    fn test_calc_new_psize_pprice_fresh_entry() {
        let (size, price) = calc_new_psize_pprice(0.0, 0.0, 1.0, 50000.0);
        assert!((size - 1.0).abs() < EPS);
        assert!((price - 50000.0).abs() < EPS);
    }

    #[test]
    fn test_calc_new_psize_pprice_average_in() {
        let (size, price) = calc_new_psize_pprice(1.0, 50000.0, 1.0, 40000.0);
        assert!((size - 2.0).abs() < EPS);
        assert!((price - 45000.0).abs() < EPS);
    }

    #[test]
    fn test_calc_new_psize_pprice_full_close() {
        let (size, price) = calc_new_psize_pprice(1.0, 50000.0, -1.0, 55000.0);
        assert_eq!(size, 0.0);
        assert_eq!(price, 0.0);
    }

    #[test]
    fn test_calc_new_psize_pprice_flip() {
        let (size, price) = calc_new_psize_pprice(1.0, 50000.0, -3.0, 55000.0);
        assert!((size - (-3.0)).abs() < EPS);
        assert!((price - 55000.0).abs() < EPS);
    }

    #[test]
    fn test_calc_new_psize_pprice_zero_qty() {
        let (size, price) = calc_new_psize_pprice(1.0, 50000.0, 0.0, 60000.0);
        assert!((size - 1.0).abs() < EPS);
        assert!((price - 50000.0).abs() < EPS);
    }

    // -- PnL ----------------------------------------------------------------

    #[test]
    fn test_pnl_long_profit() {
        let pnl = calc_pnl_long(50000.0, 55000.0, 1.0, 1.0);
        assert!((pnl - 5000.0).abs() < EPS);
    }

    #[test]
    fn test_pnl_long_loss() {
        let pnl = calc_pnl_long(50000.0, 45000.0, 1.0, 1.0);
        assert!((pnl - (-5000.0)).abs() < EPS);
    }

    #[test]
    fn test_pnl_short_profit() {
        let pnl = calc_pnl_short(50000.0, 45000.0, 1.0, 1.0);
        assert!((pnl - 5000.0).abs() < EPS);
    }

    #[test]
    fn test_pnl_short_loss() {
        let pnl = calc_pnl_short(50000.0, 55000.0, 1.0, 1.0);
        assert!((pnl - (-5000.0)).abs() < EPS);
    }

    #[test]
    fn test_pnl_with_c_mult() {
        // qty=10, price diff=5000, c_mult=100 → 10 * 5000 * 100 = 5_000_000
        let pnl = calc_pnl_long(50000.0, 55000.0, 10.0, 100.0);
        assert!((pnl - 5_000_000.0).abs() < EPS);
    }

    // -- EMA prices ---------------------------------------------------------

    #[test]
    fn test_calc_ema_price_bid() {
        let price = calc_ema_price_bid(0.01, 100.0, 99.0, 0.01);
        // ema_price = 99.0 * (1 - 0.01) = 98.01
        // min(100.0, 98.01) = 98.01 → round_dn(98.01, 0.01) = 98.01
        assert!((price - 98.01).abs() < EPS);
    }

    #[test]
    fn test_calc_ema_price_bid_bid_lower() {
        let price = calc_ema_price_bid(0.01, 97.0, 99.0, 0.01);
        // ema_price = 98.01, min(97.0, 98.01) = 97.0
        assert!((price - 97.0).abs() < EPS);
    }

    #[test]
    fn test_calc_ema_price_ask() {
        let price = calc_ema_price_ask(0.01, 100.0, 101.0, 0.01);
        // ema_price = 101.0 * 1.01 = 102.01
        // max(100.0, 102.01) = 102.01 → round_up(102.01, 0.01) = 102.01
        assert!((price - 102.01).abs() < EPS);
    }

    #[test]
    fn test_calc_ema_price_ask_ask_higher() {
        let price = calc_ema_price_ask(0.01, 105.0, 101.0, 0.01);
        // ema_price = 102.01, max(105.0, 102.01) = 105.0
        assert!((price - 105.0).abs() < EPS);
    }

    // -- interpolation ------------------------------------------------------

    #[test]
    fn test_interpolate_basic() {
        let xs = [0.0, 1.0, 2.0];
        let ys = [10.0, 20.0, 30.0];
        assert!((interpolate(0.5, &xs, &ys) - 15.0).abs() < EPS);
        assert!((interpolate(1.5, &xs, &ys) - 25.0).abs() < EPS);
    }

    #[test]
    fn test_interpolate_clamp_left() {
        let xs = [1.0, 2.0];
        let ys = [10.0, 20.0];
        assert!((interpolate(0.0, &xs, &ys) - 10.0).abs() < EPS);
    }

    #[test]
    fn test_interpolate_clamp_right() {
        let xs = [1.0, 2.0];
        let ys = [10.0, 20.0];
        assert!((interpolate(5.0, &xs, &ys) - 20.0).abs() < EPS);
    }

    #[test]
    fn test_interpolate_single_point() {
        assert!((interpolate(42.0, &[1.0], &[100.0]) - 100.0).abs() < EPS);
    }

    #[test]
    fn test_interpolate_empty() {
        assert_eq!(interpolate(1.0, &[], &[]), 0.0);
    }

    #[test]
    fn test_interpolate_exact_knot() {
        let xs = [0.0, 1.0, 2.0];
        let ys = [10.0, 20.0, 30.0];
        assert!((interpolate(1.0, &xs, &ys) - 20.0).abs() < EPS);
    }

    #[test]
    fn test_interpolate_duplicate_x() {
        // Two identical x-values — should not divide by zero.
        let xs = [1.0, 1.0, 2.0];
        let ys = [10.0, 20.0, 30.0];
        let result = interpolate(1.0, &xs, &ys);
        assert!(result.is_finite());
    }
}
