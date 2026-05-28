use crate::types::TrailingPriceBundle;

#[inline]
pub fn reset_trailing_bundle(bundle: &mut TrailingPriceBundle) {
    *bundle = TrailingPriceBundle::default();
}

#[inline]
pub fn update_trailing_bundle(bundle: &mut TrailingPriceBundle, high: f64, low: f64, close: f64) {
    if !high.is_finite() || !low.is_finite() || !close.is_finite() {
        return;
    }

    if low < bundle.min_since_open {
        bundle.min_since_open = low;
        bundle.max_since_min = close;
    } else {
        bundle.max_since_min = bundle.max_since_min.max(high);
    }

    if high > bundle.max_since_open {
        bundle.max_since_open = high;
        bundle.min_since_max = close;
    } else {
        bundle.min_since_max = bundle.min_since_max.min(low);
    }
}

pub fn update_trailing_bundle_sequence(
    bundle: &mut TrailingPriceBundle,
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
) {
    for ((&h, &l), &c) in highs.iter().zip(lows.iter()).zip(closes.iter()) {
        update_trailing_bundle(bundle, h, l, c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_min_and_max_correctly() {
        let mut b = TrailingPriceBundle::default();
        update_trailing_bundle(&mut b, 11.0, 9.0, 10.0);
        assert_eq!(b.min_since_open, 9.0);
        assert_eq!(b.max_since_min, 10.0);
        assert_eq!(b.max_since_open, 11.0);
        assert_eq!(b.min_since_max, 10.0);

        update_trailing_bundle(&mut b, 8.0, 7.0, 7.5);
        assert_eq!(b.min_since_open, 7.0);
        assert_eq!(b.max_since_min, 7.5);

        update_trailing_bundle(&mut b, 15.0, 13.0, 14.0);
        assert_eq!(b.max_since_min, 15.0);
        assert_eq!(b.max_since_open, 15.0);
        assert_eq!(b.min_since_max, 14.0);
    }

    #[test]
    fn ignores_non_finite() {
        let mut b = TrailingPriceBundle::default();
        update_trailing_bundle(&mut b, 11.0, 9.0, 10.0);
        let snap = b.clone();
        update_trailing_bundle(&mut b, f64::NAN, 8.0, 9.0);
        assert_eq!(b.min_since_open, snap.min_since_open);
    }

    #[test]
    fn reset_restores_default() {
        let mut b = TrailingPriceBundle::default();
        update_trailing_bundle(&mut b, 11.0, 9.0, 10.0);
        reset_trailing_bundle(&mut b);
        assert_eq!(b.min_since_open, f64::MAX);
    }
}
