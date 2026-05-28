/// Incremental EMA state.
///
/// Tracks two parallel EMA spans (fast + slow) so the consumer can derive
/// upper/lower bands cheaply.
#[derive(Debug, Clone, Copy)]
pub struct EmaState {
    pub alpha_0: f64,
    pub alpha_1: f64,
    pub ema_0: f64,
    pub ema_1: f64,
    pub initialized: bool,
}

impl EmaState {
    #[inline]
    pub fn new(span_0: f64, span_1: f64) -> Self {
        Self {
            alpha_0: 2.0 / (span_0 + 1.0),
            alpha_1: 2.0 / (span_1 + 1.0),
            ema_0: 0.0,
            ema_1: 0.0,
            initialized: false,
        }
    }

    #[inline]
    pub fn update(&mut self, price: f64) {
        if !price.is_finite() || price <= 0.0 {
            return;
        }
        if !self.initialized {
            self.ema_0 = price;
            self.ema_1 = price;
            self.initialized = true;
            return;
        }
        self.ema_0 = self.alpha_0 * price + (1.0 - self.alpha_0) * self.ema_0;
        self.ema_1 = self.alpha_1 * price + (1.0 - self.alpha_1) * self.ema_1;
    }

    #[inline]
    pub fn upper(&self) -> f64 {
        self.ema_0.max(self.ema_1)
    }

    #[inline]
    pub fn lower(&self) -> f64 {
        self.ema_0.min(self.ema_1)
    }
}

/// Incremental volatility state: EMA of the per-bar log-range.
///
/// `log_range = ln(high / low)` is used as a noise-resistant volatility proxy.
#[derive(Debug, Clone, Copy)]
pub struct VolatilityState {
    pub alpha: f64,
    pub value: f64,
    pub initialized: bool,
}

impl VolatilityState {
    #[inline]
    pub fn new(span_hours: f64) -> Self {
        // span is in hours; convert to bars assuming 1m candles → 60 bars/h.
        let span_bars = (span_hours * 60.0).max(1.0);
        Self {
            alpha: 2.0 / (span_bars + 1.0),
            value: 0.0,
            initialized: false,
        }
    }

    #[inline]
    pub fn update(&mut self, high: f64, low: f64) {
        if !high.is_finite() || !low.is_finite() || high <= 0.0 || low <= 0.0 || high < low {
            return;
        }
        let log_range = (high / low).ln();
        if !log_range.is_finite() || log_range < 0.0 {
            return;
        }
        if !self.initialized {
            self.value = log_range;
            self.initialized = true;
            return;
        }
        self.value = self.alpha * log_range + (1.0 - self.alpha) * self.value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ema_converges_to_constant_input() {
        let mut e = EmaState::new(10.0, 20.0);
        for _ in 0..200 {
            e.update(100.0);
        }
        assert!((e.ema_0 - 100.0).abs() < 1e-6);
        assert!((e.ema_1 - 100.0).abs() < 1e-6);
    }

    #[test]
    fn ema_bands_distinguish_when_diverging() {
        let mut e = EmaState::new(5.0, 50.0);
        for i in 0..100 {
            e.update(100.0 + i as f64);
        }
        assert!(e.upper() > e.lower());
    }

    #[test]
    fn ema_ignores_invalid_prices() {
        let mut e = EmaState::new(10.0, 20.0);
        e.update(100.0);
        let snap = (e.ema_0, e.ema_1);
        e.update(f64::NAN);
        e.update(-50.0);
        e.update(0.0);
        assert_eq!(e.ema_0, snap.0);
        assert_eq!(e.ema_1, snap.1);
    }

    #[test]
    fn volatility_increases_with_wider_range() {
        let mut v = VolatilityState::new(1.0);
        for _ in 0..100 {
            v.update(101.0, 100.0);
        }
        let calm = v.value;
        for _ in 0..100 {
            v.update(110.0, 100.0);
        }
        assert!(v.value > calm);
    }

    #[test]
    fn volatility_handles_degenerate_input() {
        let mut v = VolatilityState::new(1.0);
        v.update(100.0, 100.0);
        v.update(f64::NAN, 100.0);
        v.update(100.0, 110.0);
        assert!(v.value.is_finite());
    }
}
