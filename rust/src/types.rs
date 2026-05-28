use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Side constants
// ---------------------------------------------------------------------------
pub const LONG: usize = 0;
pub const SHORT: usize = 1;

// ---------------------------------------------------------------------------
// Candle OHLCV index constants
// ---------------------------------------------------------------------------
pub const HIGH: usize = 0;
pub const LOW: usize = 1;
pub const CLOSE: usize = 2;
pub const VOLUME: usize = 3;

// ---------------------------------------------------------------------------
// Core parameter structs
// ---------------------------------------------------------------------------

/// Per-symbol bot configuration. Controls grid spacing, trailing behaviour,
/// wallet-exposure limits, unstuck logic, and risk enforcers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BotParams {
    /// Distance from EMA for the first (initial) entry.
    pub entry_initial_ema_dist: f64,
    /// Initial entry quantity as a fraction of `wallet_exposure_limit * balance`.
    pub entry_initial_qty_pct: f64,
    /// Base percentage spacing between grid levels.
    pub entry_grid_spacing_pct: f64,
    /// Weight of volatility in grid spacing calculation.
    pub entry_grid_spacing_volatility_weight: f64,
    /// Weight of current wallet exposure in grid spacing calculation.
    pub entry_grid_spacing_we_weight: f64,
    /// Multiplicative factor for each successive grid level quantity.
    pub entry_grid_double_down_factor: f64,
    /// Price move (%) required before trailing entry activates.
    pub entry_trailing_threshold_pct: f64,
    /// Retracement (%) from extreme that triggers trailing entry fill.
    pub entry_trailing_retracement_pct: f64,
    /// Blend ratio between grid and trailing entry styles (-1.0 = pure grid, 1.0 = pure trailing).
    pub entry_trailing_grid_ratio: f64,

    /// Markup (%) for the first close-grid level.
    pub close_grid_markup_start: f64,
    /// Markup (%) for the last close-grid level.
    pub close_grid_markup_end: f64,
    /// Quantity fraction per close-grid level.
    pub close_grid_qty_pct: f64,
    /// Price move (%) required before trailing close activates.
    pub close_trailing_threshold_pct: f64,
    /// Retracement (%) from extreme that triggers trailing close fill.
    pub close_trailing_retracement_pct: f64,
    /// Blend ratio between grid and trailing close styles.
    pub close_trailing_grid_ratio: f64,
    /// Quantity fraction for trailing close orders.
    pub close_trailing_qty_pct: f64,

    /// Maximum wallet exposure for this symbol (single side).
    pub wallet_exposure_limit: f64,
    /// Number of concurrent positions allowed.
    pub n_positions: usize,
    /// Total wallet exposure limit across all positions (single side).
    pub total_wallet_exposure_limit: f64,

    /// Span (in bars) for the fast EMA.
    pub ema_span_0: f64,
    /// Span (in bars) for the slow EMA.
    pub ema_span_1: f64,

    /// Wallet-exposure ratio that triggers unstuck logic.
    pub unstuck_threshold: f64,
    /// Fraction of position to close when unstucking.
    pub unstuck_close_pct: f64,
    /// EMA distance used to compute the unstuck close price.
    pub unstuck_ema_dist: f64,
    /// Maximum allowed realised loss (as fraction of balance) for an unstuck close.
    pub unstuck_loss_allowance_pct: f64,

    /// Excess wallet-exposure allowance before risk enforcer kicks in.
    pub risk_we_excess_allowance_pct: f64,
    /// WEL threshold at which the risk enforcer begins acting.
    pub risk_wel_enforcer_threshold: f64,
    /// Total-WEL threshold at which the risk enforcer begins acting.
    pub risk_twel_enforcer_threshold: f64,
}

/// Exchange-specific parameters that govern rounding, minimum sizes, fees, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeParams {
    /// Minimum quantity increment.
    pub qty_step: f64,
    /// Minimum price increment.
    pub price_step: f64,
    /// Minimum order quantity.
    pub min_qty: f64,
    /// Minimum order cost (qty * price * c_mult).
    pub min_cost: f64,
    /// Contract multiplier (1.0 for USDT-margined linear contracts).
    pub c_mult: f64,
    /// Maker (limit) fee rate (e.g. 0.0002 = 0.02%).
    pub maker_fee: f64,
    /// Taker (market) fee rate (e.g. 0.0005 = 0.05%).
    pub taker_fee: f64,
}

// ---------------------------------------------------------------------------
// Market / position state
// ---------------------------------------------------------------------------

/// Current position for one side of a symbol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Signed size: positive = long, negative = short.
    pub size: f64,
    /// Volume-weighted average entry price.
    pub price: f64,
}

impl Default for Position {
    fn default() -> Self {
        Self {
            size: 0.0,
            price: 0.0,
        }
    }
}

/// Top-of-book snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub bid: f64,
    pub ask: f64,
}

/// Upper and lower EMA bands used for entry/close price calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EMABands {
    pub upper: f64,
    pub lower: f64,
}

/// Trailing-price state that tracks extremes since position open.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrailingPriceBundle {
    /// Lowest price observed since the position was opened.
    pub min_since_open: f64,
    /// Highest price observed since `min_since_open` was set.
    pub max_since_min: f64,
    /// Highest price observed since the position was opened.
    pub max_since_open: f64,
    /// Lowest price observed since `max_since_open` was set.
    pub min_since_max: f64,
}

impl Default for TrailingPriceBundle {
    fn default() -> Self {
        Self {
            min_since_open: f64::MAX,
            max_since_min: f64::MIN,
            max_since_open: f64::MIN,
            min_since_max: f64::MAX,
        }
    }
}

// ---------------------------------------------------------------------------
// Orders
// ---------------------------------------------------------------------------

/// Classifies every order the engine can emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    EntryInitialNormalLong,
    EntryInitialNormalShort,
    EntryGridNormalLong,
    EntryGridNormalShort,
    EntryTrailingNormalLong,
    EntryTrailingNormalShort,
    CloseGridLong,
    CloseGridShort,
    CloseTrailingLong,
    CloseTrailingShort,
    CloseUnstuckLong,
    CloseUnstuckShort,
    ClosePanicLong,
    ClosePanicShort,
}

/// A single order to be placed on the exchange.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub qty: f64,
    pub price: f64,
    pub order_type: OrderType,
}

// ---------------------------------------------------------------------------
// Composite state passed into the calculation engine
// ---------------------------------------------------------------------------

/// Snapshot of everything the engine needs to compute orders for one symbol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateParams {
    pub balance: f64,
    pub order_book: OrderBook,
    pub ema_bands: EMABands,
    /// Exponential moving average of 1-hour log-range, used as a volatility proxy.
    pub entry_volatility_logrange_ema_1h: f64,
}

// ---------------------------------------------------------------------------
// Trading mode & trend signal
// ---------------------------------------------------------------------------

/// Operating mode that restricts which order types the engine may emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradingMode {
    /// Full grid: entries + closes.
    Normal,
    /// Take-profit only: no new entries, only close orders.
    TpOnly,
    /// Graceful wind-down: close existing positions, no new entries.
    GracefulStop,
    /// Emergency: market-close everything immediately.
    Panic,
}

impl Default for TradingMode {
    fn default() -> Self {
        Self::Normal
    }
}

/// External trend signal that higher-level strategies can inject.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendSignal {
    /// Signed direction: positive = bullish, negative = bearish, 0 = neutral.
    pub direction: f64,
    /// Absolute strength in [0.0, 1.0].
    pub strength: f64,
}

impl Default for TrendSignal {
    fn default() -> Self {
        Self {
            direction: 0.0,
            strength: 0.0,
        }
    }
}
