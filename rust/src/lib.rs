use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

mod backtest;
mod closes;
mod ema;
mod entries;
mod multi_symbol;
mod orchestrator;
mod risk;
mod trailing;
mod types;
mod utils;

use crate::types::{
    BotParams, EMABands, ExchangeParams, OrderBook, Position, StateParams, TrailingPriceBundle,
};

// ---------- Conversion helpers (Python dict ↔ Rust struct) ----------

fn dict_get_f64(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<f64> {
    dict.get_item(key)?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(format!("missing key: {}", key)))?
        .extract::<f64>()
}

fn dict_get_f64_default(dict: &Bound<'_, PyDict>, key: &str, default: f64) -> PyResult<f64> {
    match dict.get_item(key)? {
        Some(v) => v.extract::<f64>(),
        None => Ok(default),
    }
}

fn dict_get_usize_default(dict: &Bound<'_, PyDict>, key: &str, default: usize) -> PyResult<usize> {
    match dict.get_item(key)? {
        Some(v) => v.extract::<usize>(),
        None => Ok(default),
    }
}

fn bot_params_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<BotParams> {
    Ok(BotParams {
        entry_initial_ema_dist: dict_get_f64(dict, "entry_initial_ema_dist")?,
        entry_initial_qty_pct: dict_get_f64(dict, "entry_initial_qty_pct")?,
        entry_grid_spacing_pct: dict_get_f64(dict, "entry_grid_spacing_pct")?,
        entry_grid_spacing_volatility_weight: dict_get_f64_default(
            dict,
            "entry_grid_spacing_volatility_weight",
            0.0,
        )?,
        entry_grid_spacing_we_weight: dict_get_f64_default(
            dict,
            "entry_grid_spacing_we_weight",
            1.0,
        )?,
        entry_grid_double_down_factor: dict_get_f64(dict, "entry_grid_double_down_factor")?,
        entry_trailing_threshold_pct: dict_get_f64_default(
            dict,
            "entry_trailing_threshold_pct",
            0.01,
        )?,
        entry_trailing_retracement_pct: dict_get_f64_default(
            dict,
            "entry_trailing_retracement_pct",
            0.005,
        )?,
        entry_trailing_grid_ratio: dict_get_f64_default(dict, "entry_trailing_grid_ratio", 0.0)?,
        close_grid_markup_start: dict_get_f64(dict, "close_grid_markup_start")?,
        close_grid_markup_end: dict_get_f64(dict, "close_grid_markup_end")?,
        close_grid_qty_pct: dict_get_f64(dict, "close_grid_qty_pct")?,
        close_trailing_threshold_pct: dict_get_f64_default(
            dict,
            "close_trailing_threshold_pct",
            0.01,
        )?,
        close_trailing_retracement_pct: dict_get_f64_default(
            dict,
            "close_trailing_retracement_pct",
            0.004,
        )?,
        close_trailing_grid_ratio: dict_get_f64_default(dict, "close_trailing_grid_ratio", 0.0)?,
        close_trailing_qty_pct: dict_get_f64_default(dict, "close_trailing_qty_pct", 0.5)?,
        wallet_exposure_limit: dict_get_f64(dict, "wallet_exposure_limit")?,
        n_positions: dict_get_usize_default(dict, "n_positions", 5)?,
        total_wallet_exposure_limit: dict_get_f64_default(
            dict,
            "total_wallet_exposure_limit",
            3.0,
        )?,
        ema_span_0: dict_get_f64_default(dict, "ema_span_0", 385.0)?,
        ema_span_1: dict_get_f64_default(dict, "ema_span_1", 620.0)?,
        unstuck_threshold: dict_get_f64_default(dict, "unstuck_threshold", 0.0)?,
        unstuck_close_pct: dict_get_f64_default(dict, "unstuck_close_pct", 0.05)?,
        unstuck_ema_dist: dict_get_f64_default(dict, "unstuck_ema_dist", 0.01)?,
        unstuck_loss_allowance_pct: dict_get_f64_default(
            dict,
            "unstuck_loss_allowance_pct",
            0.001,
        )?,
        risk_we_excess_allowance_pct: dict_get_f64_default(
            dict,
            "risk_we_excess_allowance_pct",
            0.0,
        )?,
        risk_wel_enforcer_threshold: dict_get_f64_default(
            dict,
            "risk_wel_enforcer_threshold",
            0.98,
        )?,
        risk_twel_enforcer_threshold: dict_get_f64_default(
            dict,
            "risk_twel_enforcer_threshold",
            0.95,
        )?,
    })
}

fn exchange_params_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<ExchangeParams> {
    Ok(ExchangeParams {
        qty_step: dict_get_f64(dict, "qty_step")?,
        price_step: dict_get_f64(dict, "price_step")?,
        min_qty: dict_get_f64(dict, "min_qty")?,
        min_cost: dict_get_f64(dict, "min_cost")?,
        c_mult: dict_get_f64_default(dict, "c_mult", 1.0)?,
        maker_fee: dict_get_f64_default(dict, "maker_fee", 0.0002)?,
        taker_fee: dict_get_f64_default(dict, "taker_fee", 0.0005)?,
    })
}

fn position_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<Position> {
    Ok(Position {
        size: dict_get_f64_default(dict, "size", 0.0)?,
        price: dict_get_f64_default(dict, "price", 0.0)?,
    })
}

fn state_params_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<StateParams> {
    let order_book_dict = dict
        .get_item("order_book")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing key: order_book"))?
        .downcast_into::<PyDict>()?;
    let ema_bands_dict = dict
        .get_item("ema_bands")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing key: ema_bands"))?
        .downcast_into::<PyDict>()?;

    Ok(StateParams {
        balance: dict_get_f64(dict, "balance")?,
        order_book: OrderBook {
            bid: dict_get_f64(&order_book_dict, "bid")?,
            ask: dict_get_f64(&order_book_dict, "ask")?,
        },
        ema_bands: EMABands {
            upper: dict_get_f64(&ema_bands_dict, "upper")?,
            lower: dict_get_f64(&ema_bands_dict, "lower")?,
        },
        entry_volatility_logrange_ema_1h: dict_get_f64_default(
            dict,
            "entry_volatility_logrange_ema_1h",
            0.0,
        )?,
    })
}

fn trailing_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<TrailingPriceBundle> {
    Ok(TrailingPriceBundle {
        min_since_open: dict_get_f64_default(dict, "min_since_open", f64::MAX)?,
        max_since_min: dict_get_f64_default(dict, "max_since_min", f64::MIN)?,
        max_since_open: dict_get_f64_default(dict, "max_since_open", f64::MIN)?,
        min_since_max: dict_get_f64_default(dict, "min_since_max", f64::MAX)?,
    })
}

fn order_to_dict<'py>(py: Python<'py>, order: &crate::types::Order) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("qty", order.qty)?;
    dict.set_item("price", order.price)?;
    dict.set_item("order_type", format!("{:?}", order.order_type))?;
    Ok(dict)
}

// ---------- Public Python API ----------

#[pyfunction]
fn round_step(value: f64, step: f64) -> f64 {
    utils::round_(value, step)
}

#[pyfunction]
fn round_down(value: f64, step: f64) -> f64 {
    utils::round_dn(value, step)
}

#[pyfunction]
fn round_step_up(value: f64, step: f64) -> f64 {
    utils::round_up(value, step)
}

#[pyfunction]
fn wallet_exposure(c_mult: f64, balance: f64, size: f64, price: f64) -> f64 {
    utils::calc_wallet_exposure(c_mult, balance, size, price)
}

#[pyfunction]
fn pnl_long(entry: f64, close: f64, qty: f64, c_mult: f64) -> f64 {
    utils::calc_pnl_long(entry, close, qty, c_mult)
}

#[pyfunction]
fn pnl_short(entry: f64, close: f64, qty: f64, c_mult: f64) -> f64 {
    utils::calc_pnl_short(entry, close, qty, c_mult)
}

#[pyfunction]
fn calc_entries_long<'py>(
    py: Python<'py>,
    bot_params: &Bound<'py, PyDict>,
    exchange_params: &Bound<'py, PyDict>,
    state_params: &Bound<'py, PyDict>,
    position: &Bound<'py, PyDict>,
    trailing: &Bound<'py, PyDict>,
    wel_cap: f64,
    max_levels: usize,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let bp = bot_params_from_dict(bot_params)?;
    let ep = exchange_params_from_dict(exchange_params)?;
    let sp = state_params_from_dict(state_params)?;
    let pos = position_from_dict(position)?;
    let trail = trailing_from_dict(trailing)?;

    let orders = entries::calc_entries_long(&ep, &sp, &bp, &pos, &trail, wel_cap, max_levels);
    orders.iter().map(|o| order_to_dict(py, o)).collect()
}

#[pyfunction]
fn calc_entries_short<'py>(
    py: Python<'py>,
    bot_params: &Bound<'py, PyDict>,
    exchange_params: &Bound<'py, PyDict>,
    state_params: &Bound<'py, PyDict>,
    position: &Bound<'py, PyDict>,
    trailing: &Bound<'py, PyDict>,
    wel_cap: f64,
    max_levels: usize,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let bp = bot_params_from_dict(bot_params)?;
    let ep = exchange_params_from_dict(exchange_params)?;
    let sp = state_params_from_dict(state_params)?;
    let pos = position_from_dict(position)?;
    let trail = trailing_from_dict(trailing)?;

    let orders = entries::calc_entries_short(&ep, &sp, &bp, &pos, &trail, wel_cap, max_levels);
    orders.iter().map(|o| order_to_dict(py, o)).collect()
}

#[pyfunction]
fn calc_closes_long<'py>(
    py: Python<'py>,
    bot_params: &Bound<'py, PyDict>,
    exchange_params: &Bound<'py, PyDict>,
    state_params: &Bound<'py, PyDict>,
    position: &Bound<'py, PyDict>,
    trailing: &Bound<'py, PyDict>,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let bp = bot_params_from_dict(bot_params)?;
    let ep = exchange_params_from_dict(exchange_params)?;
    let sp = state_params_from_dict(state_params)?;
    let pos = position_from_dict(position)?;
    let trail = trailing_from_dict(trailing)?;

    let orders = closes::calc_closes_long(&ep, &sp, &bp, &pos, &trail);
    orders.iter().map(|o| order_to_dict(py, o)).collect()
}

#[pyfunction]
fn calc_closes_short<'py>(
    py: Python<'py>,
    bot_params: &Bound<'py, PyDict>,
    exchange_params: &Bound<'py, PyDict>,
    state_params: &Bound<'py, PyDict>,
    position: &Bound<'py, PyDict>,
    trailing: &Bound<'py, PyDict>,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let bp = bot_params_from_dict(bot_params)?;
    let ep = exchange_params_from_dict(exchange_params)?;
    let sp = state_params_from_dict(state_params)?;
    let pos = position_from_dict(position)?;
    let trail = trailing_from_dict(trailing)?;

    let orders = closes::calc_closes_short(&ep, &sp, &bp, &pos, &trail);
    orders.iter().map(|o| order_to_dict(py, o)).collect()
}

#[pyfunction]
fn update_trailing(
    trailing: &Bound<'_, PyDict>,
    high: f64,
    low: f64,
    close: f64,
) -> PyResult<PyObject> {
    let mut bundle = trailing_from_dict(trailing)?;
    trailing::update_trailing_bundle(&mut bundle, high, low, close);

    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("min_since_open", bundle.min_since_open)?;
        dict.set_item("max_since_min", bundle.max_since_min)?;
        dict.set_item("max_since_open", bundle.max_since_open)?;
        dict.set_item("min_since_max", bundle.min_since_max)?;
        Ok(dict.into())
    })
}

#[pyfunction]
fn twel_allows_entry(
    balance: f64,
    c_mult: f64,
    current_positions: Vec<(f64, f64)>,
    new_qty: f64,
    new_price: f64,
    twel_limit: f64,
) -> bool {
    risk::twel_allows_entry(
        balance,
        c_mult,
        &current_positions,
        new_qty,
        new_price,
        twel_limit,
    )
}

#[pyfunction]
fn total_wallet_exposure(balance: f64, c_mult: f64, positions: Vec<(f64, f64)>) -> f64 {
    risk::total_wallet_exposure(balance, c_mult, &positions)
}

#[pyfunction]
fn loss_gate_allows(
    realized_pnl_peak: f64,
    realized_pnl_current: f64,
    projected_pnl: f64,
    balance_peak: f64,
    max_realized_loss_pct: f64,
) -> bool {
    risk::loss_gate_allows(
        realized_pnl_peak,
        realized_pnl_current,
        projected_pnl,
        balance_peak,
        max_realized_loss_pct,
    )
}

#[pyfunction]
#[pyo3(signature = (
    candles,
    bot_params,
    exchange_params,
    starting_balance = 10000.0,
    funding_rate = 0.0,
    funding_interval_bars = 480,
    liquidation_threshold_pct = 0.05,
    max_grid_levels = 5,
))]
fn run_backtest<'py>(
    py: Python<'py>,
    candles: &Bound<'py, PyArray2<f64>>,
    bot_params: &Bound<'py, PyDict>,
    exchange_params: &Bound<'py, PyDict>,
    starting_balance: f64,
    funding_rate: f64,
    funding_interval_bars: usize,
    liquidation_threshold_pct: f64,
    max_grid_levels: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let bp = bot_params_from_dict(bot_params)?;
    let ep = exchange_params_from_dict(exchange_params)?;
    let cfg = backtest::BacktestConfig {
        starting_balance,
        funding_rate,
        funding_interval_bars,
        liquidation_threshold_pct,
        max_grid_levels,
    };

    let arr = unsafe { candles.as_array() };
    let shape = arr.shape();
    if shape.len() != 2 || shape[1] != 5 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "candles must be shape [N, 5] (open, high, low, close, volume)",
        ));
    }
    let n_rows = shape[0];
    let mut candle_vec: Vec<[f64; 5]> = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        candle_vec.push([
            arr[[i, 0]],
            arr[[i, 1]],
            arr[[i, 2]],
            arr[[i, 3]],
            arr[[i, 4]],
        ]);
    }

    let result = py.allow_threads(|| backtest::run_backtest(&candle_vec, &bp, &ep, &cfg));

    let out = PyDict::new(py);
    out.set_item("final_balance", result.final_balance)?;
    out.set_item("final_equity", result.final_equity)?;
    out.set_item("max_drawdown", result.max_drawdown)?;
    out.set_item("n_trades", result.n_trades)?;
    out.set_item("liquidated", result.liquidated)?;
    out.set_item("liquidation_bar", result.liquidation_bar)?;
    out.set_item("equity_curve", result.equity_curve)?;

    let fills_list: Vec<Bound<'py, PyDict>> = result
        .fills
        .iter()
        .map(|f| {
            let d = PyDict::new(py);
            d.set_item("bar_index", f.bar_index)?;
            d.set_item("side", f.side)?;
            d.set_item("qty", f.qty)?;
            d.set_item("price", f.price)?;
            d.set_item("fee", f.fee)?;
            d.set_item("pnl", f.pnl)?;
            d.set_item("order_type", format!("{:?}", f.order_type))?;
            Ok(d)
        })
        .collect::<PyResult<Vec<_>>>()?;
    out.set_item("fills", fills_list)?;

    Ok(out)
}

#[pymodule]
fn combo_futures_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(round_step, m)?)?;
    m.add_function(wrap_pyfunction!(round_down, m)?)?;
    m.add_function(wrap_pyfunction!(round_step_up, m)?)?;
    m.add_function(wrap_pyfunction!(wallet_exposure, m)?)?;
    m.add_function(wrap_pyfunction!(pnl_long, m)?)?;
    m.add_function(wrap_pyfunction!(pnl_short, m)?)?;
    m.add_function(wrap_pyfunction!(calc_entries_long, m)?)?;
    m.add_function(wrap_pyfunction!(calc_entries_short, m)?)?;
    m.add_function(wrap_pyfunction!(calc_closes_long, m)?)?;
    m.add_function(wrap_pyfunction!(calc_closes_short, m)?)?;
    m.add_function(wrap_pyfunction!(update_trailing, m)?)?;
    m.add_function(wrap_pyfunction!(twel_allows_entry, m)?)?;
    m.add_function(wrap_pyfunction!(total_wallet_exposure, m)?)?;
    m.add_function(wrap_pyfunction!(loss_gate_allows, m)?)?;
    m.add_function(wrap_pyfunction!(run_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(run_multi_symbol_backtest, m)?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (
    candles_per_symbol,
    bot_params_per_symbol,
    exchange_params_per_symbol,
    starting_balance = 10000.0,
    funding_rate = 0.0,
    funding_interval_bars = 480,
    liquidation_threshold_pct = 0.05,
    max_grid_levels = 5,
    n_positions_max = 5,
    forager_volume_weight = 0.23,
    forager_volatility_weight = 0.71,
    forager_ema_readiness_weight = 0.06,
))]
fn run_multi_symbol_backtest<'py>(
    py: Python<'py>,
    candles_per_symbol: Vec<Bound<'py, PyArray2<f64>>>,
    bot_params_per_symbol: Vec<Bound<'py, PyDict>>,
    exchange_params_per_symbol: Vec<Bound<'py, PyDict>>,
    starting_balance: f64,
    funding_rate: f64,
    funding_interval_bars: usize,
    liquidation_threshold_pct: f64,
    max_grid_levels: usize,
    n_positions_max: usize,
    forager_volume_weight: f64,
    forager_volatility_weight: f64,
    forager_ema_readiness_weight: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let n_symbols = candles_per_symbol.len();
    if n_symbols == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("no symbols provided"));
    }
    if bot_params_per_symbol.len() != n_symbols
        || exchange_params_per_symbol.len() != n_symbols
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "candles, bot_params, exchange_params must have same length",
        ));
    }

    let mut candles_vec: Vec<Vec<[f64; 5]>> = Vec::with_capacity(n_symbols);
    for arr in candles_per_symbol {
        let view = unsafe { arr.as_array() };
        let shape = view.shape();
        if shape.len() != 2 || shape[1] != 5 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "each candle array must be shape [N, 5]",
            ));
        }
        let n = shape[0];
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            rows.push([
                view[[i, 0]], view[[i, 1]], view[[i, 2]],
                view[[i, 3]], view[[i, 4]],
            ]);
        }
        candles_vec.push(rows);
    }

    let bp_vec: Vec<types::BotParams> = bot_params_per_symbol
        .iter()
        .map(|d| bot_params_from_dict(d))
        .collect::<PyResult<Vec<_>>>()?;
    let ep_vec: Vec<types::ExchangeParams> = exchange_params_per_symbol
        .iter()
        .map(|d| exchange_params_from_dict(d))
        .collect::<PyResult<Vec<_>>>()?;

    let cfg = multi_symbol::MultiSymbolConfig {
        starting_balance,
        funding_rate,
        funding_interval_bars,
        liquidation_threshold_pct,
        max_grid_levels,
        n_positions_max,
    };
    let weights = multi_symbol::ForagerWeights {
        volume: forager_volume_weight,
        volatility: forager_volatility_weight,
        ema_readiness: forager_ema_readiness_weight,
    };

    let result = py.allow_threads(|| {
        multi_symbol::run_multi_symbol_backtest(&candles_vec, &bp_vec, &ep_vec, &cfg, weights)
    });

    let out = PyDict::new(py);
    out.set_item("final_balance", result.final_balance)?;
    out.set_item("final_equity", result.final_equity)?;
    out.set_item("max_drawdown", result.max_drawdown)?;
    out.set_item("n_trades", result.n_trades)?;
    out.set_item("liquidated", result.liquidated)?;
    out.set_item("liquidation_bar", result.liquidation_bar)?;
    out.set_item("equity_curve", result.equity_curve)?;

    let fills_list: Vec<Bound<'py, PyDict>> = result
        .fills
        .iter()
        .map(|f| {
            let d = PyDict::new(py);
            d.set_item("bar_index", f.bar_index)?;
            d.set_item("symbol_idx", f.symbol_idx)?;
            d.set_item("side", f.side)?;
            d.set_item("qty", f.qty)?;
            d.set_item("price", f.price)?;
            d.set_item("fee", f.fee)?;
            d.set_item("pnl", f.pnl)?;
            d.set_item("order_type", format!("{:?}", f.order_type))?;
            Ok(d)
        })
        .collect::<PyResult<Vec<_>>>()?;
    out.set_item("fills", fills_list)?;

    let final_positions: Vec<Bound<'py, PyDict>> = result
        .final_positions
        .iter()
        .map(|(idx, l, s)| {
            let d = PyDict::new(py);
            d.set_item("symbol_idx", *idx)?;
            d.set_item("long_size", l.size)?;
            d.set_item("long_price", l.price)?;
            d.set_item("short_size", s.size)?;
            d.set_item("short_price", s.price)?;
            Ok(d)
        })
        .collect::<PyResult<Vec<_>>>()?;
    out.set_item("final_positions", final_positions)?;

    Ok(out)
}
