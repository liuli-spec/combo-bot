# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Combo Futures

Combined grid + trend futures trading bot. Merges Passivbot-style grid logic with multi-indicator trend signals into a single decision engine, managed by layered risk controls.

## Tech Stack

- Python 3.11+ (`combo_bot/` package)
- Rust (`rust/` crate: `combo_futures_core`) — optional PyO3 extension for hot paths
- ccxt — exchange connectivity
- numpy — numerical computation
- optuna — hyperparameter optimization (optional)
- maturin — builds the Rust wheel

## Commands

### Install

```bash
pip install -e .            # core only
pip install -e ".[all]"     # core + pandas, pyarrow, optuna
```

### Build Rust extension (required for Rust backtest/optimize paths)

```bash
cd rust && maturin build --release
pip install rust/target/wheels/*.whl
```

### Tests

```bash
python -m pytest tests/                     # all Python tests
python -m pytest tests/test_grid.py -v      # single test file
cd rust && cargo test --release --lib        # Rust unit tests
```

### Lint / format (Rust)

```bash
cd rust && cargo fmt --check
cd rust && cargo clippy --release
```

### CLI commands

```bash
combo-futures download --exchange binance --symbol BTC/USDT:USDT --timeframe 1h
combo-futures backtest --config config.json
combo-futures optimize --config config.json --trials 200
combo-futures live --config config.json
```

## Architecture

### Data types (`combo_bot/types.py`)

All shared domain types live here. Key types:
- `Candle`, `Order`, `Fill` — market data and execution primitives
- `Position`, `AccountState`, `SymbolState` — mutable trading state
- `TrendSignal`, `TrendRegime` — trend layer output
- `TradingMode` — `NORMAL | TP_ONLY | GRACEFUL_STOP | PANIC`
- `OrderSource` — `GRID | TREND | RISK`, used to attribute PnL

### Decision pipeline

Each tick flows through three sequential filter stages:

```
GridEngine.compute_orders()        → list[Order]
  └─ DecisionMerger.filter_grid_orders()   (trim entries in adverse trend)
  └─ DecisionMerger.generate_trend_orders() (add trend-driven entries)
  └─ RiskManager.filter_orders()           (final gate — may panic-close)
```

`DecisionMerger` (`merger.py`) computes per-side `TradingMode` from `TrendSignal`, limits grid entry depth in counter-trend conditions, and injects trend overlay orders at strong-regime thresholds.

`RiskManager` (`risk.py`) implements a 4-tier drawdown guard (GREEN/YELLOW/ORANGE/RED). RED triggers `_panic_close_all` and a configurable cooldown before new entries are allowed again.

### Dual backtest/optimize paths

There are two parallel implementations. The Python path is the reference; the Rust path is the performance-optimised variant:

| | Python | Rust |
|---|---|---|
| Single symbol | `backtest.Backtester` | `rust_backtest.run_rust_backtest` |
| Multi-symbol | `backtest.Backtester` (loop) | `rust_multi_symbol.run_multi_symbol_backtest` |
| Optimizer | `optimize.Optimizer` | `rust_optimize.RustOptimizer` / `RustMultiSymbolOptimizer` |
| Adapter | — | `rust_adapter.compute_grid_orders_rust` |

`rust_adapter.py` bridges the Python `GridConfig`/`EMAState`/`VolatilityState` types to the dict-based API the Rust extension expects. If `combo_futures_core` is not installed the Python path is used as fallback (with a log warning).

### Strategy plugin layer (`combo_bot/strategy.py`)

A Freqtrade-inspired callback interface. Subclass `IStrategy` and override `populate_indicators`, `populate_entry_trend`, and `populate_exit_trend`. Optional hooks: `confirm_trade_entry`, `confirm_trade_exit`, `custom_stoploss`, `adjust_trade_position`, etc. `StrategyRunner` applies these callbacks to the order stream produced by the core engine. `DefaultStrategy` (no-op) and `ExampleTrendStrategy` (RSI + EMA crossover) are built in.

### ForagerScorer (`combo_bot/grid_engine.py`)

Ranks candidate symbols by a weighted score of volume, volatility, and EMA-readiness to select the top `n_positions` symbols dynamically. Used by the multi-symbol backtest and live trader.

### Config

`config.example.json` shows the full config schema with nested `grid`, `trend`, `merger`, and `risk` blocks. Copy to `config.json` and set exchange credentials via environment variables or a `.env` file.

### Rust crate (`rust/src/`)

| File | Purpose |
|------|---------|
| `lib.rs` | PyO3 bindings — exports `calc_entries_long/short`, `calc_closes_long/short`, `run_backtest`, `run_multi_symbol_backtest` |
| `entries.rs` | Grid entry order computation |
| `closes.rs` | Grid close order computation |
| `backtest.rs` | Single-symbol event loop |
| `multi_symbol.rs` | Multi-symbol loop with Forager selection |
| `orchestrator.rs` | Per-bar order orchestration |
| `risk.rs` | Rust-side risk checks |
| `trailing.rs` | Trailing entry/close logic |
| `ema.rs` | EMA band computation |
| `types.rs` | Rust-side structs mirroring Python types |
| `utils.rs` | Quantization helpers |
