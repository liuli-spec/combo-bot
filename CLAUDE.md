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
pip install -e ".[all]"     # core + pandas, pyarrow, optuna, UI deps
pip install -e ".[ui]"      # core + FastAPI/uvicorn/jinja2 for web UI only
```

### Build Rust extension (required for Rust backtest/optimize paths)

```bash
cd rust && maturin build --release
pip install rust/target/wheels/*.whl
```

### Tests

```bash
python -m pytest tests/                        # all Python tests
python -m pytest tests/test_grid.py -v         # single test file
python -m pytest tests/ -k "not rust" -v       # skip Rust-dependent tests
cd rust && cargo test --release --lib           # Rust unit tests
```

`tests/test_round_N.py` files are regression snapshots from consecutive tuning rounds — treat them as integration fixtures.

### Lint / format

```bash
cd rust && cargo fmt --check
cd rust && cargo clippy --release
ruff check .                # Python lint (if ruff is installed)
black .                     # Python format (if black is installed)
```

### CLI commands

```bash
combo-futures download --exchange binance --symbol BTC/USDT:USDT --timeframe 1h
combo-futures backtest --config config.json
combo-futures optimize --config config.json --trials 200
combo-futures live --config config.json                   # dry-run (default)
combo-futures live --config config.json --real            # REAL trading (requires confirmation)
combo-futures live --config config.json --testnet         # testnet exchange
combo-futures live --config config.json --clear-stuck     # manually clear persisted fill-stream STUCK state
combo-futures ui   --config config.json --port 8765       # web console
```

State files are segregated by profile: `state.dryrun.json`, `state.testnet.json`, `state.real.json` (overridable via `state_file` in config).

### Web UI launcher

```bash
./start.sh                  # Linux/terminal — opens http://127.0.0.1:8765
# macOS: double-click 启动机器人.command
```

The `ui` command spawns the live trader as a subprocess (`process_manager.py`) and serves a FastAPI dashboard at `combo_bot/webui/`.

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

### Optional module graph (`combo_bot/fusion_config.py`)

All non-core modules are opt-in — enabled by the presence of their config block. `build_fusion(cfg)` is the single factory that constructs and returns a dict of optional components:

| Module | Config key | Purpose |
|--------|-----------|---------|
| `regime.RegimeArbiter` | `regime` | Multi-indicator regime scorer; overlays scale factors on order sizes |
| `sizing.KellySizer` | `kelly` | Kelly-fraction position sizing tracked per `OrderSource` |
| `correlation.CorrelationGate` | `correlation` | Blocks entries when pair correlation exceeds threshold |
| `vol_target.VolTargetSizer` | `vol_target` | Scales orders to hit a target annualised volatility |
| `protections.ProtectionManager` | `protections[]` | Pluggable per-symbol+side locks (`StoplossGuard`, `CooldownPeriod`, …) |
| `strategy.IStrategy` | `strategy.class` | Freqtrade-style callback hooks applied on top of the core engine |

All optional modules implement a `filter_orders()` interface and are applied in `LiveTrader._tick()` after the core decision pipeline.

### Reliability infrastructure

- **`intent_journal.py`** — append-only write-ahead log (`live_state.intent_journal.jsonl`) that tracks every order from submission through fill/cancel. Replayed on restart to recover unknown orders without re-querying the exchange.
- **`fill_events_manager.py`** — deduplicates exchange trade history, bridges confirmed fills to `KellySizer`, `ProtectionManager`, and `AccountState.add_realized_pnl`. Fetches are wrapped in in-tick exponential-backoff retry (`fetch_max_retries` / `fetch_retry_base_ms`) so a transient blip never reaches the fail-closed path. STUCK state is tagged by **reason**: `cursor` (same-ms pagination stall — a real ledger-integrity risk, persisted across restart, needs operator `clear_stuck`) vs `fetch` (transient — never persistently parks; only the single-tick `last_poll_failed` blocks new risk, and it self-heals on the next poll). On restart only `cursor`-reason STUCK is restored.
- **`freshness.py` (`FreshnessLedger`)** — epoch-based data-surface freshness gate (ported from passivbot). `LiveTrader._tick` calls `begin_epoch()`; each successful refresh `stamp`s its surface (`balance`, `positions`, per-symbol `candle:<symbol>`). A failed per-symbol candle fetch `flag_symbol_block`s that symbol, which `_risk_increasing_blocked` honors; the block self-heals once the surface refreshes at/after `min_epoch`. Not persisted (runtime-only).
- **`kill_switch.py`** — standalone async utility that cancels all open orders and market-closes all positions for a given symbol list. Can be run as `python -m combo_bot.kill_switch`.
- **`hsl.py` (`HslSupervisor`)** — pure classification layer (SAFE / WARN / ALERT / HALT) based on drawdown EMA. Has no side effects; enforcement is done by `RiskManager`.
- **`monitor.py`** — read-only CLI observer: `python -m combo_bot.monitor --config config.json` prints a live snapshot of positions, fills, and exchange state without touching orders.

### Synthetic (reconstructed) realized PnL

`LiveTrader._enrich_fill_pnl` reconstructs `realized_pnl` for reduce-only fills when the exchange returns none (e.g. non-Binance). If the close qty exceeds the locally-known bucket size, the cost basis is incomplete and the fill is flagged **`pnl_degraded`** (`types.Fill.pnl_degraded`). Degraded PnL is still booked to the account ledger (equity / HSL stay whole) but is **withheld from the `KellySizer`** edge estimator, so position sizing never compounds off a guessed PnL.

### Web UI (`combo_bot/webui/`)

- **`server.py`** — FastAPI app. Operator endpoints: `/api/status`, `/api/equity`, `/api/fills`, `/api/logs/stream` (SSE), `/api/control/{start,stop,kill,clear_sentinel,clear_stuck}`. Lab endpoints: `/api/backtest/run`, `/api/optimize/run`, `/api/job/{id}`, `/api/jobs`. Exchange data is refreshed by a background asyncio task (singleton ccxt connection) every 10s rather than per request; fills are read incrementally from the JSONL sidecar; the equity curve is persisted to a `*.equity.jsonl` sidecar so it survives a UI restart. Backtest/optimize run in a thread pool as tracked jobs (capped, oldest finished evicted); optimize results are also written to `<data_dir>/optimize_results/`.
- **`process_manager.py`** (`TraderProcessManager`) — manages the `combo-futures live` subprocess lifecycle (incl. `--clear-stuck` passthrough), captures stdout, and exposes `start()`/`stop()` coroutines.
- `static/` + `templates/` — Jinja2 HTML + vanilla JS (Chart.js via CDN with SRI), no build step. Two tabs: the operator dashboard and the backtest/optimize lab.

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
