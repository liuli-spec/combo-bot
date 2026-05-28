# combo-futures

> Grid + trend fusion for crypto perpetual futures — Rust core, Python interface.

![Status: Alpha](https://img.shields.io/badge/status-alpha-orange)
![Tests: 149 passing](https://img.shields.io/badge/tests-149%20passing-brightgreen)
![License: MIT](https://img.shields.io/badge/license-MIT-blue)
![Python: 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Rust: 2021](https://img.shields.io/badge/rust-2021-orange)

> **Status: Alpha — NOT battle-tested on real money.** Backtest results do **not** predict live performance. Read the [risk warnings](#risk-warnings) before doing anything with real capital.

---

## Why combo-futures?

Two open-source bots dominate the perp-futures niche, and each has a glaring gap:

| | Passivbot | Freqtrade | **combo-futures** |
|---|---|---|---|
| Grid logic | Best-in-class | Weak | Best-in-class (reimplemented) |
| Trend / signal layer | None | Best-in-class | Adopted from Freqtrade |
| Backtester speed | Rust (~1M cdl/s) | Python (~7K cdl/s) | **Rust (1.6M cdl/s)** |
| Strategy callbacks | None | 17 hooks | 17 hooks |
| Unified risk engine | Per-coin only | Per-trade only | **Layered: per-coin + portfolio** |

combo-futures fuses grid trading (Passivbot-style) with trend signals (Freqtrade-style) behind a single Rust decision engine. Neither original bot does grid+trend well — this one does.

It is an **independent reimplementation**. No code was copied from either project; see [CREDITS.md](CREDITS.md).

---

## Quick start

```bash
git clone https://github.com/combo-futures/combo-futures
cd combo-futures

# Install the Rust core (PyO3 extension)
pip install maturin
cd rust && maturin build --release && cd ..
pip install rust/target/wheels/combo_futures_core-*.whl

# Install the Python package
pip install -e ".[all]"        # core + pandas + pyarrow + optuna

# Run the smoke backtest
combo-futures backtest --config config.example.json
```

Python 3.11+ and a recent Rust toolchain (1.75+) are required.

---

## Features

| Capability                          | combo-futures | Passivbot | Freqtrade |
|-------------------------------------|:-------------:|:---------:|:---------:|
| Rust performance core               |       ✓       |     ✓     |           |
| Grid trading (initial + DCA)        |       ✓       |     ✓     |           |
| Trailing entries / closes           |       ✓       |     ✓     |           |
| Unstucking + auto-deleveraging      |       ✓       |     ✓     |           |
| TWEL / WEL / HSL enforcers          |       ✓       |     ✓     |           |
| Forager symbol selection            |       ✓       |     ✓     |           |
| Trend / regime signals (RSI/MACD/ADX/BB) |   ✓       |           |     ✓     |
| `IStrategy` callbacks (17 hooks)    |       ✓       |           |     ✓     |
| Multi-symbol portfolio backtest     |       ✓       |     ✓     |     ✓     |
| Optuna NSGA-II optimization         |       ✓       |     ✓     |     ✓     |
| ML pipeline (FreqAI-style)          |   planned     |           |     ✓     |
| Native multi-exchange support       |   Binance only|   many    |    many   |

**Acronyms:** WEL = Wallet Exposure Limit. TWEL = Total WEL across positions. HSL = (Equity) Hard Stop Loss.

---

## Architecture

```
                ┌──────────────────────────────────┐
                │       Python User Layer          │
                │  IStrategy + 17 callbacks        │
                │  Optuna optimizer                │
                │  Live trader (ccxt)              │
                └────────────┬─────────────────────┘
                             │ numpy arrays + dicts
                             ▼
                ┌──────────────────────────────────┐
                │       Rust Core (PyO3)           │
                │  ┌────────┐ ┌──────┐ ┌────────┐  │
                │  │entries │ │closes│ │trailing│  │
                │  └────────┘ └──────┘ └────────┘  │
                │  ┌────────┐ ┌──────┐ ┌────────┐  │
                │  │  risk  │ │ ema  │ │unstuck │  │
                │  └────────┘ └──────┘ └────────┘  │
                │  ┌──────────────────────────────┐│
                │  │   orchestrator (per-tick)    ││
                │  └──────────────────────────────┘│
                │  ┌──────────────────────────────┐│
                │  │  backtest / multi_symbol     ││
                │  └──────────────────────────────┘│
                └──────────────────────────────────┘
```

**Layer responsibilities:**

- **Python user layer** — strategy authoring, optimization driving, live execution, data wrangling. This is where you write code.
- **Rust core (PyO3)** — every per-tick decision: entry/close grids, trailing logic, unstucking, EMA/volatility tracking, risk enforcement, multi-symbol coordination. Hot loop only — no I/O.
- **Boundary** — numpy `[N, 5]` OHLCV arrays in, dicts of fills/equity/positions out. The GIL is released during the inner loop so multiple optimizer workers can run in parallel.

Crate layout (`rust/src/`):

```
lib.rs            PyO3 bindings + dict ↔ struct conversion
types.rs          BotParams, ExchangeParams, OrderBook, Position
entries.rs        Grid + trailing entry pricing & sizing
closes.rs         Grid + trailing close, partial fills
trailing.rs       Reusable trailing helpers
ema.rs            Double-EMA bands + volatility EMA
risk.rs           WEL / TWEL / HSL enforcers, unstucking
orchestrator.rs   Per-bar decision loop
backtest.rs       Single-symbol event-driven backtest
multi_symbol.rs   Portfolio backtest with Forager selection
utils.rs          Quantization, helpers
```

68 Rust tests + 81 Python tests = **149 total, all passing**.

---

## Usage

### Single-symbol backtest

```python
from combo_bot.data import load_cached_data
from combo_bot.grid_engine import GridConfig
from combo_bot.rust_backtest import RustBacktestConfig, run_rust_backtest

candles = load_cached_data("BTC/USDT:USDT", "1h", "data")

grid = GridConfig(
    entry_initial_ema_dist=0.008,
    entry_initial_qty_pct=0.012,
    entry_grid_spacing_pct=0.025,
    entry_grid_double_down_factor=1.3,
    close_grid_markup_start=0.005,
    close_grid_markup_end=0.015,
    close_grid_qty_pct=0.5,
    wallet_exposure_limit=1.0,
    ema_span_0=385.0,
    ema_span_1=620.0,
    max_grid_levels=10,
)

result = run_rust_backtest(
    candles,
    grid,
    bt_config=RustBacktestConfig(
        starting_balance=10_000.0,
        funding_rate=0.0001,
        funding_interval_bars=480,
        liquidation_threshold_pct=0.05,
    ),
)

print(f"Final equity:  ${result.final_equity:,.2f}")
print(f"Total return:  {result.total_return:.2%}")
print(f"Max drawdown:  {result.max_drawdown:.2%}")
print(f"Sortino:       {result.sortino_ratio:.2f}")
print(f"Calmar:        {result.calmar_ratio:.2f}")
print(f"Trades:        {result.n_trades}")
```

### Multi-symbol portfolio backtest

```python
from combo_bot.rust_multi_symbol import MultiSymbolConfig, run_multi_symbol_backtest

candle_data = {
    "BTC/USDT:USDT": load_cached_data("BTC/USDT:USDT", "1h", "data"),
    "ETH/USDT:USDT": load_cached_data("ETH/USDT:USDT", "1h", "data"),
    "SOL/USDT:USDT": load_cached_data("SOL/USDT:USDT", "1h", "data"),
}

# Same grid config used for every symbol (or pass per-symbol configs)
grid = GridConfig(wallet_exposure_limit=0.8, max_grid_levels=8)
grid_configs = {s: grid for s in candle_data}

result = run_multi_symbol_backtest(
    candle_data,
    grid_configs,
    bt_config=MultiSymbolConfig(
        starting_balance=10_000.0,
        n_positions_max=3,
        forager_volume_weight=0.23,
        forager_volatility_weight=0.71,
        forager_ema_readiness_weight=0.06,
    ),
)
print(f"Symbols traded: {result.symbols}")
print(f"Trades:         {result.n_trades}")
print(f"Sharpe:         {result.sharpe_ratio:.2f}")
```

### Optuna optimization

```python
from combo_bot.rust_optimize import OptimizeBounds, RustOptimizeConfig, RustOptimizer

opt = RustOptimizer(
    candles=candles,
    config=RustOptimizeConfig(
        n_trials=500,
        n_jobs=4,
        walk_forward_splits=3,
        train_ratio=0.7,
        sortino_weight=0.35,
        calmar_weight=0.25,
        return_weight=0.25,
        drawdown_weight=0.15,
        bounds=OptimizeBounds(
            wallet_exposure_limit=(0.3, 2.0),
            entry_grid_double_down_factor=(1.0, 1.8),
        ),
    ),
)
study = opt.run()
print("Best score: ", study["best_score"])
print("Best params:", study["best_params"])
```

Multi-objective scoring combines Sortino, Calmar, total return, and drawdown penalty. Liquidated trials and trials with fewer than `min_trades` are penalized with `-1e6`.

### Custom strategy via `IStrategy` callbacks

```python
from combo_bot.strategy import IStrategy, TradeContext
from combo_bot.types import Side, TrendRegime

class MyStrategy(IStrategy):
    timeframe = "1h"
    stoploss = -0.10
    startup_candle_count = 30

    def populate_indicators(self, df, metadata):
        df["ema_fast"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=26, adjust=False).mean()
        return df

    def populate_entry_trend(self, df, metadata):
        cond = df["ema_fast"] > df["ema_slow"]
        df.loc[cond, "enter_long"] = 1
        df.loc[cond, "enter_tag"] = "ema_crossover"
        return df

    def populate_exit_trend(self, df, metadata):
        df.loc[df["ema_fast"] < df["ema_slow"], "exit_long"] = 1
        return df

    # Veto entries during strong counter-trend regimes
    def confirm_trade_entry(self, ctx: TradeContext, qty, price):
        if ctx.signal and ctx.signal.regime == TrendRegime.STRONG_BEAR and ctx.side == Side.LONG:
            return False
        return True

    # 5% trailing stop once in profit
    def custom_stoploss(self, ctx: TradeContext, current_profit_pct):
        if current_profit_pct <= 0 or not ctx.position.is_open:
            return None
        if ctx.side == Side.LONG:
            return ctx.current_price * 0.95
        return ctx.current_price * 1.05
```

All 17 hooks are documented in [`combo_bot/strategy.py`](combo_bot/strategy.py):

`bot_start`, `bot_loop_start`, `informative_pairs`, `confirm_trade_entry`, `confirm_trade_exit`, `custom_entry_price`, `custom_exit_price`, `custom_stake_amount`, `custom_stoploss`, `custom_exit`, `adjust_trade_position`, `adjust_entry_price`, `adjust_exit_price`, `leverage`, `check_entry_timeout`, `check_exit_timeout`, plus the three required `populate_*` methods.

### Live trading (Binance USDM perp)

```bash
# Dry run first — no real orders are sent
combo-futures live --config config.json

# Set the live flag in config.json only after dry-run looks correct
```

Live trading uses `ccxt` and reads API keys from environment variables. **Use a trade-only API key** — never grant withdrawal permission.

```bash
export BINANCE_API_KEY=...           # trade-only
export BINANCE_API_SECRET=...
```

---

## Performance

Measured on a single Apple Silicon core, release build.

| Benchmark | Throughput | Wall time |
|---|---|---|
| Rust single-symbol backtest         | **1.6M candles/sec** | — |
| Python reference backtest           | ~7K candles/sec      | — |
| **Speedup vs Python (20K candles)** | **208x**             | — |
| Multi-symbol backtest (3 × 30K)     | —                    | **107 ms** |
| 500K Optuna trials (single core)    | —                    | **~79 min** |

The Rust core releases the GIL during the inner loop, so `n_jobs > 1` in the optimizer scales near-linearly until you saturate cores.

> Reproduce with the scripts in `tests/test_rust_backtest.py` and `tests/test_multi_symbol.py`. Numbers vary with hardware, candle count, and config — treat them as ballpark, not contract.

---

## Configuration

See [`config.example.json`](config.example.json) for the full schema. Key sections:

| Section   | Purpose | Notable keys |
|-----------|---------|--------------|
| top-level | Universe & exchange basics | `symbols`, `starting_balance`, `leverage`, `maker_fee`, `taker_fee`, `data_dir` |
| `grid`    | Passivbot-style grid params | `entry_initial_ema_dist`, `entry_grid_spacing_pct`, `entry_grid_double_down_factor`, `close_grid_markup_start/end`, `wallet_exposure_limit`, `total_wallet_exposure_limit`, `n_positions`, `max_grid_levels`, `ema_span_0`, `ema_span_1` |
| `trend`   | Trend / regime signals     | `rsi_period`, `macd_fast/slow/signal`, `bb_period`, `bb_std`, `ema_fast/slow`, `strong_threshold`, `weak_threshold` |
| `merger`  | Grid+trend fusion          | `grid_depth_limit_in_downtrend`, `trend_position_max_pct`, `trend_entry_qty_pct`, `mode_switch_*_threshold`, `trend_stop_loss_pct`, `trend_take_profit_pct` |
| `risk`    | Drawdown guard             | `max_drawdown_pct`, `yellow_threshold`, `orange_threshold`, `red_threshold`, `max_total_wallet_exposure`, `max_single_exposure`, `liquidation_threshold`, `cooldown_after_red_minutes` |

The risk manager uses four bands — green / yellow / orange / red — that progressively tighten exposure and switch the engine into `TP_ONLY` → `GRACEFUL_STOP` → `PANIC` mode.

---

## Risk warnings

Read every line.

- **This bot has NOT been battle-tested on real money in production.** Treat it as alpha-quality research code.
- **Backtest results are NOT predictive of live performance.** Slippage, funding rate dynamics, exchange downtime, partial fills, and fee tier changes all matter and are only approximated here.
- **Grid trading carries risk of large drawdowns in trending markets.** A grid that DCAs into a falling asset can chew through your wallet exposure limit fast. Liquidation is a real outcome.
- **Leveraged futures can lose more than your deposit on most exchanges** if liquidation slippage is bad. Combo-futures backtests model a 5% liquidation buffer by default; reality is messier.
- **Always start on testnet.** Then with **small** real capital you can afford to lose entirely. Then, maybe, scale.
- **Never use money you can't afford to lose.** Crypto. Leverage. Bots. Three multipliers on risk.
- **API keys must be trade-only.** Disable withdrawal permission. Restrict by IP. Use a dedicated subaccount.
- **You are responsible for your own losses.** The MIT license disclaims warranty and liability — see [LICENSE](LICENSE).

---

## Project status

**Alpha.** What works today:

- Rust core: grid, trailing, EMA bands, volatility, unstucking, WEL/TWEL/HSL, Forager, single + multi-symbol backtests — covered by 68 Rust tests.
- Python interface: data loader, backtester, optimizer, merger, risk manager, `IStrategy` runner, live trader scaffolding — covered by 81 Python tests.
- Optuna optimization with walk-forward windows and TPE sampler.
- Binance USDM perpetual via `ccxt`.

What is not done:

- No production track record. Treat numbers from this README as **upper bounds** in friendly conditions.
- No FreqAI-style ML pipeline yet.
- Only Binance USDM is wired to live trading. Multi-exchange is on the roadmap.
- No GUI / dashboard. CLI + Python only.
- CI runs tests but does not yet publish wheels.

---

## Contributing

Issues and pull requests welcome. A `CONTRIBUTING.md` is planned — for now:

1. Open an issue describing the change before large PRs.
2. Run `cargo fmt && cargo clippy --all-targets -- -D warnings` for Rust changes.
3. Run `pytest tests/` for Python changes; new logic needs new tests.
4. Keep `combo_bot/` and `rust/src/` consistent — Python schemas and Rust struct fields must match.

---

## License

[MIT](LICENSE). Use it, fork it, ship it, sell it — at your own risk.

---

## Credits

Architectural inspiration from:

- [**Passivbot**](https://github.com/enarjord/passivbot) (Unlicense) — grid trading, EMA bands, WEL concept, Forager, HSL, NSGA-II optimization.
- [**Freqtrade**](https://github.com/freqtrade/freqtrade) (GPL-3.0) — `IStrategy` interface, walk-forward validation, FreqAI inspiration.

No source code was copied from either project. See [CREDITS.md](CREDITS.md) for the full attribution.

---

## Roadmap

Near term:

- `CONTRIBUTING.md` + issue templates polish
- Wheels published from CI for Linux / macOS / Windows
- More live-trading integration tests against Binance testnet
- Telemetry / dashboard for live runs

Mid term:

- FreqAI-style ML feature engineering + prediction pipeline
- Additional exchanges via `ccxt` (Bybit, OKX, Hyperliquid)
- Walk-forward visualization tools
- Risk-of-ruin and Monte Carlo equity curve analysis

Long term:

- Multi-timeframe strategy support
- Order book imbalance and microstructure features
- Sub-second polling loop for higher-frequency strategies
