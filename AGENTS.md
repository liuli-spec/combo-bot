# Combo Futures

Combined grid + trend futures trading bot. Merges Passivbot-style grid logic with multi-indicator trend signals into a single decision engine, managed by layered risk controls.

## Tech Stack

- Python 3.11+
- ccxt — exchange connectivity
- numpy — numerical computation
- optuna — hyperparameter optimization (optional)

## How to Run

### Install dependencies

```bash
pip install -e .            # core only
pip install -e ".[all]"     # core + pandas, pyarrow, optuna
```

### Download historical data

```bash
combo-futures download --exchange binance --symbol BTC/USDT:USDT --timeframe 1h
```

### Backtest

```bash
combo-futures backtest --config config.json
```

### Optimize

```bash
combo-futures optimize --config config.json --trials 200
```

### Live trading

```bash
combo-futures live --config config.json
```

## Architecture

| Module | Purpose |
|--------|---------|
| `grid_engine` | Passivbot-inspired grid order placement with EMA entry bands and wallet exposure limits |
| `trend_signal` | Multi-indicator trend detection (EMA crossover, RSI, ADX, etc.) |
| `merger` | Decision fusion — combines grid and trend signals into unified order intent |
| `risk` | 4-tier drawdown guard with equity hard stop loss |
| `backtest` | Event-driven backtesting engine |
| `optimize` | NSGA-II multi-objective optimization via Optuna |
| `live` | Live execution loop with exchange connectivity via ccxt |

## Tests

```bash
python -m pytest tests/
```

## Important

This is a trading bot. Do NOT commit API keys, secrets, or credentials. Use environment variables or a local `.env` file (which is gitignored).
