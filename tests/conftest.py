from __future__ import annotations
import math
import numpy as np
import pytest
from combo_bot.types import (
    AccountState, Candle, EMAState, ExchangeParams,
    Position, Side, SymbolState, VolatilityState,
)


@pytest.fixture
def exchange_params() -> ExchangeParams:
    return ExchangeParams(
        qty_step=0.001,
        price_step=0.01,
        min_qty=0.001,
        min_cost=5.0,
        c_mult=1.0,
        maker_fee=0.0002,
        taker_fee=0.0005,
    )


@pytest.fixture
def ema_state() -> EMAState:
    ema = EMAState()
    ema.init([385.0, 620.0], 50000.0)
    for p in [50100, 49900, 50050, 49950, 50000]:
        ema.update(float(p))
    return ema


@pytest.fixture
def volatility_state() -> VolatilityState:
    vol = VolatilityState()
    vol.init(1000.0, 0.001)
    return vol


@pytest.fixture
def account_state() -> AccountState:
    acc = AccountState(balance=10000.0, equity=10000.0, equity_peak=10000.0)
    acc.symbols["BTC/USDT:USDT"] = SymbolState(
        symbol="BTC/USDT:USDT", last_price=50000.0,
    )
    acc.symbols["BTC/USDT:USDT"].ema.init([385.0, 620.0], 50000.0)
    acc.symbols["BTC/USDT:USDT"].volatility.init(1000.0, 0.001)
    return acc


def make_candles(prices: list[float], start_ts: int = 1700000000000) -> list[Candle]:
    candles = []
    for i, p in enumerate(prices):
        spread = abs(p * 0.001)
        candles.append(Candle(
            timestamp=start_ts + i * 60000,
            open=p,
            high=p + spread,
            low=p - spread,
            close=p,
            volume=100.0,
        ))
    return candles


def make_oscillating_candles(n: int = 5000, base: float = 50000.0, amplitude: float = 500.0) -> list[Candle]:
    t = np.arange(n)
    prices = base + amplitude * np.sin(t / 100 * 2 * np.pi)
    noise = np.random.default_rng(42).normal(0, base * 0.0002, n)
    prices = prices + noise
    return make_candles(prices.tolist())
