"""Rolling OHLCV buffer that strategies can read as a pandas DataFrame.

Modeled on freqtrade's DataProvider: candles are appended one at a time as the
backtest or live tick advances, and ``get_dataframe(symbol)`` returns the rows
visible up to and including the most recent append (no lookahead).

The DataFrame is built lazily and cached until the buffer grows or resets, so
strategies that call ``get_dataframe`` multiple times per tick pay the
construction cost once. The buffer is capped to bound memory in long backtests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from combo_bot.types import Candle

if TYPE_CHECKING:
    from pandas import DataFrame


class DataProvider:
    def __init__(self, max_rows: int = 1000) -> None:
        self._max_rows = max_rows
        self._buffers: dict[str, list[Candle]] = {}
        self._cached_df: dict[str, "DataFrame"] = {}
        self._cache_len: dict[str, int] = {}

    def append(self, symbol: str, candle: Candle) -> None:
        buf = self._buffers.setdefault(symbol, [])
        buf.append(candle)
        if len(buf) > self._max_rows:
            del buf[: len(buf) - self._max_rows]
        self._cached_df.pop(symbol, None)
        self._cache_len.pop(symbol, None)

    def get_dataframe(self, symbol: str) -> "DataFrame":
        import pandas as pd

        buf = self._buffers.get(symbol, [])
        if not buf:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
        if self._cache_len.get(symbol) == len(buf):
            return self._cached_df[symbol]

        df = pd.DataFrame(
            {
                "timestamp": [c.timestamp for c in buf],
                "open": [c.open for c in buf],
                "high": [c.high for c in buf],
                "low": [c.low for c in buf],
                "close": [c.close for c in buf],
                "volume": [c.volume for c in buf],
            }
        )
        df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        self._cached_df[symbol] = df
        self._cache_len[symbol] = len(buf)
        return df

    def buffer_len(self, symbol: str) -> int:
        return len(self._buffers.get(symbol, []))

    def has_symbol(self, symbol: str) -> bool:
        return symbol in self._buffers

    def reset(self, symbol: str | None = None) -> None:
        if symbol is None:
            self._buffers.clear()
            self._cached_df.clear()
            self._cache_len.clear()
        else:
            self._buffers.pop(symbol, None)
            self._cached_df.pop(symbol, None)
            self._cache_len.pop(symbol, None)


__all__ = ["DataProvider"]
