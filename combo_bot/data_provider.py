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
        # Round-27: informative-timeframe buffers, keyed by
        # ``(pair, timeframe)``. Strategies register what they want via
        # IStrategy.informative_pairs(); LiveTrader / Backtester
        # populate these buffers; strategies read via
        # ``get_informative(pair, timeframe)``. Independent of the
        # main per-symbol buffer so the populate-* hooks against the
        # primary timeframe don't see informative-frame columns
        # mixed in.
        self._informative_buffers: dict[tuple[str, str], list[Candle]] = {}
        self._informative_cached_df: dict[tuple[str, str], "DataFrame"] = {}
        self._informative_cache_len: dict[tuple[str, str], int] = {}
        # Registered (pair, timeframe) requests from
        # strategy.informative_pairs(). Operators / live loops can
        # query this to know what to fetch from the exchange.
        self._informative_registry: set[tuple[str, str]] = set()

    def append(self, symbol: str, candle: Candle) -> None:
        buf = self._buffers.setdefault(symbol, [])
        buf.append(candle)
        if len(buf) > self._max_rows:
            del buf[: len(buf) - self._max_rows]
        self._cached_df.pop(symbol, None)
        self._cache_len.pop(symbol, None)

    def get_candles(self, symbol: str) -> list[Candle]:
        """Return the raw candle buffer (oldest→newest) for ``symbol``.

        Pandas-free accessor used by the ML signal layer, which works on
        numpy arrays. The returned list is the live buffer; callers must
        not mutate it.
        """
        return self._buffers.get(symbol, [])

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

    # ── Informative-timeframe API (Round-27) ────────────────────

    def register_informative(self, pair: str, timeframe: str) -> None:
        """Mark ``(pair, timeframe)`` as a stream the strategy wants.

        Idempotent — re-registering does nothing. LiveTrader.start()
        seeds the registry from ``strategy.informative_pairs()``;
        Backtester.run() does the same from cached data. Operators
        can also pre-register pairs manually before running.
        """
        self._informative_registry.add((pair, timeframe))

    def informative_pairs(self) -> set[tuple[str, str]]:
        """Return the registered informative streams."""
        return set(self._informative_registry)

    def append_informative(self, pair: str, timeframe: str, candle: Candle) -> None:
        """Append a candle to the ``(pair, timeframe)`` informative
        buffer. Auto-registers the pair if not already known so
        callers can append without an explicit register call."""
        key = (pair, timeframe)
        self._informative_registry.add(key)
        buf = self._informative_buffers.setdefault(key, [])
        buf.append(candle)
        if len(buf) > self._max_rows:
            del buf[: len(buf) - self._max_rows]
        self._informative_cached_df.pop(key, None)
        self._informative_cache_len.pop(key, None)

    def get_informative(self, pair: str, timeframe: str) -> "DataFrame":
        """Return the cached DataFrame for ``(pair, timeframe)``.

        Returns an empty frame (same columns as primary) when the
        stream is unregistered or empty — callers should test
        ``len(df)`` before reading the last row.
        """
        import pandas as pd

        key = (pair, timeframe)
        buf = self._informative_buffers.get(key, [])
        if not buf:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
        if self._informative_cache_len.get(key) == len(buf):
            return self._informative_cached_df[key]

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
        self._informative_cached_df[key] = df
        self._informative_cache_len[key] = len(buf)
        return df


__all__ = ["DataProvider"]
