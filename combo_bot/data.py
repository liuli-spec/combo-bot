from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from combo_bot.types import Candle, ExchangeParams

logger = logging.getLogger(__name__)

MS_IN_MINUTE = 60_000
MS_IN_HOUR = 3_600_000
MS_IN_DAY = 86_400_000

TIMEFRAME_MS: dict[str, int] = {
    "1m": MS_IN_MINUTE,
    "3m": 3 * MS_IN_MINUTE,
    "5m": 5 * MS_IN_MINUTE,
    "15m": 15 * MS_IN_MINUTE,
    "30m": 30 * MS_IN_MINUTE,
    "1h": MS_IN_HOUR,
    "2h": 2 * MS_IN_HOUR,
    "4h": 4 * MS_IN_HOUR,
    "6h": 6 * MS_IN_HOUR,
    "8h": 8 * MS_IN_HOUR,
    "12h": 12 * MS_IN_HOUR,
    "1d": MS_IN_DAY,
}

MAX_OHLCV_LIMIT = 1500
RATE_LIMIT_PAUSE = 0.5


def _get_ccxt_async():
    try:
        import ccxt.pro as ccxt_mod
    except ImportError:
        import ccxt.async_support as ccxt_mod
    return ccxt_mod


def create_exchange(testnet: bool = False) -> Any:
    ccxt_mod = _get_ccxt_async()
    exchange = ccxt_mod.binanceusdm(
        {
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                "sandboxMode": testnet,
            },
        }
    )
    if testnet:
        exchange.set_sandbox_mode(True)
    return exchange


def _ts_from_date(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _row_to_candle(row: list) -> Candle:
    return Candle(
        timestamp=int(row[0]),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
    )


async def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    since: int,
    limit: int,
    exchange: Any,
) -> list[Candle]:
    tf_ms = TIMEFRAME_MS.get(timeframe, MS_IN_MINUTE)
    all_candles: list[Candle] = []
    cursor = since
    remaining = limit

    while remaining > 0:
        batch_size = min(remaining, MAX_OHLCV_LIMIT)
        try:
            rows = await exchange.fetch_ohlcv(
                symbol, timeframe, since=cursor, limit=batch_size
            )
        except Exception as exc:
            logger.error("fetch_ohlcv failed for %s at %d: %s", symbol, cursor, exc)
            raise

        if not rows:
            break

        candles = [_row_to_candle(r) for r in rows]
        all_candles.extend(candles)
        remaining -= len(candles)

        last_ts = candles[-1].timestamp
        cursor = last_ts + tf_ms

        if len(rows) < batch_size:
            break

        await asyncio.sleep(RATE_LIMIT_PAUSE)

    return all_candles


async def fetch_funding_rates(
    symbol: str,
    since: int,
    exchange: Any,
) -> list[tuple[int, float]]:
    rates: list[tuple[int, float]] = []
    cursor = since
    page_limit = 1000

    while True:
        try:
            raw = await exchange.fetch_funding_rate_history(
                symbol, since=cursor, limit=page_limit
            )
        except Exception as exc:
            logger.error(
                "fetch_funding_rates failed for %s at %d: %s", symbol, cursor, exc
            )
            raise

        if not raw:
            break

        for entry in raw:
            ts = int(entry.get("timestamp", entry.get("datetime", 0)))
            rate = float(entry.get("fundingRate", 0.0))
            rates.append((ts, rate))

        if len(raw) < page_limit:
            break

        cursor = rates[-1][0] + 1
        await asyncio.sleep(RATE_LIMIT_PAUSE)

    return rates


async def fetch_exchange_params(
    symbol: str,
    exchange: Any,
) -> ExchangeParams:
    try:
        await exchange.load_markets(reload=True)
    except Exception as exc:
        logger.error("load_markets failed: %s", exc)
        raise

    market = exchange.market(symbol)
    precision = market.get("precision", {})
    limits = market.get("limits", {})
    amount_limits = limits.get("amount", {})
    cost_limits = limits.get("cost", {})
    fees = market.get("info", {})

    qty_step = float(precision.get("amount", 0.001))
    price_step = float(precision.get("price", 0.01))
    min_qty = float(amount_limits.get("min", qty_step))
    min_cost = float(cost_limits.get("min", 5.0))

    # Binance USD-M linear futures have a contract multiplier of 1.0;
    # inverse (coin-m) would differ but this module targets usdm only.
    c_mult = float(market.get("contractSize", 1.0))

    maker_fee = float(fees.get("makerCommissionRate", 0.0002))
    taker_fee = float(fees.get("takerCommissionRate", 0.0005))

    return ExchangeParams(
        qty_step=qty_step,
        price_step=price_step,
        min_qty=min_qty,
        min_cost=min_cost,
        c_mult=c_mult,
        maker_fee=maker_fee,
        taker_fee=taker_fee,
    )


def _try_import_parquet():
    try:
        import pandas as pd
        import pyarrow  # noqa: F401

        return pd
    except ImportError:
        return None


def _candles_to_records(candles: list[Candle]) -> list[dict]:
    return [
        {
            "timestamp": c.timestamp,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        }
        for c in candles
    ]


def _records_to_candles(records: list[dict]) -> list[Candle]:
    return [
        Candle(
            timestamp=int(r["timestamp"]),
            open=float(r["open"]),
            high=float(r["high"]),
            low=float(r["low"]),
            close=float(r["close"]),
            volume=float(r["volume"]),
        )
        for r in records
    ]


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def _save_candles(candles: list[Candle], path: Path) -> None:
    pd = _try_import_parquet()
    if pd is not None:
        parquet_path = path.with_suffix(".parquet")
        df = pd.DataFrame(_candles_to_records(candles))
        df.to_parquet(parquet_path, index=False)
        logger.info("Saved %d candles to %s", len(candles), parquet_path)
    else:
        json_path = path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(_candles_to_records(candles), f)
        logger.info("Saved %d candles to %s (json fallback)", len(candles), json_path)


def _load_candles(path: Path) -> list[Candle] | None:
    pd = _try_import_parquet()
    parquet_path = path.with_suffix(".parquet")
    json_path = path.with_suffix(".json")

    if pd is not None and parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        return _records_to_candles(df.to_dict("records"))

    if json_path.exists():
        with open(json_path) as f:
            records = json.load(f)
        return _records_to_candles(records)

    return None


async def download_backtest_data(
    symbols: list[str],
    start_date: str,
    end_date: str,
    timeframe: str = "1m",
    data_dir: str = "data",
) -> dict[str, list[Candle]]:
    exchange = create_exchange(testnet=False)
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    since_ms = _ts_from_date(start_date)
    until_ms = _ts_from_date(end_date)
    tf_ms = TIMEFRAME_MS.get(timeframe, MS_IN_MINUTE)
    total_bars = (until_ms - since_ms) // tf_ms

    result: dict[str, list[Candle]] = {}

    try:
        for symbol in symbols:
            safe = _safe_symbol(symbol)
            cache_name = f"{safe}_{timeframe}_{start_date}_{end_date}"
            cache_path = data_path / cache_name

            cached = _load_candles(cache_path)
            if cached is not None:
                logger.info("Loaded %d cached candles for %s", len(cached), symbol)
                result[symbol] = cached
                continue

            logger.info(
                "Downloading %s %s from %s to %s (~%d bars)",
                symbol,
                timeframe,
                start_date,
                end_date,
                total_bars,
            )

            candles = await fetch_ohlcv(
                symbol, timeframe, since_ms, total_bars, exchange
            )

            candles = [c for c in candles if c.timestamp < until_ms]

            if candles:
                _save_candles(candles, cache_path)

            result[symbol] = candles
    finally:
        await exchange.close()

    return result


def load_cached_data(
    symbols: list[str],
    data_dir: str = "data",
) -> dict[str, list[Candle]]:
    data_path = Path(data_dir)
    result: dict[str, list[Candle]] = {}

    for symbol in symbols:
        safe = _safe_symbol(symbol)
        matches: list[Path] = []

        for ext in (".parquet", ".json"):
            matches.extend(data_path.glob(f"{safe}_*{ext}"))

        if not matches:
            logger.warning("No cached data found for %s in %s", symbol, data_dir)
            continue

        matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        newest = matches[0].with_suffix("")
        candles = _load_candles(newest)

        if candles is not None:
            logger.info("Loaded %d candles for %s from cache", len(candles), symbol)
            result[symbol] = candles

    return result
