from __future__ import annotations


from combo_bot.data_provider import DataProvider
from combo_bot.types import Candle


def _candle(ts: int, close: float) -> Candle:
    return Candle(
        timestamp=ts,
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=10.0,
    )


class TestDataProvider:
    def test_empty_dataframe_has_expected_columns(self):
        dp = DataProvider()
        df = dp.get_dataframe("BTC")
        assert list(df.columns) == [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        assert len(df) == 0

    def test_append_grows_buffer(self):
        dp = DataProvider()
        dp.append("BTC", _candle(1_000, 100.0))
        dp.append("BTC", _candle(2_000, 101.0))
        df = dp.get_dataframe("BTC")
        assert len(df) == 2
        assert df["close"].tolist() == [100.0, 101.0]

    def test_dataframe_cached_until_append(self):
        dp = DataProvider()
        dp.append("BTC", _candle(1_000, 100.0))
        df_a = dp.get_dataframe("BTC")
        df_b = dp.get_dataframe("BTC")
        assert df_a is df_b
        dp.append("BTC", _candle(2_000, 101.0))
        df_c = dp.get_dataframe("BTC")
        assert df_c is not df_a

    def test_max_rows_slides_buffer(self):
        dp = DataProvider(max_rows=3)
        for i in range(5):
            dp.append("BTC", _candle(1_000 + i * 60_000, 100.0 + i))
        df = dp.get_dataframe("BTC")
        assert len(df) == 3
        assert df["close"].tolist() == [102.0, 103.0, 104.0]

    def test_symbols_are_isolated(self):
        dp = DataProvider()
        dp.append("BTC", _candle(1_000, 100.0))
        dp.append("ETH", _candle(1_000, 50.0))
        assert dp.get_dataframe("BTC")["close"].tolist() == [100.0]
        assert dp.get_dataframe("ETH")["close"].tolist() == [50.0]

    def test_reset_clears_symbol(self):
        dp = DataProvider()
        dp.append("BTC", _candle(1_000, 100.0))
        dp.append("ETH", _candle(1_000, 50.0))
        dp.reset("BTC")
        assert not dp.has_symbol("BTC")
        assert dp.has_symbol("ETH")

    def test_reset_all(self):
        dp = DataProvider()
        dp.append("BTC", _candle(1_000, 100.0))
        dp.append("ETH", _candle(1_000, 50.0))
        dp.reset()
        assert not dp.has_symbol("BTC")
        assert not dp.has_symbol("ETH")
