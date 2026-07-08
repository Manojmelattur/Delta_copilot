# =============================================================================
# vwap.py - Rolling VWAP Strategy
# =============================================================================

import pandas as pd

from strategies.base_strategy import BaseStrategy


class VWAPStrategy(BaseStrategy):
    """
    Rolling VWAP Strategy.

    Uses a rolling window VWAP (continuous, no daily reset).

    BUY  : Price crosses ABOVE VWAP (prev close below, curr close above)
    SELL : Price crosses BELOW VWAP (prev close above, curr close below)

    Rolling VWAP Formula:
        vwap = sum(typical_price * volume, window) / sum(volume, window)
        typical_price = (high + low + close) / 3
    """

    def __init__(self, period: int = 50, **kwargs):
        self.period = period

    # @property
    def name(self) -> str:
        return "VWAP"

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["tp_vol"] = df["typical_price"] * df["volume"]
        df["vwap"] = (
            df["tp_vol"].rolling(window=self.period).sum()
            / df["volume"].rolling(window=self.period).sum()
        )
        return df

    def get_signal(self, candle_dict: dict, indicators_dict: dict):
        prev_close = indicators_dict.get("prev_close")
        prev_vwap = indicators_dict.get("prev_vwap")
        curr_close = candle_dict.get("close")
        curr_vwap = indicators_dict.get("vwap")

        if any(v is None for v in [prev_close, prev_vwap, curr_close, curr_vwap]):
            return None

        if pd.isna(prev_vwap) or pd.isna(curr_vwap):
            return None

        # Price crosses above VWAP
        if prev_close <= prev_vwap and curr_close > curr_vwap:
            return "buy"

        # Price crosses below VWAP
        if prev_close >= prev_vwap and curr_close < curr_vwap:
            return "sell"

        return None

    def get_params(self) -> dict:
        return {"period": self.period}

    def get_min_candles(self) -> int:
        return self.period + 5
