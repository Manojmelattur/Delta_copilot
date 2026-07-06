# =============================================================================
# triple_ema.py - Triple EMA Strategy
# =============================================================================

import pandas as pd

from strategies.base_strategy import BaseStrategy


class TripleEMAStrategy(BaseStrategy):
    """
    Triple EMA Strategy.

    Uses three EMAs: fast, mid, slow.

    BUY  : Fast > Mid > Slow (full bullish alignment)
           AND previous candle was NOT fully aligned bullish
    SELL : Fast < Mid < Slow (full bearish alignment)
           AND previous candle was NOT fully aligned bearish

    This ensures we only trigger on the FIRST candle of alignment,
    not on every candle while aligned (avoids repeated signals).
    """

    def __init__(self, fast: int = 5, mid: int = 13, slow: int = 21, **kwargs):
        self.fast = fast
        self.mid = mid
        self.slow = slow
        self._prev_bull = False
        self._prev_bear = False

    @property
    def name(self) -> str:
        return "Triple EMA"

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema_fast"] = df["close"].ewm(span=self.fast, adjust=False).mean()
        df["ema_mid"] = df["close"].ewm(span=self.mid, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.slow, adjust=False).mean()
        return df

    def get_signal(self, candle_dict: dict, indicators_dict: dict):
        ema_fast = indicators_dict.get("ema_fast")
        ema_mid = indicators_dict.get("ema_mid")
        ema_slow = indicators_dict.get("ema_slow")

        # Wait until all EMAs are available
        if ema_fast is None or ema_mid is None or ema_slow is None:
            return None

        curr_bull = ema_fast > ema_mid > ema_slow
        curr_bear = ema_fast < ema_mid < ema_slow

        signal = None

        # Trigger only on first candle of alignment
        if curr_bull and not self._prev_bull:
            signal = "BUY"
        elif curr_bear and not self._prev_bear:
            signal = "SELL"

        # Update previous state for next candle
        self._prev_bull = curr_bull
        self._prev_bear = curr_bear

        return signal

    def get_params(self) -> dict:
        return {"fast": self.fast, "mid": self.mid, "slow": self.slow}

    def get_min_candles(self) -> int:
        return self.slow + 5
