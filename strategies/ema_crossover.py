# =============================================================================
# ema_crossover.py - EMA Crossover Strategy
# =============================================================================

import pandas as pd

from strategies.base_strategy import BaseStrategy


class EMACrossoverStrategy(BaseStrategy):
    """
    EMA Crossover Strategy.

    BUY  : Fast EMA crosses above Slow EMA AND close > trend EMA
    SELL : Fast EMA crosses below Slow EMA AND close < trend EMA

    Uses confirmed closed candles (current and previous) via indicators dict
    to avoid acting on incomplete candle noise.

    Parameters
    ----------
    fast : int
        Period for the fast EMA. Default 9.
    slow : int
        Period for the slow EMA. Default 21.
    trend_ema : int
        Period for the trend filter EMA. Only signals aligned with the trend
        are taken. Set to 0 to disable the filter entirely. Default 200.
    min_hold_candles : int
        Minimum candles to hold before allowing a signal_flip exit.
        Sweep showed no benefit; kept for completeness. Default 0.
    """

    def __init__(
        self,
        fast: int = 9,
        slow: int = 21,
        trend_ema: int = 200,
        min_hold_candles: int = 0,
        **kwargs,
    ):
        self.fast = fast
        self.slow = slow
        self.trend_ema = trend_ema
        self.min_hold_candles = min_hold_candles

        # Tracks the candle index at which the current position was entered.
        # Set by backtester via notify_entry(). None means no open position.
        self._entry_idx: int | None = None

    def name(self) -> str:

        return "EMA Crossover"

    def notify_entry(self, idx: int) -> None:
        """Called by the backtester when a position is opened at candle index idx."""
        self._entry_idx = idx

    def notify_exit(self) -> None:
        """Called by the backtester when a position is closed for any reason."""
        self._entry_idx = None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema_fast"] = df["close"].ewm(span=self.fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.slow, adjust=False).mean()
        if self.trend_ema > 0:
            df["ema_trend"] = df["close"].ewm(span=self.trend_ema, adjust=False).mean()
        return df

    def get_signal(self, candle: dict, indicators: dict) -> str | None:
        """
        Crossover detection using current and previous EMA values
        supplied via the indicators dict by the backtester/live loop.

        prev_ema_fast / prev_ema_slow : values from the previous closed candle
        ema_fast      / ema_slow      : values from the current closed candle
        ema_trend                     : trend filter EMA (optional)

        Signal is suppressed if:
        - Any required EMA value is None or NaN
        - Trend filter is enabled and signal is counter-trend
        - min_hold_candles > 0 and hold period has not elapsed
        """
        prev_fast = indicators.get("prev_ema_fast")
        prev_slow = indicators.get("prev_ema_slow")
        curr_fast = indicators.get("ema_fast")
        curr_slow = indicators.get("ema_slow")

        # Cannot evaluate crossover without both candles
        if any(
            v is None or pd.isna(v)
            for v in [prev_fast, prev_slow, curr_fast, curr_slow]
        ):
            return None

        # Determine raw crossover signal
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            raw_signal = "buy"
        elif prev_fast >= prev_slow and curr_fast < curr_slow:
            raw_signal = "sell"
        else:
            return None

        # --- Trend filter ---
        if self.trend_ema > 0:
            ema_trend = indicators.get("ema_trend")
            close = candle.get("close")
            if ema_trend is None or pd.isna(ema_trend) or close is None:
                return None
            if raw_signal == "buy" and close < ema_trend:
                return None
            if raw_signal == "sell" and close > ema_trend:
                return None

        # --- Min hold guard ---
        if self.min_hold_candles > 0 and self._entry_idx is not None:
            current_idx = indicators.get("current_idx")
            if current_idx is not None:
                candles_held = current_idx - self._entry_idx
                if candles_held < self.min_hold_candles:
                    return None

        return raw_signal

    def get_params(self) -> dict:
        return {
            "fast": self.fast,
            "slow": self.slow,
            "trend_ema": self.trend_ema,
            "min_hold_candles": self.min_hold_candles,
        }

    def get_min_candles(self) -> int:
        # Ensure enough candles for the longest EMA to warm up
        return max(self.slow, self.trend_ema) + 5
