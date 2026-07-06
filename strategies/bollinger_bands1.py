# =============================================================================
# bollinger_bands.py - Bollinger Bands Breakout Strategy with Fixed TP + Trailing Stop
# =============================================================================

import pandas as pd

from strategies.base_strategy import BaseStrategy


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Breakout Strategy with Fixed TP + Trailing Stop.

    BUY  : Close price breaks ABOVE upper band (bullish breakout)
    SELL : Close price breaks BELOW lower band (bearish breakout)

    Exit Logic:
    - Fixed Take Profit  : Exit when price moves take_profit_pct in favor
    - Trailing Stop      : Activates after trail_activation_pct move in favor
                           Trails by trail_pct from the highest/lowest price seen
    - Stop Loss          : Hard stop at stop_loss_pct against entry

    Uses 2-candle confirmation to reduce false breakout signals.
    """

    def __init__(
        self,
        period: int = 20,
        std: float = 2.0,
        take_profit_pct: float = 2.0,
        trail_activation_pct: float = 1.0,
        trail_pct: float = 0.8,
        stop_loss_pct: float = 1.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.period = period
        self.std = std
        self.take_profit_pct = take_profit_pct / 100
        self.trail_activation_pct = trail_activation_pct / 100
        self.trail_pct = trail_pct / 100
        self.stop_loss_pct = stop_loss_pct / 100

        # Trade state tracking
        self._reset_trade_state()

    def _reset_trade_state(self):
        """Reset all trade tracking variables."""
        self.position = None  # 'BUY' or 'SELL'
        self.entry_price = None
        self.best_price = None  # Highest price seen for BUY / Lowest for SELL
        self.trailing_active = False
        self.trail_stop = None

    @property
    def name(self) -> str:
        return "Bollinger Bands"

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["bb_mid"] = df["close"].rolling(window=self.period).mean()
        df["bb_std"] = df["close"].rolling(window=self.period).std()
        df["bb_upper"] = df["bb_mid"] + (self.std * df["bb_std"])
        df["bb_lower"] = df["bb_mid"] - (self.std * df["bb_std"])
        return df

    def get_signal(self, df: pd.DataFrame, index: int):
        if index < 2:
            return None

        prev_close = df["close"].iloc[index - 2]
        curr_close = df["close"].iloc[index - 1]
        prev_upper = df["bb_upper"].iloc[index - 2]
        curr_upper = df["bb_upper"].iloc[index - 1]
        prev_lower = df["bb_lower"].iloc[index - 2]  # FIX: was incorrectly index-1
        curr_lower = df["bb_lower"].iloc[index - 1]

        # --- Exit Logic (if in a position) ---
        if self.position is not None:
            exit_signal = self._check_exit(curr_close)
            if exit_signal:
                return exit_signal

        # --- Entry Logic (only if no open position) ---
        if self.position is None:
            # 2-candle confirmation: prev candle was inside band, curr candle breaks out
            # Breakout BUY: previous close was below upper, current close breaks above
            if prev_close <= prev_upper and curr_close > curr_upper:
                self._open_position("BUY", curr_close)
                return "BUY"

            # Breakout SELL: previous close was above lower, current close breaks below
            if prev_close >= prev_lower and curr_close < curr_lower:
                self._open_position("SELL", curr_close)
                return "SELL"

        return None

    def _open_position(self, side: str, entry_price: float):
        """Initialize trade state on new entry."""
        self.position = side
        self.entry_price = entry_price
        self.best_price = entry_price
        self.trailing_active = False
        self.trail_stop = None

    def _check_exit(self, curr_close: float) -> str | None:
        """
        Check exit conditions in priority order:
        1. Fixed Take Profit
        2. Hard Stop Loss
        3. Trailing Stop (once activated)
        """
        entry = self.entry_price

        if self.position == "BUY":
            # Update best price seen
            if curr_close > self.best_price:
                self.best_price = curr_close

            move_pct = (curr_close - entry) / entry

            # 1. Fixed Take Profit
            if move_pct >= self.take_profit_pct:
                self._reset_trade_state()
                return "EXIT_TP"

            # 2. Hard Stop Loss
            if move_pct <= -self.stop_loss_pct:
                self._reset_trade_state()
                return "EXIT_SL"

            # 3. Trailing Stop
            best_move_pct = (self.best_price - entry) / entry
            if best_move_pct >= self.trail_activation_pct:
                self.trailing_active = True

            if self.trailing_active:
                self.trail_stop = self.best_price * (1 - self.trail_pct)
                if curr_close <= self.trail_stop:
                    self._reset_trade_state()
                    return "EXIT_TRAIL"

        elif self.position == "SELL":
            # Update best price seen (lowest for short)
            if curr_close < self.best_price:
                self.best_price = curr_close

            move_pct = (entry - curr_close) / entry

            # 1. Fixed Take Profit
            if move_pct >= self.take_profit_pct:
                self._reset_trade_state()
                return "EXIT_TP"

            # 2. Hard Stop Loss
            if move_pct <= -self.stop_loss_pct:
                self._reset_trade_state()
                return "EXIT_SL"

            # 3. Trailing Stop
            best_move_pct = (entry - self.best_price) / entry
            if best_move_pct >= self.trail_activation_pct:
                self.trailing_active = True

            if self.trailing_active:
                self.trail_stop = self.best_price * (1 + self.trail_pct)
                if curr_close >= self.trail_stop:
                    self._reset_trade_state()
                    return "EXIT_TRAIL"

        return None

    def get_params(self) -> dict:
        return {
            "period": self.period,
            "std": self.std,
            "take_profit_pct": self.take_profit_pct * 100,
            "trail_activation_pct": self.trail_activation_pct * 100,
            "trail_pct": self.trail_pct * 100,
            "stop_loss_pct": self.stop_loss_pct * 100,
        }

    def get_min_candles(self) -> int:
        return self.period + 5
