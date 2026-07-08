# =============================================================================
# triple_ema_vwap.py - Triple EMA Cross + VWAP Exit + Trailing Stop
#                      + VWAP Touch Re-entry + Momentum Capture
# =============================================================================

import pandas as pd

from strategies.base_strategy import BaseStrategy


class TripleEMAVWAPStrategy(BaseStrategy):
    """
    Triple EMA Cross + VWAP Band Exit + Trailing Stop + Re-entry Strategy.

    ENTRY:
        BUY  : Fast EMA > Mid EMA > Slow EMA (first candle of bullish alignment)
        SELL : Fast EMA < Mid EMA < Slow EMA (first candle of bearish alignment)

    EXIT PRIORITY (checked in order):
        1. VWAP Band Hit    : Close >= VWAP*(1+band%) for LONG
                              Close <= VWAP*(1-band%) for SHORT
        2. Trailing Stop    : Trails trail_pct% below highest close (LONG)
                              Trails trail_pct% above lowest close (SHORT)
        3. Take Profit      : Fixed take_profit_pct% from entry

    RE-ENTRY LOGIC (after VWAP exit or trailing stop exit):
        Rule 1 - After VWAP Exit:
            If EMA still aligned AND price closes below VWAP then back above
            within vwap_touch_window candles --> Re-enter LONG
            (reverse for SHORT)

        Rule 2 - After Trailing Stop Exit:
            Same VWAP touch confirmation required before re-entry

        Rule 3 - Momentum Reset While In Position:
            If price closes below VWAP then back above while in LONG
            AND EMA still aligned --> Reset trailing stop peak to current close
            (captures fresh momentum leg)

    RE-ENTRY PARAMETERS:
        vwap_touch_window : Max candles after exit to wait for VWAP touch (10)
        reentry_cooldown  : Min candles between re-entries (3)
        max_reentries     : Max re-entries per EMA alignment window (3)

    VWAP TOUCH DEFINITION (Option B - confirmed close):
        LONG  : Candle closes below VWAP, then next candle closes back above VWAP
        SHORT : Candle closes above VWAP, then next candle closes back below VWAP
    """

    def __init__(
        self,
        fast: int = 9,
        mid: int = 25,
        slow: int = 50,
        vwap_period: int = 50,
        vwap_band_pct: float = 1.3,
        trail_pct: float = 1.1,
        take_profit_pct: float = 2.0,
        vwap_touch_window: int = 10,
        reentry_cooldown: int = 3,
        max_reentries: int = 3,
        **kwargs,
    ):
        self.fast = fast
        self.mid = mid
        self.slow = slow
        self.vwap_period = vwap_period
        self.vwap_band_pct = vwap_band_pct / 100.0
        self.trail_pct = trail_pct / 100.0
        self.take_profit_pct = take_profit_pct / 100.0
        self.vwap_touch_window = vwap_touch_window
        self.reentry_cooldown = reentry_cooldown
        self.max_reentries = max_reentries

        # State tracking (prev candle values for crossover/touch detection)
        self._prev_bull = False
        self._prev_bear = False
        self._prev_close = None
        self._prev_vwap = None

    # @property
    def name(self) -> str:
        return "Triple EMA + VWAP"

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Triple EMA
        df["ema_fast"] = df["close"].ewm(span=self.fast, adjust=False).mean()
        df["ema_mid"] = df["close"].ewm(span=self.mid, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.slow, adjust=False).mean()

        # Rolling VWAP
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["tp_vol"] = df["typical_price"] * df["volume"]
        df["vwap"] = (
            df["tp_vol"].rolling(window=self.vwap_period).sum()
            / df["volume"].rolling(window=self.vwap_period).sum()
        )

        # VWAP Bands
        df["vwap_upper"] = df["vwap"] * (1 + self.vwap_band_pct)
        df["vwap_lower"] = df["vwap"] * (1 - self.vwap_band_pct)

        # EMA alignment flags (precomputed for efficiency)
        df["ema_bull"] = (df["ema_fast"] > df["ema_mid"]) & (
            df["ema_mid"] > df["ema_slow"]
        )
        df["ema_bear"] = (df["ema_fast"] < df["ema_mid"]) & (
            df["ema_mid"] < df["ema_slow"]
        )

        return df

    def get_signal(self, candle_dict: dict, indicators_dict: dict):
        """
        Entry signal on first candle of EMA alignment.
        Re-entry signals are handled separately in get_reentry_signal().
        NOTE: _update_prev_state is called by the backtester at end of
        each candle, not here.
        """
        ema_bull = indicators_dict.get("ema_bull")
        ema_bear = indicators_dict.get("ema_bear")

        # Wait until EMA alignment flags are available
        if ema_bull is None or ema_bear is None:
            return None

        signal = None

        if ema_bull and not self._prev_bull:
            signal = "BUY"
        elif ema_bear and not self._prev_bear:
            signal = "SELL"

        return signal

    def get_exit_signal(
        self, candle_dict: dict, indicators_dict: dict, position: dict
    ) -> tuple:
        """
        Check exit conditions for an open position.

        Also handles Rule 3: Reset trailing peak on VWAP touch
        while in position (momentum capture).

        Args:
            candle_dict      : Current candle OHLCV as dict
            indicators_dict  : Current candle indicators as dict
            position         : {
                "side"        : "buy" or "sell",
                "entry_price" : float,
                "peak_price"  : float
            }

        Returns:
            (exit_price, exit_reason) or (None, None) if no exit.
            Also mutates position["peak_price"] for trailing stop tracking.
        """
        close = candle_dict.get("close")
        high = candle_dict.get("high")
        low = candle_dict.get("low")

        vwap = indicators_dict.get("vwap")
        vwap_upper = indicators_dict.get("vwap_upper")
        vwap_lower = indicators_dict.get("vwap_lower")

        # Cannot evaluate exit without required values
        if any(v is None for v in [close, high, low, vwap, vwap_upper, vwap_lower]):
            return None, None

        side = position["side"]
        entry = position["entry_price"]
        peak = position.get("peak_price", entry)

        if side == "buy":
            # --- Rule 3: Momentum Reset on VWAP Touch While In Position ---
            # Previous close was below VWAP, current close is back above VWAP
            if self._prev_close is not None and self._prev_vwap is not None:
                if self._prev_close < self._prev_vwap and close > vwap:
                    position["peak_price"] = close
                    peak = close

            # Update peak (highest close reached)
            peak = max(peak, close)
            position["peak_price"] = peak

            tp_price = round(entry * (1 + self.take_profit_pct), 1)
            trail_price = round(peak * (1 - self.trail_pct), 1)

            # Priority 1: VWAP upper band hit
            if close >= vwap_upper:
                return round(close, 1), "vwap_upper"

            # Priority 2: Trailing stop hit
            if low <= trail_price:
                return trail_price, "trailing_stop"

            # Priority 3: Take profit hit
            if high >= tp_price:
                return tp_price, "take_profit"

        else:  # SHORT
            # --- Rule 3: Momentum Reset on VWAP Touch While In Position ---
            if self._prev_close is not None and self._prev_vwap is not None:
                if self._prev_close > self._prev_vwap and close < vwap:
                    position["peak_price"] = close
                    peak = close

            # Update peak (lowest close reached)
            peak = min(peak, close)
            position["peak_price"] = peak

            tp_price = round(entry * (1 - self.take_profit_pct), 1)
            trail_price = round(peak * (1 + self.trail_pct), 1)

            # Priority 1: VWAP lower band hit
            if close <= vwap_lower:
                return round(close, 1), "vwap_lower"

            # Priority 2: Trailing stop hit
            if high >= trail_price:
                return trail_price, "trailing_stop"

            # Priority 3: Take profit hit
            if low <= tp_price:
                return tp_price, "take_profit"

        return None, None

    def get_reentry_signal(
        self,
        candle_dict: dict,
        indicators_dict: dict,
        last_exit_reason: str,
        last_exit_side: str,
        candles_since_exit: int,
        reentry_count: int,
    ) -> str:
        """
        Check if re-entry conditions are met after an exit.

        Re-entry Rules:
            Rule 1 (after vwap_upper/vwap_lower exit):
                EMA still aligned + VWAP touch confirmed (Option B pattern)
            Rule 2 (after trailing_stop exit):
                Same as Rule 1
            Both rules require:
                - Within vwap_touch_window candles of exit
                - Cooldown candles have passed
                - Max re-entries not exceeded

        Args:
            candle_dict         : Current candle OHLCV as dict
            indicators_dict     : Current candle indicators as dict
            last_exit_reason    : "vwap_upper", "vwap_lower", "trailing_stop"
            last_exit_side      : "buy" or "sell" (direction of closed trade)
            candles_since_exit  : How many candles since last exit
            reentry_count       : How many re-entries already taken

        Returns:
            "BUY", "SELL", or None
        """
        # Gate checks
        if reentry_count >= self.max_reentries:
            return None
        if candles_since_exit < self.reentry_cooldown:
            return None
        if candles_since_exit > self.vwap_touch_window:
            return None

        # Only re-enter after VWAP band or trailing stop exits
        valid_exits = {"vwap_upper", "vwap_lower", "trailing_stop"}
        if last_exit_reason not in valid_exits:
            return None

        curr_bull = indicators_dict.get("ema_bull")
        curr_bear = indicators_dict.get("ema_bear")
        curr_close = candle_dict.get("close")
        curr_vwap = indicators_dict.get("vwap")

        if any(v is None for v in [curr_bull, curr_bear, curr_close, curr_vwap]):
            return None

        # Previous close/vwap tracked via instance state
        if self._prev_close is None or self._prev_vwap is None:
            return None

        # VWAP touch confirmation (Option B):
        # Previous close was below VWAP, current close is back above VWAP
        vwap_touch_bull = (self._prev_close < self._prev_vwap) and (
            curr_close > curr_vwap
        )

        # Previous close was above VWAP, current close is back below VWAP
        vwap_touch_bear = (self._prev_close > self._prev_vwap) and (
            curr_close < curr_vwap
        )

        # Re-entry LONG: was in long, EMA still bullish, VWAP touch confirmed
        if last_exit_side == "buy" and curr_bull and vwap_touch_bull:
            return "BUY"

        # Re-entry SHORT: was in short, EMA still bearish, VWAP touch confirmed
        if last_exit_side == "sell" and curr_bear and vwap_touch_bear:
            return "SELL"

        return None

    def _update_prev_state(self, indicators_dict: dict) -> None:
        """
        Called by the backtester at the end of every candle to keep
        _prev_close, _prev_vwap, _prev_bull, _prev_bear in sync.

        Must be called on EVERY candle including candles where a
        position is open, so that VWAP touch detection is accurate.
        """
        self._prev_bull = bool(indicators_dict.get("ema_bull") or False)
        self._prev_bear = bool(indicators_dict.get("ema_bear") or False)
        self._prev_close = indicators_dict.get("close")
        self._prev_vwap = indicators_dict.get("vwap")

    def get_params(self) -> dict:
        return {
            "fast": self.fast,
            "mid": self.mid,
            "slow": self.slow,
            "vwap_period": self.vwap_period,
            "vwap_band_pct": round(self.vwap_band_pct * 100, 2),
            "trail_pct": round(self.trail_pct * 100, 2),
            "take_profit_pct": round(self.take_profit_pct * 100, 2),
            "vwap_touch_window": self.vwap_touch_window,
            "reentry_cooldown": self.reentry_cooldown,
            "max_reentries": self.max_reentries,
        }

    def get_min_candles(self) -> int:
        return self.slow + self.vwap_period + 5
