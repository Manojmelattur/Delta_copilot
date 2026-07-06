# =============================================================================
# strategies/triple_ema_vwap_macd_tsl.py
# Triple EMA + VWAP Exit + MACD + Trailing Stop Loss Strategy
#
# Entry Logic:
#   LONG  : fast_ema > slow_ema > anchor_ema AND close > vwap
#           AND macd_line > signal_line AND macd_line > 0
#           AND macd histogram accelerating (growing) AND hist > min_macd_hist
#           AND fast_ema slope rising
#           AND ema_spread_pct >= min_ema_spread_pct
#           AND atr_pct >= min_atr_pct  (volatility gate)
#           AND 2-candle confirmation (trigger + confirm, body >= 50% range,
#               confirm close > trigger close)
#
#   SHORT : fast_ema < slow_ema < anchor_ema AND close < vwap
#           AND macd_line < signal_line AND macd_line < 0
#           AND macd histogram accelerating (falling) AND hist < -min_macd_hist
#           AND fast_ema slope falling
#           AND ema_spread_pct >= min_ema_spread_pct
#           AND atr_pct >= min_atr_pct  (volatility gate)
#           AND 2-candle confirmation (trigger + confirm, body >= 50% range,
#               confirm close < trigger close)
#
# Exit Logic (priority order):
#   1. Take Profit      : price moves tp_pct% in favour
#   2. Stop Loss        : price moves sl_pct% against
#   3. Trailing Stop    : activates after trail_activation_pct%, trails by trail_pct%
#   4. VWAP Flip        : only allowed if position gain < vwap_flip_min_profit_pct
#                         (prevents premature exit on profitable trades)
#
# Notes:
#   - VWAP resets each calendar day (session-based, matches Pine Script behaviour)
#   - Designed for intraday timeframes (1m - 4H)
#   - 1 lot = 0.001 BTC on BTCUSD perpetual
# =============================================================================

from typing import Optional

import numpy as np
import pandas as pd


class TripleEmaVwapMacdTsl:
    def __init__(
        self,
        fast_ema_period: int = 9,
        slow_ema_period: int = 21,
        anchor_ema_period: int = 50,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        stop_loss_pct: float = 1.0,
        take_profit_pct: float = 2.0,
        trail_activation_pct: float = 1.0,
        trail_pct: float = 0.5,
        min_ema_spread_pct: float = 0.5,
        vwap_flip_min_profit_pct: float = 0.3,
        # --- NEW: ATR volatility gate ---
        atr_period: int = 14,
        min_atr_pct: float = 0.15,  # ATR must be >= 0.15% of price to enter
        # --- NEW: MACD histogram absolute floor ---
        min_macd_hist: float = 0.0,  # histogram absolute value floor; tune after first run
    ):
        self.fast_ema_period = fast_ema_period
        self.slow_ema_period = slow_ema_period
        self.anchor_ema_period = anchor_ema_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal_period = macd_signal
        self.stop_loss_pct = stop_loss_pct / 100.0
        self.take_profit_pct = take_profit_pct / 100.0
        self.trail_activation_pct = trail_activation_pct / 100.0
        self.trail_pct = trail_pct / 100.0
        self.min_ema_spread_pct = min_ema_spread_pct
        self.vwap_flip_min_profit_pct = vwap_flip_min_profit_pct / 100.0
        self.atr_period = atr_period
        self.min_atr_pct = min_atr_pct / 100.0
        self.min_macd_hist = min_macd_hist

        # Live position state
        self._position: Optional[str] = None
        self._entry_price: Optional[float] = None
        self._trail_high: Optional[float] = None
        self._trail_low: Optional[float] = None
        self._trail_active: bool = False

        # 2-candle confirmation state
        self._pending_signal: Optional[str] = None
        # Store trigger candle close for confirmation check
        self._trigger_close: Optional[float] = None

        # Indicator cache
        self._df_cache: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @staticmethod
    def name() -> str:
        return "TripleEmaVwapMacdTsl"

    def get_min_candles(self) -> int:
        return (
            max(
                self.anchor_ema_period,
                self.macd_slow + self.macd_signal_period,
                self.atr_period,
            )
            + 2
        )

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        result = self._compute_indicators(df)
        if result is not None:
            self._df_cache = result
            return result
        self._df_cache = df
        return df

    def get_signal(self, candle_dict: dict, indicators_dict: dict) -> Optional[str]:
        if self._df_cache is None or len(self._df_cache) < 2:
            return None

        ts = candle_dict["timestamp"]
        mask = self._df_cache["timestamp"] == ts
        matched = self._df_cache.index[mask]
        if len(matched) == 0:
            return None
        i = matched[0]
        if i < 1:
            return None

        current = self._df_cache.iloc[i]
        prev = self._df_cache.iloc[i - 1]
        return self._check_entries(current, prev)

    def get_exit_signal(
        self,
        candle_dict: dict,
        indicators_dict: dict,
        position: dict,
    ) -> tuple:
        if self._df_cache is None:
            return None, None

        ts = candle_dict["timestamp"]
        mask = self._df_cache["timestamp"] == ts
        matched = self._df_cache.index[mask]
        if len(matched) == 0:
            return None, None
        i = matched[0]
        if i < 1:
            return None, None

        current = self._df_cache.iloc[i]
        prev = self._df_cache.iloc[i - 1]
        price = current["close"]
        entry = position["entry_price"]
        side = position["side"]

        # Sync internal state on first call
        if self._position is None:
            self._open_position("long" if side == "buy" else "short", entry)

        # ---- LONG exits ----
        if side == "buy":
            if self._trail_high is None or price > self._trail_high:
                self._trail_high = price
            best = self._trail_high
            gain_pct = (best - entry) / entry
            current_gain_pct = (price - entry) / entry

            # 1. Take Profit
            if current_gain_pct >= self.take_profit_pct:
                self._close_position()
                return "buy", "EXIT_TP"

            # 2. Stop Loss
            if (entry - price) / entry >= self.stop_loss_pct:
                self._close_position()
                return "buy", "EXIT_SL"

            # 3. Trailing Stop
            if gain_pct >= self.trail_activation_pct:
                self._trail_active = True
            if self._trail_active and price <= best * (1.0 - self.trail_pct):
                self._close_position()
                return "buy", "EXIT_TRAIL"

            # 4. VWAP Flip — only if position gain is below profit buffer
            vwap_flip = (
                prev["close"] >= prev["vwap"] and current["close"] < current["vwap"]
            )
            if vwap_flip and current_gain_pct < self.vwap_flip_min_profit_pct:
                self._close_position()
                return "buy", "EXIT_VWAP_FLIP"

        # ---- SHORT exits ----
        elif side == "sell":
            if self._trail_low is None or price < self._trail_low:
                self._trail_low = price
            best = self._trail_low
            gain_pct = (entry - best) / entry
            current_gain_pct = (entry - price) / entry

            # 1. Take Profit
            if current_gain_pct >= self.take_profit_pct:
                self._close_position()
                return "sell", "EXIT_TP"

            # 2. Stop Loss
            if (price - entry) / entry >= self.stop_loss_pct:
                self._close_position()
                return "sell", "EXIT_SL"

            # 3. Trailing Stop
            if gain_pct >= self.trail_activation_pct:
                self._trail_active = True
            if self._trail_active and price >= best * (1.0 + self.trail_pct):
                self._close_position()
                return "sell", "EXIT_TRAIL"

            # 4. VWAP Flip — only if position gain is below profit buffer
            vwap_flip = (
                prev["close"] <= prev["vwap"] and current["close"] > current["vwap"]
            )
            if vwap_flip and current_gain_pct < self.vwap_flip_min_profit_pct:
                self._close_position()
                return "sell", "EXIT_VWAP_FLIP"

        return None, None

    # ------------------------------------------------------------------
    # Indicator computation
    # ------------------------------------------------------------------

    def _compute_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        min_periods = self.get_min_candles()
        if len(df) < min_periods:
            return None

        df = df.copy()

        # --- Triple EMA ---
        df["ema_fast"] = df["close"].ewm(span=self.fast_ema_period, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.slow_ema_period, adjust=False).mean()
        df["ema_anchor"] = (
            df["close"].ewm(span=self.anchor_ema_period, adjust=False).mean()
        )

        # EMA fast slope
        df["ema_fast_slope"] = df["ema_fast"] - df["ema_fast"].shift(1)

        # EMA spread % (trend strength)
        df["ema_spread_pct"] = (
            abs(df["ema_fast"] - df["ema_anchor"]) / df["close"] * 100
        )

        # --- Session VWAP (resets each calendar day) ---
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        df["tp"] = (df["high"] + df["low"] + df["close"]) / 3.0
        df["tp_vol"] = df["tp"] * df["volume"]
        df["cum_tp_vol"] = df.groupby("date")["tp_vol"].cumsum()
        df["cum_vol"] = df.groupby("date")["volume"].cumsum()
        df["vwap"] = df["cum_tp_vol"] / df["cum_vol"].replace(0, np.nan)
        df["vwap"] = df["vwap"].ffill()

        # --- MACD ---
        ema_fast_macd = df["close"].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow_macd = df["close"].ewm(span=self.macd_slow, adjust=False).mean()
        df["macd_line"] = ema_fast_macd - ema_slow_macd
        df["macd_signal"] = (
            df["macd_line"].ewm(span=self.macd_signal_period, adjust=False).mean()
        )
        df["macd_hist"] = df["macd_line"] - df["macd_signal"]

        # MACD histogram slope
        df["macd_hist_slope"] = df["macd_hist"] - df["macd_hist"].shift(1)

        # --- CHANGE 1: ATR(14) volatility gate ---
        high_low = df["high"] - df["low"]
        high_prev_close = abs(df["high"] - df["close"].shift(1))
        low_prev_close = abs(df["low"] - df["close"].shift(1))
        true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(
            axis=1
        )
        df["atr"] = true_range.ewm(span=self.atr_period, adjust=False).mean()
        df["atr_pct"] = df["atr"] / df["close"]  # ATR as fraction of price

        return df

    # ------------------------------------------------------------------
    # Entry logic with 2-candle confirmation
    # ------------------------------------------------------------------

    def _check_entries(self, current: pd.Series, prev: pd.Series) -> Optional[str]:
        if self._position is not None:
            self._pending_signal = None
            self._trigger_close = None
            return None

        # --- CHANGE 1: ATR volatility gate ---
        # Skip entry entirely if market is too quiet
        if pd.isna(current["atr_pct"]) or current["atr_pct"] < self.min_atr_pct:
            self._pending_signal = None
            self._trigger_close = None
            return None

        # Option 1: MACD histogram momentum + CHANGE 3: absolute floor
        macd_hist_growing = (
            current["macd_hist_slope"] > 0
            and current["macd_hist"] > self.min_macd_hist  # absolute floor
        )
        macd_hist_falling = (
            current["macd_hist_slope"] < 0
            and current["macd_hist"] < -self.min_macd_hist  # absolute floor (mirrored)
        )

        # Option 2: EMA slope
        ema_slope_up = current["ema_fast_slope"] > 0
        ema_slope_down = current["ema_fast_slope"] < 0

        # Option 3: EMA spread (strong trend only)
        strong_trend = current["ema_spread_pct"] >= self.min_ema_spread_pct

        raw_long = (
            current["ema_fast"] > current["ema_slow"]
            and current["ema_slow"] > current["ema_anchor"]
            and current["close"] > current["vwap"]
            and current["macd_line"] > current["macd_signal"]
            and current["macd_line"] > 0
            and macd_hist_growing
            and ema_slope_up
            and strong_trend
        )

        raw_short = (
            current["ema_fast"] < current["ema_slow"]
            and current["ema_slow"] < current["ema_anchor"]
            and current["close"] < current["vwap"]
            and current["macd_line"] < current["macd_signal"]
            and current["macd_line"] < 0
            and macd_hist_falling
            and ema_slope_down
            and strong_trend
        )

        # --- 2-candle confirmation (CHANGE 2: stronger confirm candle checks) ---
        # Candle 1 (trigger): sets _pending_signal and stores trigger close
        # Candle 2 (confirm): must pass body >= 50% range AND close beyond trigger close

        if raw_long:
            if self._pending_signal == "buy" and self._trigger_close is not None:
                # CHANGE 2a: confirm candle body must be >= 50% of its range
                candle_range = current["high"] - current["low"]
                candle_body = abs(current["close"] - current["open"])
                body_ratio_ok = (candle_range > 0) and (
                    candle_body / candle_range >= 0.5
                )

                # CHANGE 2b: confirm candle close must be above trigger candle close
                close_beyond_trigger = current["close"] > self._trigger_close

                if body_ratio_ok and close_beyond_trigger:
                    self._open_position("long", current["close"])
                    self._pending_signal = None
                    self._trigger_close = None
                    return "buy"
                else:
                    # Confirm candle was weak — reset and treat current as new trigger
                    self._pending_signal = "buy"
                    self._trigger_close = current["close"]
            else:
                # First trigger candle
                self._pending_signal = "buy"
                self._trigger_close = current["close"]

        elif raw_short:
            if self._pending_signal == "sell" and self._trigger_close is not None:
                # CHANGE 2a: confirm candle body must be >= 50% of its range
                candle_range = current["high"] - current["low"]
                candle_body = abs(current["close"] - current["open"])
                body_ratio_ok = (candle_range > 0) and (
                    candle_body / candle_range >= 0.5
                )

                # CHANGE 2b: confirm candle close must be below trigger candle close
                close_beyond_trigger = current["close"] < self._trigger_close

                if body_ratio_ok and close_beyond_trigger:
                    self._open_position("short", current["close"])
                    self._pending_signal = None
                    self._trigger_close = None
                    return "sell"
                else:
                    # Confirm candle was weak — reset and treat current as new trigger
                    self._pending_signal = "sell"
                    self._trigger_close = current["close"]

        else:
            # Conditions broke — reset pending
            self._pending_signal = None
            self._trigger_close = None

        return None

    # ------------------------------------------------------------------
    # Position state helpers
    # ------------------------------------------------------------------

    def _open_position(self, direction: str, price: float):
        self._position = direction
        self._entry_price = price
        self._trail_high = price if direction == "long" else None
        self._trail_low = price if direction == "short" else None
        self._trail_active = False

    def _close_position(self):
        self._position = None
        self._entry_price = None
        self._trail_high = None
        self._trail_low = None
        self._trail_active = False

    # ------------------------------------------------------------------
    # Backtester compatibility
    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        return {
            "fast_ema_period": self.fast_ema_period,
            "slow_ema_period": self.slow_ema_period,
            "anchor_ema_period": self.anchor_ema_period,
            "macd_fast": self.macd_fast,
            "macd_slow": self.macd_slow,
            "macd_signal_period": self.macd_signal_period,
            "stop_loss_pct": round(self.stop_loss_pct * 100, 4),
            "take_profit_pct": round(self.take_profit_pct * 100, 4),
            "trail_activation_pct": round(self.trail_activation_pct * 100, 4),
            "trail_pct": round(self.trail_pct * 100, 4),
            "min_ema_spread_pct": self.min_ema_spread_pct,
            "vwap_flip_min_profit_pct": round(self.vwap_flip_min_profit_pct * 100, 4),
            "atr_period": self.atr_period,
            "min_atr_pct": self.min_atr_pct,
            "min_macd_hist": self.min_macd_hist,
        }

    def __repr__(self) -> str:
        p = self.get_params()
        return (
            f"TripleEmaVwapMacdTsl("
            f"fast={p['fast_ema_period']}, slow={p['slow_ema_period']}, "
            f"anchor={p['anchor_ema_period']}, "
            f"macd={p['macd_fast']}/{p['macd_slow']}/{p['macd_signal_period']}, "
            f"sl={p['stop_loss_pct']}%, tp={p['take_profit_pct']}%, "
            f"trail_act={p['trail_activation_pct']}%, trail={p['trail_pct']}%, "
            f"min_ema_spread={p['min_ema_spread_pct']}%, "
            f"vwap_flip_buffer={p['vwap_flip_min_profit_pct']}%, "
            f"atr_period={p['atr_period']}, min_atr_pct={p['min_atr_pct']}, "
            f"min_macd_hist={p['min_macd_hist']})"
        )
