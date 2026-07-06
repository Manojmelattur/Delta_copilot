# =============================================================================
# smc_strategy.py - Smart Money Concepts Strategy (CHoCH + BOS + Order Block)
# =============================================================================
# Logic:
#   1. Detect swing highs and swing lows (close-based, lookback N candles each side)
#   2. BOS confirms trend direction (bullish BOS = close above last swing high)
#   3. Price pulls back into a valid Order Block zone
#   4. CHoCH inside/near OB confirms reversal (close-based)
#   5. Entry on CHoCH candle close
#   6. SL: OB boundary +/- ATR buffer
#   7. TP: Entry +/- (SL distance x RR ratio)
#   8. Trail: ATR trailing after trail_activation_atr profit
#
# Contract specs (ETHUSD):
#   1 lot = 0.01 ETH, tick size = 0.05
# =============================================================================

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class OrderBlock:
    """Represents a single Order Block zone."""

    index: int
    top: float
    bottom: float
    direction: str
    strength: float
    age: int = 0
    mitigated: bool = False


@dataclass
class SwingPoint:
    """Represents a confirmed swing high or swing low."""

    index: int
    price: float
    kind: str  # 'high' or 'low'


@dataclass
class SMCSignal:
    """Output signal from the SMC strategy."""

    side: str  # 'buy' or 'sell'
    entry: float
    sl: float
    tp: float
    trail_activation: float
    ob: OrderBlock
    bos_level: float
    choch_level: float


# =============================================================================
# SMC Strategy
# =============================================================================


class SMCStrategy:
    """
    Smart Money Concepts strategy combining:
    - Swing high/low detection
    - Break of Structure (BOS) for trend confirmation
    - Change of Character (CHoCH) as entry trigger
    - Order Block zone detection
    - ATR-based SL, fixed RR TP, ATR trailing stop
    """

    def __init__(
        self,
        swing_lookback: int = 5,
        atr_length: int = 14,
        atr_multiplier: float = 1.5,
        trail_activation_atr: float = 2.0,
        sl_atr_buffer: float = 0.3,
        ob_strength_period: int = 20,
        ob_strength_mult: float = 0.3,
        max_ob_age_candles: int = 30,
        rr_ratio: float = 2.0,
        trend_ema_period: int = 50,
        min_atr_threshold: float = 0.0,
        bos_type: str = "close",
        choch_type: str = "close",
        structure_lookback: int = 50,
        ob_proximity_atr: float = 3.0,
    ):
        self.swing_lookback = swing_lookback
        self.atr_length = atr_length
        self.atr_multiplier = atr_multiplier
        self.trail_activation_atr = trail_activation_atr
        self.sl_atr_buffer = sl_atr_buffer
        self.ob_strength_period = ob_strength_period
        self.ob_strength_mult = ob_strength_mult
        self.max_ob_age_candles = max_ob_age_candles
        self.rr_ratio = rr_ratio
        self.trend_ema_period = trend_ema_period
        self.min_atr_threshold = min_atr_threshold
        self.bos_type = bos_type
        self.choch_type = choch_type
        self.structure_lookback = structure_lookback
        self.ob_proximity_atr = ob_proximity_atr

        # Internal state
        self._pending_signal = None
        self._active_trade = None
        self._last_signal = None
        self._df = None

        # FIX: Track position open state for notify_entry / notify_exit
        self._position_open = False

    # -------------------------------------------------------------------------
    # Backtester Interface
    # -------------------------------------------------------------------------

    def name(self) -> str:
        return "SMCStrategy"

    def get_params(self) -> str:
        return (
            f"swing_lookback={self.swing_lookback}, "
            f"atr_length={self.atr_length}, "
            f"atr_multiplier={self.atr_multiplier}, "
            f"trail_activation_atr={self.trail_activation_atr}, "
            f"sl_atr_buffer={self.sl_atr_buffer}, "
            f"ob_strength_period={self.ob_strength_period}, "
            f"ob_strength_mult={self.ob_strength_mult}, "
            f"max_ob_age_candles={self.max_ob_age_candles}, "
            f"rr_ratio={self.rr_ratio}, "
            f"trend_ema_period={self.trend_ema_period}, "
            f"bos_type={self.bos_type}, "
            f"choch_type={self.choch_type}, "
            f"ob_proximity_atr={self.ob_proximity_atr}"
        )

    def __repr__(self):
        return f"SMCStrategy({self.get_params()})"

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Required by backtester.py interface.
        Pre-computes indicators and stores full df on instance
        for use by get_signal() during the backtest loop.
        """
        df = df.copy()
        df["atr"] = self._compute_atr(df)
        df["ema_trend"] = self._compute_ema(df["close"], self.trend_ema_period)
        df["atr_strength"] = df["atr"].rolling(self.ob_strength_period).mean()
        self._df = df
        return df

    def get_min_candles(self) -> int:
        """
        Required by backtester.py interface.
        Returns minimum number of candles needed before strategy
        can generate a valid signal.
        """
        return (
            max(
                self.atr_length,
                self.trend_ema_period,
                self.swing_lookback * 2 + 1,
                self.structure_lookback,
                self.ob_strength_period,
            )
            + 10
        )

    def get_signal(self, candle_dict: dict, indicators_dict: dict) -> Optional[str]:
        """
        Required by backtester.py interface.
        Called per candle with current candle data and indicators.
        Uses stored full df sliced to current_idx for lookback-dependent logic.
        Returns "buy" or "sell" string, or None if no signal.
        """
        # FIX: Do not generate a new entry signal if a position is already open.
        if self._position_open:
            return None

        current_idx = indicators_dict.get("current_idx")
        if self._df is None or current_idx is None:
            return None

        df_slice = self._df.iloc[: current_idx + 1]

        if len(df_slice) < self.get_min_candles():
            return None

        signal = self.generate_signal(df_slice)
        if signal is None:
            self._last_signal = None
            return None

        self._last_signal = signal
        return signal.side

    # FIX: Align key names with what bot.py expects in _execute_entry.
    # bot.py reads: _ob_sl_price, _ob_tp_price, _ob_atr_at_entry
    def get_last_entry_levels(self) -> Optional[dict]:
        """
        Called by bot.py _execute_entry() after get_signal() returns a signal.
        Returns cached signal levels using keys expected by bot.py.
        """
        if self._last_signal is None:
            return None
        signal = self._last_signal
        atr = None
        if self._df is not None:
            atr_series = self._compute_atr(self._df)
            atr = float(atr_series.iloc[-1]) if not atr_series.empty else None
        return {
            "_ob_sl_price": signal.sl,
            "_ob_tp_price": signal.tp,
            "_ob_atr_at_entry": atr,
            "trail_activation": signal.trail_activation,
            "ob_top": signal.ob.top,
            "ob_bottom": signal.ob.bottom,
            "bos_level": signal.bos_level,
            "choch_level": signal.choch_level,
        }

    # FIX: Add get_exit_signal() — called by bot.py on every iteration
    # when a position is open. Checks SL, TP, and ATR trailing stop.
    def get_exit_signal(
        self,
        candle_dict: dict,
        indicators_dict: dict,
        position: dict,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Required by bot.py main loop.
        Called every candle when a position is open.

        Returns:
            (exit_side, exit_reason) if exit condition met
            (None, None) if position should be held
        """
        side = position["side"]
        entry_price = position["entry_price"]
        sl_price = position.get("sl_price")
        tp_price = position.get("tp_price")
        current_price = float(candle_dict.get("close", entry_price))

        # Retrieve ATR from indicators if available
        current_atr = indicators_dict.get("atr")
        if current_atr is None and self._df is not None:
            atr_series = self._compute_atr(self._df)
            current_atr = float(atr_series.iloc[-1]) if not atr_series.empty else None

        exit_side = "sell" if side == "buy" else "buy"

        # --- Hard Stop Loss ---
        if sl_price is not None:
            if side == "buy" and current_price <= sl_price:
                return exit_side, "SL"
            if side == "sell" and current_price >= sl_price:
                return exit_side, "SL"

        # --- Take Profit ---
        if tp_price is not None:
            if side == "buy" and current_price >= tp_price:
                return exit_side, "TP"
            if side == "sell" and current_price <= tp_price:
                return exit_side, "TP"

        # --- ATR Trailing Stop ---
        if current_atr is not None and sl_price is not None:
            trail_activation = position.get(
                "trail_activation",
                entry_price + self.trail_activation_atr * current_atr
                if side == "buy"
                else entry_price - self.trail_activation_atr * current_atr,
            )
            new_trail_sl = self.update_trailing_stop(
                current_price,
                current_atr,
                {
                    "side": side,
                    "trail_activation": trail_activation,
                    "sl": sl_price,
                    "trail_sl": position.get("trail_sl", sl_price),
                },
            )
            if new_trail_sl is not None:
                position["trail_sl"] = new_trail_sl
                position["sl_price"] = new_trail_sl

            # Check updated trailing SL
            trail_sl = position.get("trail_sl", sl_price)
            if side == "buy" and current_price <= trail_sl:
                return exit_side, "TRAIL_SL"
            if side == "sell" and current_price >= trail_sl:
                return exit_side, "TRAIL_SL"

        # --- Signal Flip (optional: exit if opposite signal fires) ---
        if not self._position_open:
            return exit_side, "signal_flip"

        return None, None

    # FIX: Add notify_entry() — called by bot.py after a position is opened.
    def notify_entry(self, candle_idx: int):
        """Called by bot.py after entry order is confirmed."""
        self._position_open = True

    # FIX: Add notify_exit() — called by bot.py after a position is closed.
    def notify_exit(self):
        """Called by bot.py after exit order is confirmed."""
        self._position_open = False
        self._last_signal = None

    def reset_state(self):
        """Reset internal state. Called by backtester between runs."""
        self._pending_signal = None
        self._active_trade = None
        self._last_signal = None
        self._df = None
        self._position_open = False

    # -------------------------------------------------------------------------
    # Indicator Calculations
    # -------------------------------------------------------------------------

    def _compute_atr(self, df: pd.DataFrame) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        return tr.ewm(alpha=1 / self.atr_length, adjust=False).mean()

    def _compute_ema(self, series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    # -------------------------------------------------------------------------
    # Swing Detection
    # -------------------------------------------------------------------------

    def _detect_swings(self, df: pd.DataFrame) -> list:
        n = self.swing_lookback
        swings = []
        highs = df["high"].values
        lows = df["low"].values

        for i in range(n, len(df) - n):
            if all(highs[i] > highs[i - j] for j in range(1, n + 1)) and all(
                highs[i] > highs[i + j] for j in range(1, n + 1)
            ):
                swings.append(SwingPoint(index=i, price=highs[i], kind="high"))

            if all(lows[i] < lows[i - j] for j in range(1, n + 1)) and all(
                lows[i] < lows[i + j] for j in range(1, n + 1)
            ):
                swings.append(SwingPoint(index=i, price=lows[i], kind="low"))

        return sorted(swings, key=lambda s: s.index)

    # -------------------------------------------------------------------------
    # BOS Detection
    # -------------------------------------------------------------------------

    def _detect_bos(
        self, df: pd.DataFrame, swings: list, current_idx: int
    ) -> Optional[dict]:
        lookback_start = max(0, current_idx - self.structure_lookback * 2)
        relevant_swings = [
            s
            for s in swings
            if lookback_start <= s.index < current_idx - self.swing_lookback
        ]

        if len(relevant_swings) < 2:
            return None

        swing_highs = [s for s in relevant_swings if s.kind == "high"]
        swing_lows = [s for s in relevant_swings if s.kind == "low"]

        if not swing_highs or not swing_lows:
            return None

        bos_result = None
        bearish_bos = None

        for sh in reversed(swing_highs):
            for i in range(sh.index + 1, current_idx):
                if self.bos_type == "close":
                    broke = df["close"].iloc[i] > sh.price
                else:
                    broke = df["high"].iloc[i] > sh.price
                if broke:
                    bos_result = {
                        "direction": "bullish",
                        "level": sh.price,
                        "swing_index": sh.index,
                        "bos_candle_index": i,
                    }
                    break
            if bos_result:
                break

        for sl in reversed(swing_lows):
            for i in range(sl.index + 1, current_idx):
                if self.bos_type == "close":
                    broke = df["close"].iloc[i] < sl.price
                else:
                    broke = df["low"].iloc[i] < sl.price
                if broke:
                    bearish_bos = {
                        "direction": "bearish",
                        "level": sl.price,
                        "swing_index": sl.index,
                        "bos_candle_index": i,
                    }
                    break
            if bearish_bos:
                break

        if bos_result and bearish_bos:
            return (
                bos_result
                if bos_result["bos_candle_index"] > bearish_bos["bos_candle_index"]
                else bearish_bos
            )
        return bos_result or bearish_bos

    # -------------------------------------------------------------------------
    # CHoCH Detection
    # -------------------------------------------------------------------------

    def _detect_choch(
        self, df: pd.DataFrame, swings: list, bos: dict, current_idx: int
    ) -> Optional[dict]:
        bos_candle = bos["bos_candle_index"]
        bos_direction = bos["direction"]
        lookback_start = max(0, current_idx - self.structure_lookback * 2)
        pre_bos_swings = [s for s in swings if lookback_start <= s.index < bos_candle]

        if not pre_bos_swings:
            return None

        if bos_direction == "bullish":
            pre_bos_highs = [s for s in pre_bos_swings if s.kind == "high"]
            if not pre_bos_highs:
                return None
            choch_swing = pre_bos_highs[-1]
            search_start = max(choch_swing.index + 1, bos_candle + 1)
            for i in range(search_start, current_idx + 1):
                if self.choch_type == "close":
                    broke = df["close"].iloc[i] > choch_swing.price
                else:
                    broke = df["high"].iloc[i] > choch_swing.price
                if broke:
                    return {
                        "direction": "bullish",
                        "level": choch_swing.price,
                        "choch_candle_index": i,
                    }

        elif bos_direction == "bearish":
            pre_bos_lows = [s for s in pre_bos_swings if s.kind == "low"]
            if not pre_bos_lows:
                return None
            choch_swing = pre_bos_lows[-1]
            search_start = max(choch_swing.index + 1, bos_candle + 1)
            for i in range(search_start, current_idx + 1):
                if self.choch_type == "close":
                    broke = df["close"].iloc[i] < choch_swing.price
                else:
                    broke = df["low"].iloc[i] < choch_swing.price
                if broke:
                    return {
                        "direction": "bearish",
                        "level": choch_swing.price,
                        "choch_candle_index": i,
                    }

        return None

    # -------------------------------------------------------------------------
    # Order Block Detection
    # -------------------------------------------------------------------------

    def _detect_order_blocks(
        self, df: pd.DataFrame, atr: pd.Series, direction: str, current_idx: int
    ) -> list:
        obs = []
        lookback_start = max(0, current_idx - self.max_ob_age_candles)
        strength_atr = atr.rolling(self.ob_strength_period).mean()

        for i in range(lookback_start, current_idx - 2):
            candle = df.iloc[i]
            next_candle = df.iloc[i + 1]
            atr_val = strength_atr.iloc[i]

            if pd.isna(atr_val) or atr_val == 0:
                continue

            if direction == "bullish":
                is_bearish_candle = candle["close"] < candle["open"]
                impulse_body = next_candle["close"] - next_candle["open"]
                is_bullish_impulse = (
                    next_candle["close"] > next_candle["open"]
                    and impulse_body > self.ob_strength_mult * atr_val
                )
                if is_bearish_candle and is_bullish_impulse:
                    ob = OrderBlock(
                        index=i,
                        top=candle["high"],
                        bottom=candle["low"],
                        direction="bullish",
                        strength=atr_val,
                        age=current_idx - i,
                    )
                    ob.mitigated = any(
                        ob.bottom <= df["close"].iloc[j] <= ob.top
                        for j in range(i + 1, current_idx)
                    )
                    if not ob.mitigated:
                        obs.append(ob)

            elif direction == "bearish":
                is_bullish_candle = candle["close"] > candle["open"]
                impulse_body = next_candle["open"] - next_candle["close"]
                is_bearish_impulse = (
                    next_candle["close"] < next_candle["open"]
                    and impulse_body > self.ob_strength_mult * atr_val
                )
                if is_bullish_candle and is_bearish_impulse:
                    ob = OrderBlock(
                        index=i,
                        top=candle["high"],
                        bottom=candle["low"],
                        direction="bearish",
                        strength=atr_val,
                        age=current_idx - i,
                    )
                    ob.mitigated = any(
                        ob.bottom <= df["close"].iloc[j] <= ob.top
                        for j in range(i + 1, current_idx)
                    )
                    if not ob.mitigated:
                        obs.append(ob)

        return obs

    # -------------------------------------------------------------------------
    # Price in OB Zone Check
    # -------------------------------------------------------------------------

    def _price_in_ob(self, price: float, ob: OrderBlock) -> bool:
        return ob.bottom <= price <= ob.top

    # -------------------------------------------------------------------------
    # Core Signal Generation
    # -------------------------------------------------------------------------

    def generate_signal(self, df: pd.DataFrame) -> Optional[SMCSignal]:
        if len(df) < self.get_min_candles():
            return None

        atr = self._compute_atr(df)
        ema = self._compute_ema(df["close"], self.trend_ema_period)

        current_idx = len(df) - 1
        current_close = df["close"].iloc[current_idx]
        current_atr = atr.iloc[current_idx]
        current_ema = ema.iloc[current_idx]

        if self.min_atr_threshold > 0 and current_atr < self.min_atr_threshold:
            return None

        swings = self._detect_swings(df)
        if len(swings) < 4:
            return None

        bos = self._detect_bos(df, swings, current_idx)
        if bos is None:
            return None

        bos_direction = bos["direction"]

        if bos_direction == "bullish" and current_close < current_ema:
            return None
        if bos_direction == "bearish" and current_close > current_ema:
            return None

        choch = self._detect_choch(df, swings, bos, current_idx)
        if choch is None:
            return None

        if bos_direction != choch["direction"]:
            return None

        obs = self._detect_order_blocks(df, atr, bos_direction, current_idx)
        if not obs:
            return None

        valid_ob = None
        for ob in reversed(obs):
            if bos_direction == "bullish":
                distance = current_close - ob.top
                if (
                    ob.top < current_close
                    and distance < self.ob_proximity_atr * current_atr
                ):
                    valid_ob = ob
                    break
            elif bos_direction == "bearish":
                distance = ob.bottom - current_close
                if (
                    ob.bottom > current_close
                    and distance < self.ob_proximity_atr * current_atr
                ):
                    valid_ob = ob
                    break

        if valid_ob is None:
            return None

        entry = current_close

        if bos_direction == "bullish":
            sl = round(valid_ob.bottom - (self.sl_atr_buffer * current_atr), 2)
            risk = entry - sl
            if risk <= 0:
                return None
            tp = round(entry + (risk * self.rr_ratio), 2)
            trail_activation = entry + (self.trail_activation_atr * current_atr)
        else:
            sl = round(valid_ob.top + (self.sl_atr_buffer * current_atr), 2)
            risk = sl - entry
            if risk <= 0:
                return None
            tp = round(entry - (risk * self.rr_ratio), 2)
            trail_activation = entry - (self.trail_activation_atr * current_atr)

        return SMCSignal(
            side="buy" if bos_direction == "bullish" else "sell",
            entry=entry,
            sl=sl,
            tp=tp,
            trail_activation=trail_activation,
            ob=valid_ob,
            bos_level=bos["level"],
            choch_level=choch["level"],
        )

    # -------------------------------------------------------------------------
    # Trade Management
    # -------------------------------------------------------------------------

    def update_trailing_stop(
        self, current_price: float, current_atr: float, trade: dict
    ) -> Optional[float]:
        side = trade["side"]
        trail_activation = trade["trail_activation"]
        current_sl = trade.get("trail_sl", trade["sl"])
        trail_distance = self.atr_multiplier * current_atr

        if side == "buy":
            if current_price >= trail_activation:
                new_sl = round(current_price - trail_distance, 2)
                if new_sl > current_sl:
                    return new_sl
        elif side == "sell":
            if current_price <= trail_activation:
                new_sl = round(current_price + trail_distance, 2)
                if new_sl < current_sl:
                    return new_sl

        return None

    # -------------------------------------------------------------------------
    # Debug Methods
    # -------------------------------------------------------------------------

    def debug_signal(self, df: pd.DataFrame) -> dict:
        report = {}
        min_candles = self.get_min_candles()
        report["candles_available"] = len(df)
        report["min_candles_required"] = min_candles

        if len(df) < min_candles:
            report["blocked_at"] = "insufficient_candles"
            return report

        atr = self._compute_atr(df)
        ema = self._compute_ema(df["close"], self.trend_ema_period)
        current_idx = len(df) - 1
        current_close = df["close"].iloc[current_idx]
        current_atr = atr.iloc[current_idx]
        current_ema = ema.iloc[current_idx]

        report["current_close"] = round(current_close, 4)
        report["current_atr"] = round(current_atr, 4)
        report["current_ema"] = round(current_ema, 4)
        report["ema_filter"] = "pass"
        report["ob_proximity_atr_threshold"] = self.ob_proximity_atr

        if self.min_atr_threshold > 0 and current_atr < self.min_atr_threshold:
            report["blocked_at"] = "atr_threshold"
            return report

        swings = self._detect_swings(df)
        swing_highs = [s for s in swings if s.kind == "high"]
        swing_lows = [s for s in swings if s.kind == "low"]
        report["total_swings"] = len(swings)
        report["swing_highs"] = len(swing_highs)
        report["swing_lows"] = len(swing_lows)

        if len(swings) < 4:
            report["blocked_at"] = "insufficient_swings"
            return report

        bos = self._detect_bos(df, swings, current_idx)
        report["bos_found"] = bos is not None
        if bos:
            report["bos_direction"] = bos["direction"]
            report["bos_level"] = round(bos["level"], 4)
            report["bos_candle"] = bos["bos_candle_index"]

        if bos is None:
            report["blocked_at"] = "no_bos"
            return report

        bos_direction = bos["direction"]
        ema_aligned = (bos_direction == "bullish" and current_close >= current_ema) or (
            bos_direction == "bearish" and current_close <= current_ema
        )
        report["ema_aligned"] = ema_aligned
        if not ema_aligned:
            report["blocked_at"] = "ema_filter"
            return report

        choch = self._detect_choch(df, swings, bos, current_idx)
        report["choch_found"] = choch is not None
        if choch:
            report["choch_direction"] = choch["direction"]
            report["choch_level"] = round(choch["level"], 4)
            report["choch_candle"] = choch["choch_candle_index"]

        if choch is None:
            report["blocked_at"] = "no_choch"
            return report

        if bos_direction != choch["direction"]:
            report["blocked_at"] = "bos_choch_direction_mismatch"
            return report

        obs = self._detect_order_blocks(df, atr, bos_direction, current_idx)
        report["obs_found"] = len(obs)

        if not obs:
            report["blocked_at"] = "no_order_blocks"
            return report

        valid_ob = None
        for ob in reversed(obs):
            if bos_direction == "bullish":
                distance = current_close - ob.top
                if (
                    ob.top < current_close
                    and distance < self.ob_proximity_atr * current_atr
                ):
                    valid_ob = ob
                    break
            elif bos_direction == "bearish":
                distance = ob.bottom - current_close
                if (
                    ob.bottom > current_close
                    and distance < self.ob_proximity_atr * current_atr
                ):
                    valid_ob = ob
                    break

        report["valid_ob_near_choch"] = valid_ob is not None
        if valid_ob:
            report["ob_top"] = round(valid_ob.top, 4)
            report["ob_bottom"] = round(valid_ob.bottom, 4)
            report["ob_age"] = valid_ob.age
            if bos_direction == "bullish":
                report["ob_distance_atr"] = round(
                    (current_close - valid_ob.top) / current_atr, 2
                )
            else:
                report["ob_distance_atr"] = round(
                    (valid_ob.bottom - current_close) / current_atr, 2
                )

        if valid_ob is None:
            report["blocked_at"] = "no_ob_within_proximity"
            return report

        report["blocked_at"] = "none - signal should generate"
        return report

    def debug_choch(self, df: pd.DataFrame) -> dict:
        report = {}
        current_idx = len(df) - 1
        swings = self._detect_swings(df)
        bos = self._detect_bos(df, swings, current_idx)

        if bos is None:
            report["error"] = "no_bos_found"
            return report

        bos_candle = bos["bos_candle_index"]
        bos_direction = bos["direction"]
        lookback_start = max(0, current_idx - self.structure_lookback * 2)

        report["bos_direction"] = bos_direction
        report["bos_candle"] = bos_candle
        report["bos_level"] = round(bos["level"], 4)
        report["current_idx"] = current_idx
        report["lookback_start"] = lookback_start

        pre_bos_swings = [s for s in swings if lookback_start <= s.index < bos_candle]
        pre_bos_highs = [s for s in pre_bos_swings if s.kind == "high"]
        pre_bos_lows = [s for s in pre_bos_swings if s.kind == "low"]

        report["pre_bos_swings_total"] = len(pre_bos_swings)
        report["pre_bos_highs"] = len(pre_bos_highs)
        report["pre_bos_lows"] = len(pre_bos_lows)

        if bos_direction == "bullish" and pre_bos_highs:
            choch_swing = pre_bos_highs[-1]
            search_start = max(choch_swing.index + 1, bos_candle + 1)
            report["choch_swing_index"] = choch_swing.index
            report["choch_swing_price"] = round(choch_swing.price, 4)
            report["search_from"] = search_start
            report["search_to"] = current_idx + 1
            closes = [
                round(df["close"].iloc[i], 4)
                for i in range(search_start, current_idx + 1)
            ]
            report["closes_in_range_count"] = len(closes)
            report["max_close_in_range"] = max(closes) if closes else None
            report["choch_swing_price_target"] = round(choch_swing.price, 4)
            report["any_close_above_target"] = any(
                c > choch_swing.price for c in closes
            )

        elif bos_direction == "bearish" and pre_bos_lows:
            choch_swing = pre_bos_lows[-1]
            search_start = max(choch_swing.index + 1, bos_candle + 1)
            report["choch_swing_index"] = choch_swing.index
            report["choch_swing_price"] = round(choch_swing.price, 4)
            report["search_from"] = search_start
            report["search_to"] = current_idx + 1
            closes = [
                round(df["close"].iloc[i], 4)
                for i in range(search_start, current_idx + 1)
            ]
            report["closes_in_range_count"] = len(closes)
            report["min_close_in_range"] = min(closes) if closes else None
            report["choch_swing_price_target"] = round(choch_swing.price, 4)
            report["any_close_below_target"] = any(
                c < choch_swing.price for c in closes
            )

        return report

    def debug_ob(self, df: pd.DataFrame) -> dict:
        report = {}
        atr = self._compute_atr(df)
        current_idx = len(df) - 1
        lookback_start = max(0, current_idx - self.max_ob_age_candles)
        strength_atr = atr.rolling(self.ob_strength_period).mean()

        swings = self._detect_swings(df)
        bos = self._detect_bos(df, swings, current_idx)

        if bos is None:
            report["error"] = "no_bos"
            return report

        direction = bos["direction"]
        report["bos_direction"] = direction
        report["ob_strength_mult"] = self.ob_strength_mult
        report["max_ob_age_candles"] = self.max_ob_age_candles
        report["lookback_start"] = lookback_start
        report["current_idx"] = current_idx
        report["candles_scanned"] = current_idx - lookback_start

        candidates = failed_direction = failed_impulse_size = failed_mitigated = (
            passed
        ) = 0
        impulse_sizes = []

        for i in range(lookback_start, current_idx - 2):
            candle = df.iloc[i]
            next_candle = df.iloc[i + 1]
            atr_val = strength_atr.iloc[i]

            if pd.isna(atr_val) or atr_val == 0:
                continue

            if direction == "bullish":
                is_correct_candle = candle["close"] < candle["open"]
                impulse_body = next_candle["close"] - next_candle["open"]
                is_bullish_impulse = next_candle["close"] > next_candle["open"]
                threshold = self.ob_strength_mult * atr_val

                if is_correct_candle:
                    candidates += 1
                    if not is_bullish_impulse:
                        failed_direction += 1
                    elif impulse_body <= threshold:
                        failed_impulse_size += 1
                        impulse_sizes.append(round(impulse_body / atr_val, 3))
                    else:
                        mitigated = any(
                            candle["low"] <= df["close"].iloc[j] <= candle["high"]
                            for j in range(i + 1, current_idx)
                        )
                        if mitigated:
                            failed_mitigated += 1
                        else:
                            passed += 1

            elif direction == "bearish":
                is_correct_candle = candle["close"] > candle["open"]
                impulse_body = next_candle["open"] - next_candle["close"]
                is_bearish_impulse = next_candle["close"] < next_candle["open"]
                threshold = self.ob_strength_mult * atr_val

                if is_correct_candle:
                    candidates += 1
                    if not is_bearish_impulse:
                        failed_direction += 1
                    elif impulse_body <= threshold:
                        failed_impulse_size += 1
                        impulse_sizes.append(round(impulse_body / atr_val, 3))
                    else:
                        mitigated = any(
                            candle["low"] <= df["close"].iloc[j] <= candle["high"]
                            for j in range(i + 1, current_idx)
                        )
                        if mitigated:
                            failed_mitigated += 1
                        else:
                            passed += 1

        report["ob_candidates"] = candidates
        report["failed_wrong_dir"] = failed_direction
        report["failed_impulse_size"] = failed_impulse_size
        report["failed_mitigated"] = failed_mitigated
        report["obs_passed"] = passed

        if impulse_sizes:
            report["impulse_atr_ratio_min"] = min(impulse_sizes)
            report["impulse_atr_ratio_max"] = max(impulse_sizes)
            report["impulse_atr_ratio_avg"] = round(
                sum(impulse_sizes) / len(impulse_sizes), 3
            )
            report["suggested_strength_mult"] = round(
                sum(impulse_sizes) / len(impulse_sizes) * 0.7, 2
            )

        return report
