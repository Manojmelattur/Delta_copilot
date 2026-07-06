# =============================================================================
# ob_strategy.py - Order Block (OB) Strategy
# =============================================================================

import logging
import sqlite3

import pandas as pd

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class OrderBlockStrategy(BaseStrategy):
    """
    Order Block (OB) Trend Continuation Strategy.

    Detection:
        Bullish OB : Last bearish candle before a strong bullish move
                     Strong move = next candle body >= ob_strength_mult * 20-period avg body
                     Gap zone    : [low[i-1], high[i-1]]  (the bearish OB candle)

        Bearish OB : Last bullish candle before a strong bearish move
                     Strong move = next candle body >= ob_strength_mult * 20-period avg body
                     Gap zone    : [low[i-1], high[i-1]]  (the bullish OB candle)

    Entry:
        Price candle closes inside the OB zone (retracement confirmation).
        EMA trend filter: bullish OBs only above trend_ema, bearish OBs only below.
        ATR threshold filter: skip entries when ATR < min_atr_threshold (low volatility).

    Exit Priority (highest to lowest):
        1. EXIT_TP    - Take Profit hit (1:RR from entry)
        2. EXIT_SL    - Hard Stop Loss hit (OB wick extreme + sl_atr_buffer * ATR)
        3. EXIT_TRAIL - Trailing Stop hit (atr_multiplier * ATR, activates after
                        trail_activation_atr * ATR move)
    """

    def __init__(
        self,
        atr_length: int = 14,
        atr_multiplier: float = 1.5,
        trail_activation_atr: float = 1.5,
        ob_strength_period: int = 20,
        ob_strength_mult: float = 1.5,
        rr_ratio: float = 3.0,
        sl_atr_buffer: float = 0.1,
        max_ob_age_candles: int = 50,
        trend_ema_period: int = 50,
        min_atr_threshold: float = 0.0,
        db_path: str = "strategy_state.db",
        is_backtest: bool = False,  # FIX 2: backtest isolation flag
    ):
        self.atr_length = atr_length
        self.atr_multiplier = atr_multiplier
        self.trail_activation_atr = trail_activation_atr
        self.ob_strength_period = ob_strength_period
        self.ob_strength_mult = ob_strength_mult
        self.rr_ratio = rr_ratio
        self.sl_atr_buffer = sl_atr_buffer
        self.max_ob_age_candles = max_ob_age_candles
        self.trend_ema_period = trend_ema_period
        self.min_atr_threshold = min_atr_threshold
        self.db_path = db_path
        self.is_backtest = is_backtest  # FIX 2

        # Position tracking
        self._in_position = False

        # Last computed entry levels
        self._last_sl_price: float | None = None
        self._last_tp_price: float | None = None
        self._last_atr_at_entry: float | None = None
        self._last_ob_high: float | None = None
        self._last_ob_low: float | None = None

        # Diagnostic counters
        self._diag_candles_processed = 0
        self._diag_ob_bull_detected = 0
        self._diag_ob_bear_detected = 0
        self._diag_nan_skipped = 0
        self._diag_retracement_checks = 0
        self._diag_entries_fired = 0
        self._diag_trend_filtered = 0
        self._diag_atr_filtered = 0

        self._init_db()

        # FIX 2: Skip loading persisted state entirely during backtest
        if self.is_backtest:
            self._pending_ob_bias = None
            self._pending_ob_high = None
            self._pending_ob_low = None
            self._pending_ob_age = 0
            self._candle_index = 0
            self._in_position = False
        else:
            self._pending_ob_bias = self._load_state_value("ob_pending_bias")
            self._pending_ob_high = self._load_float("ob_pending_high")
            self._pending_ob_low = self._load_float("ob_pending_low")
            self._pending_ob_age = int(self._load_state_value("ob_pending_age") or 0)
            self._candle_index = int(self._load_state_value("ob_candle_index") or 0)

            # FIX 5: Restore _in_position from DB for live bot
            in_pos_val = self._load_state_value("ob_in_position")
            self._in_position = in_pos_val == "true"
            if self._in_position:
                logger.warning(
                    "[OB] Restored in_position=True from database. "
                    "Bot may have restarted mid-trade."
                )

            if self._pending_ob_bias is not None:
                logger.warning(
                    f"Restored pending OB '{self._pending_ob_bias}' from database "
                    f"(zone=[{self._pending_ob_low}, {self._pending_ob_high}], "
                    f"age={self._pending_ob_age}). Bot may have restarted mid-detection."
                )

    # -------------------------------------------------------------------------
    # State persistence helpers
    # -------------------------------------------------------------------------

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_state (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )
            conn.commit()

    def _load_state_value(self, key: str) -> str | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT value FROM strategy_state WHERE key = ?", (key,)
            ).fetchone()
        if row is None or row[0] == "null":
            return None
        return row[0]

    def _load_float(self, key: str) -> float | None:
        val = self._load_state_value(key)
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def _save_state_value(self, key: str, value: str | None) -> None:
        # FIX 2: Skip DB writes in backtest mode
        if self.is_backtest:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO strategy_state (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value if value is not None else "null"),
            )
            conn.commit()

    def _save_ob_state(self) -> None:
        # FIX 2: Skip DB writes in backtest mode
        if self.is_backtest:
            return
        with sqlite3.connect(self.db_path) as conn:
            fields = {
                "ob_pending_bias": self._pending_ob_bias
                if self._pending_ob_bias
                else "null",
                "ob_pending_high": str(self._pending_ob_high)
                if self._pending_ob_high is not None
                else "null",
                "ob_pending_low": str(self._pending_ob_low)
                if self._pending_ob_low is not None
                else "null",
                "ob_pending_age": str(self._pending_ob_age),
                "ob_candle_index": str(self._candle_index),
            }
            for key, value in fields.items():
                conn.execute(
                    """
                    INSERT INTO strategy_state (key, value)
                    VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    (key, value),
                )
            conn.commit()

    def _clear_pending_ob(self) -> None:
        self._pending_ob_bias = None
        self._pending_ob_high = None
        self._pending_ob_low = None
        self._pending_ob_age = 0
        self._save_ob_state()

    def _set_pending_ob(self, bias: str, ob_high: float, ob_low: float) -> None:
        self._pending_ob_bias = bias
        self._pending_ob_high = ob_high
        self._pending_ob_low = ob_low
        self._pending_ob_age = 0
        self._save_ob_state()

    def _clear_last_entry_levels(self) -> None:
        self._last_sl_price = None
        self._last_tp_price = None
        self._last_atr_at_entry = None
        self._last_ob_high = None
        self._last_ob_low = None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_last_entry_levels(self) -> dict:
        if self._last_sl_price is None:
            return {}
        return {
            "_ob_sl_price": self._last_sl_price,
            "_ob_tp_price": self._last_tp_price,
            "_ob_atr_at_entry": self._last_atr_at_entry,
            "_ob_high": self._last_ob_high,
            "_ob_low": self._last_ob_low,
        }

    # -------------------------------------------------------------------------
    # Position lifecycle hooks
    # -------------------------------------------------------------------------

    def notify_entry(self, idx: int) -> None:
        self._in_position = True
        self._save_state_value("ob_in_position", "true")  # FIX 5

    def notify_exit(self) -> None:
        self._in_position = False
        self._save_state_value("ob_in_position", "false")  # FIX 5

    # -------------------------------------------------------------------------
    # Strategy metadata
    # -------------------------------------------------------------------------

    def name(self) -> str:
        return (
            f"OrderBlock Strategy("
            f"atr_length={self.atr_length}, "
            f"atr_multiplier={self.atr_multiplier}, "
            f"trail_activation_atr={self.trail_activation_atr}, "
            f"ob_strength_period={self.ob_strength_period}, "
            f"ob_strength_mult={self.ob_strength_mult}, "
            f"rr_ratio={self.rr_ratio}, "
            f"sl_atr_buffer={self.sl_atr_buffer}, "
            f"max_ob_age_candles={self.max_ob_age_candles}, "
            f"trend_ema_period={self.trend_ema_period}, "
            f"min_atr_threshold={self.min_atr_threshold})"
        )

    def get_params(self) -> dict:
        return {
            "atr_length": self.atr_length,
            "atr_multiplier": self.atr_multiplier,
            "trail_activation_atr": self.trail_activation_atr,
            "ob_strength_period": self.ob_strength_period,
            "ob_strength_mult": self.ob_strength_mult,
            "rr_ratio": self.rr_ratio,
            "sl_atr_buffer": self.sl_atr_buffer,
            "max_ob_age_candles": self.max_ob_age_candles,
            "trend_ema_period": self.trend_ema_period,
            "min_atr_threshold": self.min_atr_threshold,
        }

    # -------------------------------------------------------------------------
    # Indicators
    # -------------------------------------------------------------------------

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute OB detection columns, ATR, and trend EMA on the full DataFrame.
        """
        df = df.copy()

        # --- ATR (Wilder's) ---
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = true_range.ewm(
            alpha=1 / self.atr_length, min_periods=self.atr_length, adjust=False
        ).mean()

        # --- Trend EMA ---
        if self.trend_ema_period > 0:
            df["trend_ema"] = (
                df["close"]
                .ewm(
                    span=self.trend_ema_period,
                    min_periods=self.trend_ema_period,
                    adjust=False,
                )
                .mean()
            )
        else:
            df["trend_ema"] = float("nan")  # disabled - get_signal will skip filter

        # --- Average body size (ob_strength_period SMA) ---
        body_size = (df["close"] - df["open"]).abs()
        avg_body = body_size.rolling(self.ob_strength_period).mean()

        curr_body = body_size
        prev_bearish = df["close"].shift(1) < df["open"].shift(1)
        prev_bullish = df["close"].shift(1) > df["open"].shift(1)
        strong_move = curr_body >= self.ob_strength_mult * avg_body.shift(1)

        # --- Bullish OB ---
        bull_ob = prev_bearish & strong_move & (df["close"] > df["open"])
        df["ob_bull"] = bull_ob
        df["ob_bull_high"] = df["high"].shift(1).where(bull_ob)
        df["ob_bull_low"] = df["low"].shift(1).where(bull_ob)

        # --- Bearish OB ---
        bear_ob = prev_bullish & strong_move & (df["close"] < df["open"])
        df["ob_bear"] = bear_ob
        df["ob_bear_high"] = df["high"].shift(1).where(bear_ob)
        df["ob_bear_low"] = df["low"].shift(1).where(bear_ob)

        # --- Diagnostics ---
        bull_count = int(df["ob_bull"].sum())
        bear_count = int(df["ob_bear"].sum())
        logger.info(
            f"[OB Indicators] Computed on {len(df)} candles | "
            f"Bullish OBs: {bull_count} | Bearish OBs: {bear_count} | "
            f"ob_strength_mult={self.ob_strength_mult} | "
            f"ob_strength_period={self.ob_strength_period} | "
            f"trend_ema_period={self.trend_ema_period}"
        )

        if bull_count == 0 and bear_count == 0:
            logger.warning(
                f"[OB Indicators] ZERO Order Blocks detected. "
                f"Try reducing --ob-strength-mult (current={self.ob_strength_mult}). "
                f"Sample body size range: "
                f"min={body_size.min():.6f} max={body_size.max():.6f}"
            )

        return df

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._compute_indicators(df)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._compute_indicators(df)

    def get_min_candles(self) -> int:
        periods = [self.ob_strength_period, self.atr_length]
        if self.trend_ema_period > 0:
            periods.append(self.trend_ema_period)
        return max(periods) + 2

    # -------------------------------------------------------------------------
    # Entry signal
    # -------------------------------------------------------------------------

    def get_signal(self, candle: dict, indicators: dict) -> str | None:
        """
        Order Block entry with retracement confirmation, EMA trend filter,
        and minimum ATR threshold filter.
        """
        self._diag_candles_processed += 1
        self._candle_index += 1

        if self._in_position:
            return None

        # --- Required indicator keys ---
        ob_bull = indicators.get("ob_bull")
        ob_bear = indicators.get("ob_bear")
        ob_bull_high = indicators.get("ob_bull_high")
        ob_bull_low = indicators.get("ob_bull_low")
        ob_bear_high = indicators.get("ob_bear_high")
        ob_bear_low = indicators.get("ob_bear_low")
        atr = indicators.get("atr")
        trend_ema = indicators.get("trend_ema")

        if ob_bull is None or ob_bear is None:
            self._diag_nan_skipped += 1
            if self._diag_nan_skipped == 1:
                logger.error(
                    "[OB get_signal] 'ob_bull' or 'ob_bear' key is MISSING from "
                    "indicators dict. Keys received: " + str(list(indicators.keys()))
                )
            return None

        if pd.isna(ob_bull) or pd.isna(ob_bear):
            self._diag_nan_skipped += 1
            return None

        if atr is None or pd.isna(atr):
            self._diag_nan_skipped += 1
            return None

        atr = float(atr)

        # --- Minimum ATR threshold filter ---
        if self.min_atr_threshold > 0 and atr < self.min_atr_threshold:
            self._diag_atr_filtered += 1
            return None

        # --- Trend EMA - NaN is treated as filter disabled for that candle ---
        trend_ema_valid = trend_ema is not None and not pd.isna(trend_ema)
        trend_ema_val = float(trend_ema) if trend_ema_valid else None

        # Log diagnostic summary every 500 candles
        if self._diag_candles_processed % 500 == 0:
            logger.info(
                f"[OB Diag @{self._diag_candles_processed}] "
                f"bull_detected={self._diag_ob_bull_detected} "
                f"bear_detected={self._diag_ob_bear_detected} "
                f"nan_skipped={self._diag_nan_skipped} "
                f"retracement_checks={self._diag_retracement_checks} "
                f"entries_fired={self._diag_entries_fired} "
                f"trend_filtered={self._diag_trend_filtered} "
                f"atr_filtered={self._diag_atr_filtered} "
                f"pending={self._pending_ob_bias}"
            )

        close = candle["close"]

        # ------------------------------------------------------------------
        # Step 1: Age out or invalidate existing pending OB
        # ------------------------------------------------------------------
        if self._pending_ob_bias is not None:
            self._pending_ob_age += 1

            if self._pending_ob_age > self.max_ob_age_candles:
                logger.debug(
                    f"[OB] Pending {self._pending_ob_bias} OB expired after "
                    f"{self._pending_ob_age} candles. "
                    f"Zone=[{self._pending_ob_low:.2f}, {self._pending_ob_high:.2f}]"
                )
                self._clear_pending_ob()

            elif self._pending_ob_bias == "long" and close < self._pending_ob_low:
                logger.debug(
                    f"[OB] Bullish OB invalidated: "
                    f"close={close:.2f} < ob_low={self._pending_ob_low:.2f}"
                )
                self._clear_pending_ob()

            elif self._pending_ob_bias == "short" and close > self._pending_ob_high:
                logger.debug(
                    f"[OB] Bearish OB invalidated: "
                    f"close={close:.2f} > ob_high={self._pending_ob_high:.2f}"
                )
                self._clear_pending_ob()

        # ------------------------------------------------------------------
        # Step 2: Check retracement into pending OB zone
        # ------------------------------------------------------------------
        if self._pending_ob_bias is not None:
            self._diag_retracement_checks += 1
            ob_h = self._pending_ob_high
            ob_l = self._pending_ob_low

            if self._pending_ob_bias == "long" and ob_l <= close <= ob_h:
                # EMA trend filter - only take longs above trend EMA
                if trend_ema_val is not None and close < trend_ema_val:
                    logger.debug(
                        f"[OB] BUY filtered by EMA trend: "
                        f"close={close:.2f} < trend_ema={trend_ema_val:.2f}"
                    )
                    self._diag_trend_filtered += 1
                    self._clear_pending_ob()
                    return None

                sl_price = ob_l - self.sl_atr_buffer * atr
                tp_price = close + self.rr_ratio * (close - sl_price)

                # FIX 1: Corrected log message from "SELL entry" to "BUY entry"
                logger.info(
                    f"[OB] BUY entry: close={close:.2f} inside "
                    f"bullish zone=[{ob_l:.2f}, {ob_h:.2f}] | "
                    f"SL={sl_price:.2f} TP={tp_price:.2f} ATR={atr:.4f} "
                    f"EMA={f'{trend_ema_val:.2f}' if trend_ema_val is not None else 'N/A'}"
                )

                self._diag_entries_fired += 1

                self._last_sl_price = sl_price
                self._last_tp_price = tp_price
                self._last_atr_at_entry = atr
                self._last_ob_high = ob_h
                self._last_ob_low = ob_l

                candle["_ob_sl_price"] = sl_price
                candle["_ob_tp_price"] = tp_price
                candle["_ob_atr_at_entry"] = atr
                candle["_ob_high"] = ob_h
                candle["_ob_low"] = ob_l

                self._clear_pending_ob()
                return "buy"

            elif self._pending_ob_bias == "short" and ob_l <= close <= ob_h:
                # EMA trend filter - only take shorts below trend EMA
                if trend_ema_val is not None and close > trend_ema_val:
                    logger.debug(
                        f"[OB] SELL filtered by EMA trend: "
                        f"close={close:.2f} > trend_ema={trend_ema_val:.2f}"
                    )
                    self._diag_trend_filtered += 1
                    self._clear_pending_ob()
                    return None

                sl_price = ob_h + self.sl_atr_buffer * atr
                tp_price = close - self.rr_ratio * (sl_price - close)

                logger.info(
                    f"[OB] SELL entry: close={close:.2f} inside "
                    f"bearish zone=[{ob_l:.2f}, {ob_h:.2f}] | "
                    f"SL={sl_price:.2f} TP={tp_price:.2f} ATR={atr:.4f} "
                    f"EMA={f'{trend_ema_val:.2f}' if trend_ema_val is not None else 'N/A'}"
                )

                self._diag_entries_fired += 1

                self._last_sl_price = sl_price
                self._last_tp_price = tp_price
                self._last_atr_at_entry = atr
                self._last_ob_high = ob_h
                self._last_ob_low = ob_l

                candle["_ob_sl_price"] = sl_price
                candle["_ob_tp_price"] = tp_price
                candle["_ob_atr_at_entry"] = atr
                candle["_ob_high"] = ob_h
                candle["_ob_low"] = ob_l

                self._clear_pending_ob()
                return "sell"

        # ------------------------------------------------------------------
        # Step 3: Detect new OB on this candle
        # FIX 3: Changed second `if` to `elif` to prevent bear OB from
        #        silently overwriting bull OB on the same candle
        # ------------------------------------------------------------------
        if bool(ob_bull) and ob_bull_high is not None and not pd.isna(ob_bull_high):
            self._diag_ob_bull_detected += 1
            logger.debug(
                f"[OB] New Bullish OB: zone=[{float(ob_bull_low):.2f}, "
                f"{float(ob_bull_high):.2f}] candle={self._candle_index}"
            )
            self._set_pending_ob("long", float(ob_bull_high), float(ob_bull_low))

        elif (
            bool(ob_bear) and ob_bear_high is not None and not pd.isna(ob_bear_high)
        ):  # FIX 3
            self._diag_ob_bear_detected += 1
            logger.debug(
                f"[OB] New Bearish OB: zone=[{float(ob_bear_low):.2f}, "
                f"{float(ob_bear_high):.2f}] candle={self._candle_index}"
            )
            self._set_pending_ob("short", float(ob_bear_high), float(ob_bear_low))

        return None

    # -------------------------------------------------------------------------
    # State reset
    # -------------------------------------------------------------------------

    def reset_state(self) -> None:
        self._clear_pending_ob()
        self._clear_last_entry_levels()
        self._in_position = False
        self._save_state_value("ob_in_position", "false")  # FIX 5
        self._candle_index = 0
        self._diag_candles_processed = 0
        self._diag_ob_bull_detected = 0
        self._diag_ob_bear_detected = 0
        self._diag_nan_skipped = 0
        self._diag_retracement_checks = 0
        self._diag_entries_fired = 0
        self._diag_trend_filtered = 0
        self._diag_atr_filtered = 0
        self._save_state_value("ob_candle_index", "0")
        logger.info("[OB] State fully reset. All pending zones and counters cleared.")

    # -------------------------------------------------------------------------
    # Exit signal
    # FIX 4: Both TP and SL now use close-based price reference for consistency
    # -------------------------------------------------------------------------

    def get_exit_signal(
        self,
        candle: dict,
        indicators: dict,
        position: dict,
    ) -> tuple[str | None, str]:
        side = position["side"]
        entry_price = position["entry_price"]
        best_price = position.get("best_price", entry_price)
        sl_price = position.get("sl_price")
        tp_price = position.get("tp_price")
        atr = position.get("atr_at_entry")

        high = candle["high"]
        low = candle["low"]
        close = candle["close"]

        if sl_price is None or tp_price is None or atr is None:
            atr_live = float(indicators.get("atr") or 0)
            if atr_live == 0:
                logger.error(
                    f"[OB get_exit_signal] sl_price/tp_price/atr all missing AND "
                    f"live ATR=0. Cannot compute exits. Holding. "
                    f"entry={entry_price} side={side}"
                )
                return None, ""

            logger.warning(
                f"[OB get_exit_signal] sl_price/tp_price/atr_at_entry missing from "
                f"position dict. Falling back to ATR-based defaults. "
                f"entry={entry_price} side={side}"
            )
            atr = atr_live
            if side == "buy":
                sl_price = entry_price - 2 * atr
                tp_price = entry_price + self.rr_ratio * 2 * atr
            else:
                sl_price = entry_price + 2 * atr
                tp_price = entry_price - self.rr_ratio * 2 * atr

        sl_price = float(sl_price)
        tp_price = float(tp_price)
        atr = float(atr)

        trail_distance = self.atr_multiplier * atr
        activation_delta = self.trail_activation_atr * atr

        if side == "buy":
            new_best = max(best_price, high)
            position["best_price"] = new_best
            trail_activated = new_best >= entry_price + activation_delta

            # FIX 4: Both TP and SL now use close for consistency
            # Ambiguous case is now theoretically impossible (close cannot be
            # both >= tp and <= sl unless tp <= sl which is a config error),
            # but guard is kept for safety
            if close >= tp_price and close <= sl_price:
                logger.warning(
                    f"[OB] Ambiguous exit on BUY: close={close} >= tp={tp_price:.4f} "
                    f"AND close={close} <= sl={sl_price:.4f}. TP wins. entry={entry_price}"
                )

            if close >= tp_price:
                return "sell", "EXIT_TP"
            if close <= sl_price:
                return "sell", "EXIT_SL"
            if trail_activated:
                trail_stop = new_best - trail_distance
                if low <= trail_stop:
                    return "sell", "EXIT_TRAIL"

        elif side == "sell":
            new_best = min(best_price, low)
            position["best_price"] = new_best
            trail_activated = new_best <= entry_price - activation_delta

            # FIX 4: Both TP and SL now use close for consistency
            if close <= tp_price and close >= sl_price:
                logger.warning(
                    f"[OB] Ambiguous exit on SELL: close={close} <= tp={tp_price:.4f} "
                    f"AND close={close} >= sl={sl_price:.4f}. TP wins. entry={entry_price}"
                )

            if close <= tp_price:
                return "buy", "EXIT_TP"
            if close >= sl_price:
                return "buy", "EXIT_SL"
            if trail_activated:
                trail_stop = new_best + trail_distance
                if high >= trail_stop:
                    return "buy", "EXIT_TRAIL"

        return None, ""
