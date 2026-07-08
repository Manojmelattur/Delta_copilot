# =============================================================================
# bollinger_bands.py - Bollinger Bands Mean Reversion Strategy (v5 Fixes)
# =============================================================================

import logging
import os
import sqlite3

import pandas as pd

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Mean Reversion Strategy (v5 Fixes).

    Entry Logic (strict 2-candle confirmation with volume filter on trigger candle only):
      BUY  : Candle N closes BELOW lower band AND volume >= volume_multiplier * volume_avg
             → wait for candle N+1 (and ONLY N+1) to close back ABOVE lower band.
             The confirmation candle (N+1) does NOT require elevated volume.
      SELL : Candle N closes ABOVE upper band AND volume >= volume_multiplier * volume_avg
             → wait for candle N+1 (and ONLY N+1) to close back BELOW upper band.
             The confirmation candle (N+1) does NOT require elevated volume.

    Exit Logic (priority order):
      1. Fixed TP         : Exit when price moves take_profit_pct in favor.
                            If trailing stop has already activated (based on new_best
                            for the current candle), TP is extended by tp_extension_pct.
      2. Hard SL          : Exit when price moves stop_loss_pct against entry.
      3. Trailing Stop    : Activates after trail_activation_pct move in favor,
                            trails by trail_pct from best price seen.
      4. Band Flip        : Exit if price crosses to opposite band (reversal).

    Fixes vs v4:
      - [#3] Strict N+1 confirmation: pending signal expires if not confirmed on
             the very next candle. Candle index tracked and persisted to DB.
      - [#4] Explicit NaN guard on bb_upper/bb_lower in both get_signal and
             get_exit_signal. Silent failures during warmup are now prevented.
      - [#5] new_best computed at top of get_exit_signal before any exit checks,
             so TP extension and trailing stop activate on the same candle.
      - [#1] Ambiguous TP/SL same-candle condition is logged as a warning for
             backtest review. TP wins by convention (unchanged behavior).

    State Persistence:
      _pending_signal and _trigger_candle_index are persisted to SQLite so that
      a mid-confirmation restart does not silently lose pending state.
    """

    def __init__(
        self,
        period: int = 20,
        std: float = 2.2,
        trail_activation_pct: float = 1.0,
        trail_pct: float = 0.8,
        stop_loss_pct: float = 1.0,
        take_profit_pct: float = 1.2,
        tp_extension_pct: float = 0.3,
        volume_period: int = 20,
        volume_multiplier: float = 1.2,
        db_path: str = "strategy_state.db",
    ):
        self.period = period
        self.std = std
        self.trail_activation_pct = trail_activation_pct / 100.0
        self.trail_pct = trail_pct / 100.0
        self.stop_loss_pct = stop_loss_pct / 100.0
        self.take_profit_pct = take_profit_pct / 100.0
        self.tp_extension_pct = tp_extension_pct / 100.0
        self.volume_period = volume_period
        self.volume_multiplier = volume_multiplier
        self.db_path = db_path

        # Initialise DB and restore state
        self._init_db()
        self._pending_signal = self._load_state_value("pending_signal")
        self._trigger_candle_index = int(
            self._load_state_value("trigger_candle_index") or -1
        )
        self._candle_index = int(self._load_state_value("candle_index") or 0)

        if self._pending_signal is not None:
            logger.warning(
                f"Restored pending signal '{self._pending_signal}' from database "
                f"(trigger_candle_index={self._trigger_candle_index}, "
                f"candle_index={self._candle_index}). "
                f"Bot may have restarted mid-confirmation."
            )

    def reset_state(self) -> None:
        """
        Reset all in-memory and persisted signal state.
        Must be called at the start of every backtest run to prevent
        stale state from a previous live bot session affecting results.
        """
        self._pending_signal = None
        self._trigger_candle_index = -1
        self._candle_index = 0
        self._save_signal_state(None, -1, 0)
        logger.debug("BollingerBandsStrategy state reset.")

    # -------------------------------------------------------------------------
    # State persistence helpers
    # -------------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create the state table if it does not already exist."""
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
        """Load a single value from the state table. Returns None if not set."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT value FROM strategy_state WHERE key = ?", (key,)
            ).fetchone()
        if row is None or row[0] == "null":
            return None
        return row[0]

    def _save_state_value(self, key: str, value: str | None) -> None:
        """Persist a single key/value pair to the state table."""
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

    def _save_signal_state(
        self,
        pending_signal: str | None,
        trigger_candle_index: int,
        candle_index: int,
    ) -> None:
        """
        Atomically persist all three signal-tracking fields in a single
        transaction to prevent partial state on restart.
        """
        with sqlite3.connect(self.db_path) as conn:
            for key, value in (
                (
                    "pending_signal",
                    pending_signal if pending_signal is not None else "null",
                ),
                ("trigger_candle_index", str(trigger_candle_index)),
                ("candle_index", str(candle_index)),
            ):
                conn.execute(
                    """
                    INSERT INTO strategy_state (key, value)
                    VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    (key, value),
                )
            conn.commit()

    def _set_pending_signal(self, value: str | None) -> None:
        """Set and persist _pending_signal. Candle index fields saved atomically."""
        self._pending_signal = value
        self._save_signal_state(value, self._trigger_candle_index, self._candle_index)

    # -------------------------------------------------------------------------
    # Strategy metadata
    # -------------------------------------------------------------------------
    # @property
    def name(self):
        return "Bollinger Bands Strategy"

    # def name(self) -> str:
    #     return (
    #         f"Bollinger Bands(period={self.period}, std={self.std}, "
    #         f"trail_activation_pct={self.trail_activation_pct * 100}, "
    #         f"trail_pct={self.trail_pct * 100}, "
    #         f"stop_loss_pct={self.stop_loss_pct * 100}, "
    #         f"take_profit_pct={self.take_profit_pct * 100}, "
    #         f"tp_extension_pct={self.tp_extension_pct * 100}, "
    #         f"volume_period={self.volume_period}, "
    #         f"volume_multiplier={self.volume_multiplier})"
    #     )

    def get_params(self) -> dict:
        return {
            "period": self.period,
            "std": self.std,
            "trail_activation_pct": self.trail_activation_pct * 100,
            "trail_pct": self.trail_pct * 100,
            "stop_loss_pct": self.stop_loss_pct * 100,
            "take_profit_pct": self.take_profit_pct * 100,
            "tp_extension_pct": self.tp_extension_pct * 100,
            "volume_period": self.volume_period,
            "volume_multiplier": self.volume_multiplier,
        }

    # -------------------------------------------------------------------------
    # Indicators (single implementation shared by live loop and backtester)
    # -------------------------------------------------------------------------

    # def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Core indicator computation. Used by both compute_indicators (live)
    #     and calculate_indicators (backtester) to avoid duplication.
    #     """
    #     df = df.copy()
    #     df["bb_mid"] = df["close"].rolling(self.period).mean()
    #     df["bb_std"] = df["close"].rolling(self.period).std()
    #     df["bb_upper"] = df["bb_mid"] + self.std * df["bb_std"]
    #     df["bb_lower"] = df["bb_mid"] - self.std * df["bb_std"]
    #     df["volume_avg"] = df["volume"].rolling(self.volume_period).mean()
    #     return df

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Core indicator computation. Used by both compute_indicators (live)
        and calculate_indicators (backtester) to avoid duplication.
        """
        df = df.copy()
        df["bb_mid"] = df["close"].rolling(self.period).mean()
        df["bb_std"] = (
            df["close"].rolling(self.period).std(ddof=0)
        )  # population std, matches TradingView
        df["bb_upper"] = df["bb_mid"] + self.std * df["bb_std"]
        df["bb_lower"] = df["bb_mid"] - self.std * df["bb_std"]
        df["volume_avg"] = df["volume"].rolling(self.volume_period).mean()
        return df

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Called by the live trading loop."""
        return self._compute_indicators(df)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Called by the backtester."""
        return self._compute_indicators(df)

    # -------------------------------------------------------------------------
    # Entry signal
    # -------------------------------------------------------------------------

    def name(self) -> str:
        return "Bollinger Bands"

    def get_signal(self, candle: dict, indicators: dict) -> str | None:
        """
        Strict 2-candle confirmation logic with volume filter on trigger candle only.

        Fix #4: NaN guard - returns None immediately if bb_upper or bb_lower
                is NaN (warmup period not yet complete).

        Fix #3: Candle index tracker - pending signal expires if not confirmed
                on the candle immediately following the trigger (N+1 only).
                Re-queuing the same direction on a failed confirmation is
                prevented unless a fresh volume-confirmed trigger fires.

        Step 1 (trigger candle N):
          - Volume must be >= volume_multiplier * volume_avg
          - If close < bb_lower  → set _pending_signal = 'long',  return None
          - If close > bb_upper  → set _pending_signal = 'short', return None

        Step 2 (confirmation candle N+1 ONLY):
          - Pending long  + close > bb_lower  → confirm BUY,  clear pending
          - Pending short + close < bb_upper  → confirm SELL, clear pending
          - Not N+1, or no confirmation       → clear pending, return None

        Conflict guard:
          If the confirmation candle simultaneously confirms the pending signal
          AND triggers a new signal in the opposite direction, the opposing
          trigger is discarded. The confirmed entry is returned and pending is
          cleared.
        """
        # Fix #4: NaN guard
        bb_upper = indicators.get("bb_upper")
        bb_lower = indicators.get("bb_lower")
        volume_avg = indicators.get("volume_avg")

        if bb_upper is None or bb_lower is None:
            return None
        if pd.isna(bb_upper) or pd.isna(bb_lower):
            return None

        close = candle["close"]
        volume = candle["volume"]

        # Fix #3: Advance candle index at the start of every call
        self._candle_index += 1

        # Volume filter: applies to trigger candles only
        volume_confirmed = (
            volume_avg is not None
            and not pd.isna(volume_avg)
            and volume >= self.volume_multiplier * volume_avg
        )

        # Evaluate new trigger on this candle (volume-gated)
        new_trigger = None
        if volume_confirmed:
            if close < bb_lower:
                new_trigger = "long"
            elif close > bb_upper:
                new_trigger = "short"

        # --- Confirmation check (previous candle set a pending signal) ---
        if self._pending_signal is not None:
            # Fix #3: Only allow confirmation on the candle immediately after trigger
            is_next_candle = self._candle_index == self._trigger_candle_index + 1

            if not is_next_candle:
                # Pending signal has expired - start fresh with any new trigger
                logger.debug(
                    f"Pending signal '{self._pending_signal}' expired at candle "
                    f"{self._candle_index} (triggered at {self._trigger_candle_index})."
                )
                self._trigger_candle_index = self._candle_index if new_trigger else -1
                self._set_pending_signal(new_trigger)
                return None

            if self._pending_signal == "long":
                if close > bb_lower:
                    # Confirmed long entry.
                    # Conflict guard: discard opposing short trigger on same candle.
                    carry = new_trigger if new_trigger != "short" else None
                    carry_index = self._candle_index if carry else -1
                    self._trigger_candle_index = carry_index
                    self._set_pending_signal(carry)
                    return "buy"
                else:
                    # Confirmation failed - clear pending, record any fresh trigger.
                    self._trigger_candle_index = (
                        self._candle_index if new_trigger else -1
                    )
                    self._set_pending_signal(new_trigger)
                    return None

            if self._pending_signal == "short":
                if close < bb_upper:
                    # Confirmed short entry.
                    # Conflict guard: discard opposing long trigger on same candle.
                    carry = new_trigger if new_trigger != "long" else None
                    carry_index = self._candle_index if carry else -1
                    self._trigger_candle_index = carry_index
                    self._set_pending_signal(carry)
                    return "sell"
                else:
                    # Confirmation failed - clear pending, record any fresh trigger.
                    self._trigger_candle_index = (
                        self._candle_index if new_trigger else -1
                    )
                    self._set_pending_signal(new_trigger)
                    return None

        # --- No pending signal: record trigger for next candle ---
        if new_trigger:
            self._trigger_candle_index = self._candle_index
        self._set_pending_signal(new_trigger)
        return None

    def get_min_candles(self) -> int:
        """Minimum candles needed before strategy can generate signals."""
        return max(self.period, self.volume_period) + 1

    # -------------------------------------------------------------------------
    # Exit signal
    # -------------------------------------------------------------------------

    def get_exit_signal(
        self,
        candle: dict,
        indicators: dict,
        position: dict,
    ) -> tuple[str | None, str]:
        """
        Exit logic (priority order):
          1. Fixed TP  : Exit at take_profit_pct. TP is extended by tp_extension_pct
                         if trail has activated THIS candle (uses new_best, not best_price).
          2. Hard SL   : Exit at stop_loss_pct against entry.
          3. Trailing  : Activates after trail_activation_pct, trails by trail_pct.
          4. Band Flip : Exit if close crosses to the opposite band.

        Fix #4: NaN guard on bb_upper/bb_lower.
        Fix #5: new_best computed at top before any exit checks so TP extension
                and trailing stop activate on the same candle (no one-candle lag).
        Fix #1: Ambiguous same-candle TP+SL condition is logged as a warning.
                TP wins by convention (behavior unchanged).

        Returns (side_to_exit, exit_reason) or (None, '').
        """
        # Fix #4: NaN guard
        bb_upper = indicators.get("bb_upper")
        bb_lower = indicators.get("bb_lower")

        if bb_upper is None or bb_lower is None:
            return None, ""
        if pd.isna(bb_upper) or pd.isna(bb_lower):
            return None, ""

        side = position["side"]
        entry_price = position["entry_price"]
        best_price = position.get("best_price", entry_price)

        high = candle["high"]
        low = candle["low"]
        close = candle["close"]

        if side == "buy":
            # Fix #5: Compute new_best FIRST so trail_activated is current-candle accurate
            new_best = max(best_price, high)
            position["best_price"] = new_best
            trail_activated = new_best >= entry_price * (1 + self.trail_activation_pct)

            # ---- Take Profit ----
            tp_price = (
                entry_price * (1 + self.take_profit_pct + self.tp_extension_pct)
                if trail_activated
                else entry_price * (1 + self.take_profit_pct)
            )
            sl_price = entry_price * (1 - self.stop_loss_pct)

            # Fix #1: Log ambiguous same-candle TP+SL condition
            if high >= tp_price and low <= sl_price:
                logger.warning(
                    f"Ambiguous exit on BUY: high={high} >= tp={tp_price:.4f} AND "
                    f"low={low} <= sl={sl_price:.4f}. TP wins by convention. "
                    f"entry={entry_price}, candle_time={candle.get('time', 'unknown')}"
                )

            if high >= tp_price:
                return "sell", "EXIT_TP"

            if low <= sl_price:
                return "sell", "EXIT_SL"

            # ---- Trailing Stop ----
            if trail_activated:
                trail_stop = new_best * (1 - self.trail_pct)
                if low <= trail_stop:
                    return "sell", "EXIT_TRAIL"

            # ---- Band Flip ----
            if close >= bb_upper:
                return "sell", "EXIT_BAND_FLIP"

        elif side == "sell":
            # Fix #5: Compute new_best FIRST
            new_best = min(best_price, low)
            position["best_price"] = new_best
            trail_activated = new_best <= entry_price * (1 - self.trail_activation_pct)

            # ---- Take Profit ----
            tp_price = (
                entry_price * (1 - self.take_profit_pct - self.tp_extension_pct)
                if trail_activated
                else entry_price * (1 - self.take_profit_pct)
            )
            sl_price = entry_price * (1 + self.stop_loss_pct)

            # Fix #1: Log ambiguous same-candle TP+SL condition
            if low <= tp_price and high >= sl_price:
                logger.warning(
                    f"Ambiguous exit on SELL: low={low} <= tp={tp_price:.4f} AND "
                    f"high={high} >= sl={sl_price:.4f}. TP wins by convention. "
                    f"entry={entry_price}, candle_time={candle.get('time', 'unknown')}"
                )

            if low <= tp_price:
                return "buy", "EXIT_TP"

            if high >= sl_price:
                return "buy", "EXIT_SL"

            # ---- Trailing Stop ----
            if trail_activated:
                trail_stop = new_best * (1 + self.trail_pct)
                if high >= trail_stop:
                    return "buy", "EXIT_TRAIL"

            # ---- Band Flip ----
            if close <= bb_lower:
                return "buy", "EXIT_BAND_FLIP"

        return None, ""
