# =============================================================================
# fvg_strategy.py - Fair Value Gap (FVG) Strategy
# =============================================================================

import logging
import sqlite3

import pandas as pd

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class FVGStrategy(BaseStrategy):
    """
    Fair Value Gap (FVG) Mean Reversion / Trend Continuation Strategy.
    """

    def __init__(
        self,
        min_fvg_size_pct: float = 0.05,
        max_fvg_age_candles: int = 50,
        trail_activation_pct: float = 0.8,
        trail_pct: float = 0.5,
        stop_loss_pct: float = 1.0,
        take_profit_pct: float = 1.5,
        tp_extension_pct: float = 0.3,
        db_path: str = "strategy_state.db",
    ):
        # NOTE: all pct params divided by 100 here - stored as decimals throughout
        self.min_fvg_size_pct = min_fvg_size_pct / 100.0  # 0.05% → 0.0005
        self.max_fvg_age_candles = max_fvg_age_candles
        self.trail_activation_pct = trail_activation_pct / 100.0
        self.trail_pct = trail_pct / 100.0
        self.stop_loss_pct = stop_loss_pct / 100.0
        self.take_profit_pct = take_profit_pct / 100.0
        self.tp_extension_pct = tp_extension_pct / 100.0
        self.db_path = db_path

        # Position tracking - prevents get_signal() firing while in a trade
        self._in_position = False

        # Diagnostic counters (reset each backtest run, not persisted)
        self._diag_candles_processed = 0
        self._diag_fvg_bull_detected = 0
        self._diag_fvg_bear_detected = 0
        self._diag_nan_skipped = 0
        self._diag_retracement_checks = 0
        self._diag_entries_fired = 0

        self._init_db()
        self._pending_fvg_bias = self._load_state_value("fvg_pending_bias")
        self._pending_fvg_high = self._load_float("fvg_pending_high")
        self._pending_fvg_low = self._load_float("fvg_pending_low")
        self._pending_fvg_age = int(self._load_state_value("fvg_pending_age") or 0)
        self._candle_index = int(self._load_state_value("fvg_candle_index") or 0)

        if self._pending_fvg_bias is not None:
            logger.warning(
                f"Restored pending FVG '{self._pending_fvg_bias}' from database "
                f"(zone=[{self._pending_fvg_low}, {self._pending_fvg_high}], "
                f"age={self._pending_fvg_age}). Bot may have restarted mid-detection."
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

    def _save_fvg_state(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            fields = {
                "fvg_pending_bias": self._pending_fvg_bias
                if self._pending_fvg_bias
                else "null",
                "fvg_pending_high": str(self._pending_fvg_high)
                if self._pending_fvg_high is not None
                else "null",
                "fvg_pending_low": str(self._pending_fvg_low)
                if self._pending_fvg_low is not None
                else "null",
                "fvg_pending_age": str(self._pending_fvg_age),
                "fvg_candle_index": str(self._candle_index),
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

    def _clear_pending_fvg(self) -> None:
        self._pending_fvg_bias = None
        self._pending_fvg_high = None
        self._pending_fvg_low = None
        self._pending_fvg_age = 0
        self._save_fvg_state()

    def _set_pending_fvg(self, bias: str, fvg_high: float, fvg_low: float) -> None:
        self._pending_fvg_bias = bias
        self._pending_fvg_high = fvg_high
        self._pending_fvg_low = fvg_low
        self._pending_fvg_age = 0
        self._save_fvg_state()

    # -------------------------------------------------------------------------
    # Position lifecycle hooks (called by backtester and live bot)
    # -------------------------------------------------------------------------

    def notify_entry(self, idx: int) -> None:
        """Called by backtester/live bot when a position is opened."""
        self._in_position = True

    def notify_exit(self) -> None:
        """Called by backtester/live bot when a position is closed."""
        self._in_position = False

    # -------------------------------------------------------------------------
    # Strategy metadata
    # -------------------------------------------------------------------------

    def name(self) -> str:
        return "FVG Strategy "

    def get_params(self) -> dict:
        return {
            "min_fvg_size_pct": self.min_fvg_size_pct * 100,
            "max_fvg_age_candles": self.max_fvg_age_candles,
            "trail_activation_pct": self.trail_activation_pct * 100,
            "trail_pct": self.trail_pct * 100,
            "stop_loss_pct": self.stop_loss_pct * 100,
            "take_profit_pct": self.take_profit_pct * 100,
            "tp_extension_pct": self.tp_extension_pct * 100,
        }

    # -------------------------------------------------------------------------
    # Indicators
    # -------------------------------------------------------------------------

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute FVG detection columns on the full DataFrame.

        Bullish FVG : df['low'].iloc[i] > df['high'].iloc[i-2]
                      → gap zone: [high[i-2], low[i]]

        Bearish FVG : df['high'].iloc[i] < df['low'].iloc[i-2]
                      → gap zone: [high[i], low[i-2]]

        Uses shift(2) which requires the FULL DataFrame - not row-by-row.
        """
        df = df.copy()

        # --- Bullish FVG ---
        bull_fvg_low = df["high"].shift(2)  # bottom of gap = candle[i-2].high
        bull_fvg_high = df["low"]  # top of gap    = candle[i].low
        bull_gap_pct = (bull_fvg_high - bull_fvg_low) / bull_fvg_low.replace(
            0, float("nan")
        )

        df["fvg_bull"] = (bull_fvg_high > bull_fvg_low) & (
            bull_gap_pct >= self.min_fvg_size_pct
        )
        df["fvg_bull_high"] = bull_fvg_high.where(df["fvg_bull"])
        df["fvg_bull_low"] = bull_fvg_low.where(df["fvg_bull"])

        # --- Bearish FVG ---
        bear_fvg_high = df["low"].shift(2)  # top of gap    = candle[i-2].low
        bear_fvg_low = df["high"]  # bottom of gap = candle[i].high
        bear_gap_pct = (bear_fvg_high - bear_fvg_low) / bear_fvg_low.replace(
            0, float("nan")
        )

        df["fvg_bear"] = (bear_fvg_high > bear_fvg_low) & (
            bear_gap_pct >= self.min_fvg_size_pct
        )
        df["fvg_bear_high"] = bear_fvg_high.where(df["fvg_bear"])
        df["fvg_bear_low"] = bear_fvg_low.where(df["fvg_bear"])

        # --- Diagnostic: log FVG counts after full computation ---
        bull_count = int(df["fvg_bull"].sum())
        bear_count = int(df["fvg_bear"].sum())
        logger.info(
            f"[FVG Indicators] Computed on {len(df)} candles | "
            f"Bullish FVGs: {bull_count} | Bearish FVGs: {bear_count} | "
            f"min_fvg_size_pct={self.min_fvg_size_pct * 100:.4f}%"
        )

        if bull_count == 0 and bear_count == 0:
            logger.warning(
                f"[FVG Indicators] ZERO FVGs detected. "
                f"Try reducing --fvg-min-size (current={self.min_fvg_size_pct * 100:.4f}%). "
                f"Sample bull_gap_pct range: "
                f"min={bull_gap_pct.min():.6f} max={bull_gap_pct.max():.6f}"
            )

        return df

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Called by the live trading loop."""
        return self._compute_indicators(df)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Called by the backtester."""
        return self._compute_indicators(df)

    def get_min_candles(self) -> int:
        return 3

    # -------------------------------------------------------------------------
    # Entry signal
    # -------------------------------------------------------------------------

    def get_signal(self, candle: dict, indicators: dict) -> str | None:
        """
        FVG entry with retracement confirmation and full diagnostics.

        Returns None immediately if a position is already open - all exit
        logic is handled exclusively by get_exit_signal(). This prevents
        signal_flip exits from firing on FVG detections mid-trade.
        """
        self._diag_candles_processed += 1
        self._candle_index += 1

        # ------------------------------------------------------------------
        # Guard: do not generate entries while a position is open
        # ------------------------------------------------------------------
        if self._in_position:
            return None

        # ------------------------------------------------------------------
        # NaN / missing key guard
        # ------------------------------------------------------------------
        fvg_bull = indicators.get("fvg_bull")
        fvg_bear = indicators.get("fvg_bear")
        fvg_bull_high = indicators.get("fvg_bull_high")
        fvg_bull_low = indicators.get("fvg_bull_low")
        fvg_bear_high = indicators.get("fvg_bear_high")
        fvg_bear_low = indicators.get("fvg_bear_low")

        # Detect if keys are missing entirely (backtester not passing all columns)
        if fvg_bull is None or fvg_bear is None:
            self._diag_nan_skipped += 1
            if self._diag_nan_skipped == 1:
                logger.error(
                    "[FVG get_signal] 'fvg_bull' or 'fvg_bear' key is MISSING from "
                    "indicators dict. The backtester may not be passing all DataFrame "
                    "columns. Keys received: " + str(list(indicators.keys()))
                )
            return None

        # Detect NaN (normal during warmup for first 2 candles)
        if pd.isna(fvg_bull) or pd.isna(fvg_bear):
            self._diag_nan_skipped += 1
            return None

        # Log diagnostic summary every 500 candles
        if self._diag_candles_processed % 500 == 0:
            logger.info(
                f"[FVG Diag @{self._diag_candles_processed}] "
                f"bull_detected={self._diag_fvg_bull_detected} "
                f"bear_detected={self._diag_fvg_bear_detected} "
                f"nan_skipped={self._diag_nan_skipped} "
                f"retracement_checks={self._diag_retracement_checks} "
                f"entries_fired={self._diag_entries_fired} "
                f"pending={self._pending_fvg_bias}"
            )

        close = candle["close"]

        # ------------------------------------------------------------------
        # Step 1: Age out or invalidate existing pending FVG
        # ------------------------------------------------------------------
        if self._pending_fvg_bias is not None:
            self._pending_fvg_age += 1

            if self._pending_fvg_age > self.max_fvg_age_candles:
                logger.debug(
                    f"[FVG] Pending {self._pending_fvg_bias} FVG expired after "
                    f"{self._pending_fvg_age} candles. "
                    f"Zone=[{self._pending_fvg_low:.2f}, {self._pending_fvg_high:.2f}]"
                )
                self._clear_pending_fvg()

            elif self._pending_fvg_bias == "long" and close < self._pending_fvg_low:
                logger.debug(
                    f"[FVG] Bullish FVG invalidated: "
                    f"close={close:.2f} < fvg_low={self._pending_fvg_low:.2f}"
                )
                self._clear_pending_fvg()

            elif self._pending_fvg_bias == "short" and close > self._pending_fvg_high:
                logger.debug(
                    f"[FVG] Bearish FVG invalidated: "
                    f"close={close:.2f} > fvg_high={self._pending_fvg_high:.2f}"
                )
                self._clear_pending_fvg()

        # ------------------------------------------------------------------
        # Step 2: Check retracement into pending FVG zone
        # ------------------------------------------------------------------
        if self._pending_fvg_bias is not None:
            self._diag_retracement_checks += 1
            fvg_h = self._pending_fvg_high
            fvg_l = self._pending_fvg_low

            if self._pending_fvg_bias == "long" and fvg_l <= close <= fvg_h:
                logger.info(
                    f"[FVG] BUY entry: close={close:.2f} inside "
                    f"bullish zone=[{fvg_l:.2f}, {fvg_h:.2f}]"
                )
                self._diag_entries_fired += 1
                self._clear_pending_fvg()
                return "buy"

            elif self._pending_fvg_bias == "short" and fvg_l <= close <= fvg_h:
                logger.info(
                    f"[FVG] SELL entry: close={close:.2f} inside "
                    f"bearish zone=[{fvg_l:.2f}, {fvg_h:.2f}]"
                )
                self._diag_entries_fired += 1
                self._clear_pending_fvg()
                return "sell"

        # ------------------------------------------------------------------
        # Step 3: Detect new FVG on this candle
        # ------------------------------------------------------------------
        if bool(fvg_bull) and fvg_bull_high is not None and not pd.isna(fvg_bull_high):
            self._diag_fvg_bull_detected += 1
            logger.debug(
                f"[FVG] New Bullish FVG: zone=[{float(fvg_bull_low):.2f}, "
                f"{float(fvg_bull_high):.2f}] candle={self._candle_index}"
            )
            self._set_pending_fvg("long", float(fvg_bull_high), float(fvg_bull_low))

        if bool(fvg_bear) and fvg_bear_high is not None and not pd.isna(fvg_bear_high):
            self._diag_fvg_bear_detected += 1
            logger.debug(
                f"[FVG] New Bearish FVG: zone=[{float(fvg_bear_low):.2f}, "
                f"{float(fvg_bear_high):.2f}] candle={self._candle_index}"
            )
            self._set_pending_fvg("short", float(fvg_bear_high), float(fvg_bear_low))

        # ------------------------------------------------------------------
        # End-of-run diagnostic summary (last candle heuristic)
        # ------------------------------------------------------------------
        if self._diag_candles_processed == 2879:
            logger.info(
                f"[FVG Final Diag] "
                f"total_candles={self._diag_candles_processed} | "
                f"nan_skipped={self._diag_nan_skipped} | "
                f"bull_fvgs_detected={self._diag_fvg_bull_detected} | "
                f"bear_fvgs_detected={self._diag_fvg_bear_detected} | "
                f"retracement_checks={self._diag_retracement_checks} | "
                f"entries_fired={self._diag_entries_fired}"
            )

        return None

        def reset_state(self) -> None:
            self._clear_pending_fvg()
            self._in_position = False
            self._candle_index = 0
            self._diag_candles_processed = 0
            self._diag_fvg_bull_detected = 0
            self._diag_fvg_bear_detected = 0
            self._diag_nan_skipped = 0
            self._diag_retracement_checks = 0
            self._diag_entries_fired = 0
            self._save_state_value("fvg_candle_index", "0")
            logger.info(
                "[FVG] State fully reset. All pending zones and counters cleared."
            )

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
        Manages all exit logic for open positions.

        Exit priority (highest to lowest):
          1. EXIT_TP  - take profit hit
          2. EXIT_SL  - hard stop loss hit
          3. EXIT_TRAIL - trailing stop hit (only after trail_activation_pct move)
          4. EXIT_FVG_FLIP - early exit at 80% of SL distance (structural close)

        All pct attributes are stored as decimals (e.g. 0.01 = 1%).
        Exit prices are computed here and also re-derived by _resolve_exit_price()
        in the backtester - both use the same decimal math so they stay in sync.
        """
        side = position["side"]
        entry_price = position["entry_price"]
        best_price = position.get("best_price", entry_price)

        high = candle["high"]
        low = candle["low"]
        close = candle["close"]

        if side == "buy":
            new_best = max(best_price, high)
            position["best_price"] = new_best
            trail_activated = new_best >= entry_price * (1 + self.trail_activation_pct)

            tp_price = (
                entry_price * (1 + self.take_profit_pct + self.tp_extension_pct)
                if trail_activated
                else entry_price * (1 + self.take_profit_pct)
            )
            sl_price = entry_price * (1 - self.stop_loss_pct)

            if high >= tp_price and low <= sl_price:
                logger.warning(
                    f"Ambiguous exit on BUY: high={high} >= tp={tp_price:.4f} AND "
                    f"low={low} <= sl={sl_price:.4f}. TP wins. entry={entry_price}"
                )

            if high >= tp_price:
                return "sell", "EXIT_TP"
            if low <= sl_price:
                return "sell", "EXIT_SL"
            if trail_activated:
                trail_stop = new_best * (1 - self.trail_pct)
                if low <= trail_stop:
                    return "sell", "EXIT_TRAIL"
            # EXIT_FVG_FLIP: structural early exit at 80% of SL distance
            # if close < entry_price * (1 - self.stop_loss_pct * 0.8):
            #     return "sell", "EXIT_FVG_FLIP"

        elif side == "sell":
            new_best = min(best_price, low)
            position["best_price"] = new_best
            trail_activated = new_best <= entry_price * (1 - self.trail_activation_pct)

            tp_price = (
                entry_price * (1 - self.take_profit_pct - self.tp_extension_pct)
                if trail_activated
                else entry_price * (1 - self.take_profit_pct)
            )
            sl_price = entry_price * (1 + self.stop_loss_pct)

            if low <= tp_price and high >= sl_price:
                logger.warning(
                    f"Ambiguous exit on SELL: low={low} <= tp={tp_price:.4f} AND "
                    f"high={high} >= sl={sl_price:.4f}. TP wins. entry={entry_price}"
                )

            if low <= tp_price:
                return "buy", "EXIT_TP"
            if high >= sl_price:
                return "buy", "EXIT_SL"
            if trail_activated:
                trail_stop = new_best * (1 + self.trail_pct)
                if high >= trail_stop:
                    return "buy", "EXIT_TRAIL"
            # EXIT_FVG_FLIP: structural early exit at 80% of SL distance
            # if close > entry_price * (1 + self.stop_loss_pct * 0.8):
            #     return "buy", "EXIT_FVG_FLIP"

        return None, ""
