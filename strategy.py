# =============================================================================
# strategy.py - EMA Crossover Signal Logic
# =============================================================================

import pandas as pd
from logger import get_logger

from config import EMA_FAST, EMA_SLOW

logger = get_logger(__name__)


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Compute Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def get_signal(df: pd.DataFrame):
    """
    Determine trading signal based on EMA crossover.

    Uses last 2 confirmed CLOSED candles (index -2 and -3) to avoid
    acting on an incomplete/forming candle.

    Returns:
        "BUY"  - fast EMA crossed above slow EMA
        "SELL" - fast EMA crossed below slow EMA
        None   - no crossover detected
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame passed to get_signal.")
        return None

    min_required = EMA_SLOW + 5
    if len(df) < min_required:
        logger.warning(f"Not enough candles: {len(df)} < {min_required}")
        return None

    df = df.copy()
    df["ema_fast"] = compute_ema(df["close"], EMA_FAST)
    df["ema_slow"] = compute_ema(df["close"], EMA_SLOW)

    # Use confirmed closed candles: -3 (previous) and -2 (last closed)
    # -1 is the currently forming candle — excluded
    prev_fast = df["ema_fast"].iloc[-3]
    prev_slow = df["ema_slow"].iloc[-3]
    curr_fast = df["ema_fast"].iloc[-2]
    curr_slow = df["ema_slow"].iloc[-2]

    logger.debug(
        f"EMA Check | prev_fast={prev_fast:.2f} prev_slow={prev_slow:.2f} "
        f"curr_fast={curr_fast:.2f} curr_slow={curr_slow:.2f}"
    )

    # Bullish crossover: fast crosses above slow
    if prev_fast <= prev_slow and curr_fast > curr_slow:
        logger.info("BUY signal detected (EMA fast crossed above slow).")
        return "BUY"

    # Bearish crossover: fast crosses below slow
    if prev_fast >= prev_slow and curr_fast < curr_slow:
        logger.info("SELL signal detected (EMA fast crossed below slow).")
        return "SELL"

    return None
