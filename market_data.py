# =============================================================================
# market_data.py - Live Candle Fetcher + Product ID Resolver
# =============================================================================

import time

import pandas as pd
import requests

from config import BASE_URL, PRODUCT_ID, SYMBOL
from logger import get_logger

logger = get_logger(__name__)

RESOLUTION_MAP = {
    "5s": "5s",
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "12h": "12h",
    "1d": "1d",
    "1w": "1w",
}

# -----------------------------------------------------------------------------
# Session cache for product IDs — populated at runtime via API.
# No hardcoded IDs: production and testnet use different product IDs.
# -----------------------------------------------------------------------------
PRODUCT_ID_MAP = {}  # empty — resolved from BASE_URL (respects USE_TESTNET)


def get_product_id(symbol: str) -> int:
    """
    Resolve product_id for a given symbol.

    Resolution order:
      1. Session cache         — fast, no network call (populated after first lookup)
      2. Live API lookup       — uses BASE_URL from config (respects USE_TESTNET)
      3. Config PRODUCT_ID     — last-resort fallback

    Args:
        symbol: Trading symbol e.g. 'BTCUSD', 'ETHUSD'

    Returns:
        product_id as int
    """
    symbol = symbol.upper()

    # 1. Session cache lookup
    if symbol in PRODUCT_ID_MAP:
        pid = PRODUCT_ID_MAP[symbol]
        logger.debug(f"[{symbol}] product_id resolved from cache: {pid}")
        return pid

    # 2. Live API lookup — BASE_URL respects USE_TESTNET flag in config
    url = f"{BASE_URL}/v2/products/{symbol}"
    try:
        response = requests.get(
            url,
            headers={"Accept": "application/json"},
            timeout=(3, 27),
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            raise ValueError(f"API returned failure for symbol '{symbol}': {data}")

        pid = int(data["result"]["id"])
        logger.debug(f"[{symbol}] product_id resolved from API ({BASE_URL}): {pid}")

        # Cache for subsequent calls in this session
        PRODUCT_ID_MAP[symbol] = pid
        return pid

    except requests.exceptions.ConnectionError as e:
        logger.warning(f"[{symbol}] Connection error fetching product_id: {e}")
    except requests.exceptions.Timeout:
        logger.warning(f"[{symbol}] Timeout fetching product_id")
    except requests.exceptions.HTTPError as e:
        logger.warning(f"[{symbol}] HTTP error fetching product_id: {e}")
    except (KeyError, TypeError, ValueError) as e:
        logger.warning(f"[{symbol}] Unexpected response fetching product_id: {e}")

    # 3. Last-resort fallback to config
    logger.warning(
        f"[{symbol}] Falling back to config PRODUCT_ID={PRODUCT_ID}. "
        f"This may be incorrect for symbol '{symbol}'."
    )
    return PRODUCT_ID


def fetch_candles(
    symbol: str = SYMBOL,
    timeframe: str = "15m",
    limit: int = 100,
) -> pd.DataFrame:
    """
    Fetch the most recent `limit` candles for a symbol from Delta Exchange.

    Candle URL is derived from BASE_URL (respects USE_TESTNET in config).
    Fetches 2x the requested limit to account for sparse candles (e.g. XAUTUSD),
    then trims to the most recent `limit` rows.
    Timestamps are returned as UTC-aware datetimes to avoid IST/UTC display confusion.

    Args:
        symbol    : Trading symbol (default: SYMBOL from config)
        timeframe : Candle resolution (default: 15m)
        limit     : Number of candles to return (default: 100)

    Returns:
        pd.DataFrame with columns: timestamp, open, high, low, close, volume
        Returns empty DataFrame on failure.
    """
    resolution = RESOLUTION_MAP.get(timeframe, "15m")
    interval_seconds = _get_interval_seconds(resolution)

    # Fix 1: Use BASE_URL so testnet/production is respected
    candle_url = f"{BASE_URL}/v2/history/candles"

    # Fix 2: Fetch 2x limit to handle sparse candles (XAUTUSD, low-liquidity symbols)
    fetch_limit = limit * 2
    end_time = int(time.time())
    start_time = end_time - (fetch_limit * interval_seconds)

    params = {
        "resolution": resolution,
        "symbol": symbol,
        "start": start_time,
        "end": end_time,
    }

    try:
        response = requests.get(candle_url, params=params, timeout=(3, 27)).json()

        candles = response.get("result", [])

        if not candles:
            logger.warning(f"[{symbol}] No candles returned for timeframe={timeframe}")
            return pd.DataFrame()

        df = pd.DataFrame(candles)
        df = df.rename(columns={"time": "timestamp"})

        # Fix 3: Attach UTC timezone so timestamps display correctly (no IST/UTC confusion)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.drop_duplicates(subset="timestamp")
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Trim to the most recent `limit` candles after sorting
        df = df.tail(limit).reset_index(drop=True)

        logger.debug(
            f"[{symbol}] Fetched {len(df)} candles: "
            f"{df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}"
        )
        return df

    except Exception as e:
        logger.error(f"[{symbol}] Error fetching candles: {e}")
        return pd.DataFrame()


def _get_interval_seconds(resolution: str) -> int:
    """Return interval in seconds for a given resolution string."""
    mapping = {
        "5s": 5,
        "1m": 60,
        "3m": 180,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "2h": 7200,
        "4h": 14400,
        "6h": 21600,
        "12h": 43200,
        "1d": 86400,
        "1w": 604800,
    }
    return mapping.get(resolution, 900)
