# =============================================================================
# position_manager.py - Query and manage open positions
# =============================================================================

from logger import get_logger

from auth import signed_get
from config import PRODUCT_ID

logger = get_logger(__name__)


def get_open_position():
    """
    Fetch the current open position for the configured product.

    Returns:
        Position dict if open, None otherwise.
    """
    try:
        response = signed_get("/v2/positions", params={"product_id": PRODUCT_ID})
        result = response.get("result", {})

        if result and float(result.get("size", 0)) != 0:
            return result

        return None

    except Exception as e:
        logger.error(f"Error fetching position: {e}")
        return None


def has_open_position():
    """Returns True if there is an open position."""
    return get_open_position() is not None


def get_position_side(position: dict):
    """
    Extract the side of an open position.

    Returns:
        "buy" for long, "sell" for short
    """
    return position.get("side", "").lower()


def get_position_size(position: dict):
    """
    Extract the size (in lots) of an open position.

    Returns:
        int: number of lots
    """
    return int(float(position.get("size", 0)))


def get_entry_price(position: dict):
    """
    Extract the average entry price of an open position.

    Returns:
        float: entry price
    """
    return float(position.get("avg_entry_price", 0))
