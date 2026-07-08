# =============================================================================
# position_manager.py - Query and manage open positions
# =============================================================================

import config
from auth import signed_get
from logger import get_logger
from market_data import get_product_id

logger = get_logger(__name__)


def get_open_position(symbol: str = None) -> dict | None:
    """
    Fetch the current open position for the given symbol.

    Resolves product_id dynamically via get_product_id() so the correct ID
    is used for whichever environment (testnet/production) is currently active.
    Falls back to config.PRODUCT_ID if no symbol is provided.

    Args:
        symbol: Trading symbol e.g. 'BTCUSD', 'ETHUSD'. If None, falls back
                to config.PRODUCT_ID (last-resort, may be wrong on testnet).

    Returns:
        Position dict if an open position exists, None otherwise.
    """
    # FIX: Resolve product_id dynamically from symbol so testnet and production
    # both use the correct ID. Previously this always used config.PRODUCT_ID
    # which is a static production value and caused invalid_contract on testnet.
    if symbol:
        product_id = get_product_id(symbol)
    else:
        product_id = config.PRODUCT_ID
        logger.warning(
            "get_open_position called without symbol — using config.PRODUCT_ID "
            f"({product_id}) as fallback. This may be incorrect on testnet."
        )

    try:
        response = signed_get("/v2/positions", params={"product_id": product_id})
        result = response.get("result", {})

        if result and float(result.get("size", 0)) != 0:
            return result

        return None

    except Exception as e:
        logger.error(f"Error fetching position: {e}")
        return None


def has_open_position(symbol: str = None) -> bool:
    """Returns True if there is an open position for the given symbol."""
    return get_open_position(symbol) is not None


def get_position_side(position: dict) -> str:
    """
    Extract the side of an open position.

    Returns:
        "buy" for long, "sell" for short
    """
    return position.get("side", "").lower()


def get_position_size(position: dict) -> int:
    """
    Extract the size (in lots) of an open position.

    Returns:
        int: number of lots
    """
    return int(float(position.get("size", 0)))


def get_entry_price(position: dict) -> float:
    """
    Extract the average entry price of an open position.

    Returns:
        float: entry price
    """
    return float(position.get("avg_entry_price", 0))
