# =============================================================================
# order_manager.py - Order placement and management
# =============================================================================
import json

import requests

import config
from auth import get_headers, signed_delete, signed_get, signed_post

# from config import BASE_URL, ORDER_SIZE, PRODUCT_ID, STOP_LOSS_PCT, TAKE_PROFIT_PCT
from logger import get_logger
from market_data import get_product_id  # single shared implementation

logger = get_logger(__name__)


def place_market_order(side: str, size: int = config.ORDER_SIZE, symbol: str = None):
    """
    Place a market order.

    Args:
        side  : "buy" or "sell"
        size  : Number of lots (default from config)
        symbol: Trading symbol e.g. "ETHUSD" (optional, falls back to config PRODUCT_ID)

    Returns:
        API response dict or None on failure
    """
    # Always resolve product_id dynamically — get_product_id() uses config.getbaseUrl()
    # at runtime so it automatically hits the correct environment (testnet or production)
    product_id = get_product_id(symbol) if symbol else config.PRODUCT_ID

    logger.info(
        f"[{symbol or 'default'}] Placing order with product_id={int(product_id)} on "
        f"base_url={config.getbaseUrl()} | USE_TESTNET={config.USE_TESTNET}"
    )

    payload = {
        "product_id": int(product_id),  # force int — API rejects strings/floats
        "size": int(size),  # force int — API rejects floats like 1.0
        "side": side,
        "order_type": "market_order",
    }

    logger.debug(
        f"[{symbol or 'default'}] Order payload: product_id={int(product_id)} ({type(product_id).__name__}), size={int(size)} ({type(size).__name__}), side={side}"
    )
    try:
        response = signed_post("/v2/orders", payload)
        logger.info(
            f"[{symbol or 'default'}] Market order placed: {side.upper()} {size} lots | Response: {response}"
        )
        return response
    except requests.exceptions.HTTPError as e:
        logger.error(
            f"[{symbol or 'default'}] Failed to place market order ({side}): {e} | Response body: {e.response.text}"
        )
        return None
    except Exception as e:
        logger.error(
            f"[{symbol or 'default'}] Failed to place market order ({side}): {e}"
        )
        return None


def place_bracket_order(
    product_id,
    order_id,
    side,
    fill_price,
    stop_loss_pct,
    take_profit_pct,
    tick_size=0.5,
):
    """
    Place bracket order using actual fill_price from market order response.

    Args:
        product_id     : Product ID for the symbol
        order_id       : Order ID of the filled entry order
        side           : Side of the ENTRY order ('buy' or 'sell')
        fill_price     : Actual fill price of the entry order
        stop_loss_pct  : Stop loss percentage (e.g. 0.6 means 0.6%)
        take_profit_pct: Take profit percentage (e.g. 1.2 means 1.2%)
        tick_size      : Tick size for the symbol (default: 0.5 for BTCUSD)
    """

    def round_to_tick(price, tick):
        return round(round(price / tick) * tick, 10)

    if side == "buy":
        sl_price = round_to_tick(fill_price * (1 - stop_loss_pct / 100), tick_size)
        tp_price = round_to_tick(fill_price * (1 + take_profit_pct / 100), tick_size)
        bracket_side = "sell"
    else:
        sl_price = round_to_tick(fill_price * (1 + stop_loss_pct / 100), tick_size)
        tp_price = round_to_tick(fill_price * (1 - take_profit_pct / 100), tick_size)
        bracket_side = "buy"

    payload = {
        "product_id": product_id,
        "order_id": order_id,
        "stop_loss_order": {
            "order_type": "limit_order",
            "stop_price": str(sl_price),
            "limit_price": str(sl_price),
            "side": bracket_side,
        },
        "take_profit_order": {
            "order_type": "limit_order",
            "stop_price": str(tp_price),
            "limit_price": str(tp_price),
            "side": bracket_side,
        },
    }

    payload_json = json.dumps(payload)
    headers = get_headers("POST", "/v2/orders/bracket", "", payload_json)
    url = f"{config.getbaseUrl()}/v2/orders/bracket"

    try:
        response = requests.post(
            url, headers=headers, data=payload_json, timeout=(3, 27)
        )
        body = response.text
        logger.info(
            f"place_bracket_order | status={response.status_code} | body={body}"
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"place_bracket_order failed: {e}")
        return None


def cancel_all_orders(symbol: str = None):
    """
    Cancel all open orders (limit + stop) for the given symbol.
    Falls back to config PRODUCT_ID if symbol not provided.
    """
    product_id = get_product_id(symbol) if symbol else config.PRODUCT_ID

    params = {
        "product_id": product_id,
        "cancel_stop_orders": "true",
        "cancel_limit_orders": "true",
    }

    try:
        response = signed_delete("/v2/orders/all", params=params)
        logger.info(
            f"[{symbol or 'default'}] All orders cancelled | Response: {response}"
        )
        return response
    except Exception as e:
        err_str = str(e)
        if "invalid_contract" in err_str or "400" in err_str:
            logger.debug(
                f"[{symbol or 'default'}] cancel_all_orders: no open orders to cancel — continuing"
            )
            return {"success": True, "result": []}
        logger.error(f"[{symbol or 'default'}] Failed to cancel all orders: {e}")
        return None


def close_position(side: str, size: int, symbol: str = None):
    """
    Close an open position with a market order in the opposite direction.

    Args:
        side  : Current position side ("buy" for long, "sell" for short)
        size  : Number of lots to close
        symbol: Trading symbol e.g. "ETHUSD" (optional, falls back to config PRODUCT_ID)
    """
    close_side = "sell" if side == "buy" else "buy"
    logger.info(
        f"[{symbol or 'default'}] Closing position: {close_side.upper()} {size} lots"
    )
    return place_market_order(close_side, size, symbol)
