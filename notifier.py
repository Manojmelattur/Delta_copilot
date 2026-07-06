# =============================================================================
# notifier.py - Telegram Notifications
# =============================================================================

import requests
from logger import get_logger

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = get_logger(__name__)

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"


def send_message(text: str):
    """Send a plain text message via Telegram."""
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "your_telegram_bot_token_here":
        logger.warning("Telegram not configured. Skipping notification.")
        return

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
    }

    try:
        response = requests.post(TELEGRAM_API, json=payload, timeout=10)
        response.raise_for_status()
        logger.debug(f"Telegram message sent: {text[:60]}...")
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")


def notify_trade_entry(side: str, size: int, entry_price: float, sl: float, tp: float):
    msg = (
        f"*TRADE ENTRY*\n"
        f"Side: `{side.upper()}`\n"
        f"Size: `{size} lots ({size * 0.001} BTC)`\n"
        f"Entry Price: `{entry_price}`\n"
        f"Stop Loss: `{sl}`\n"
        f"Take Profit: `{tp}`"
    )
    send_message(msg)


def notify_trade_exit(
    side: str, size: int, entry_price: float, exit_price: float, pnl: float
):
    direction = "LONG" if side == "buy" else "SHORT"
    pnl_str = f"+{pnl:.4f}" if pnl >= 0 else f"{pnl:.4f}"
    msg = (
        f"*TRADE EXIT*\n"
        f"Direction: `{direction}`\n"
        f"Size: `{size} lots`\n"
        f"Entry: `{entry_price}` | Exit: `{exit_price}`\n"
        f"PnL: `{pnl_str} BTC`"
    )
    send_message(msg)


def notify_signal(signal: str, fast_ema: float, slow_ema: float):
    msg = (
        f"*SIGNAL DETECTED*\n"
        f"Signal: `{signal}`\n"
        f"EMA Fast ({9}): `{fast_ema:.2f}`\n"
        f"EMA Slow ({21}): `{slow_ema:.2f}`"
    )
    send_message(msg)


def notify_error(error_msg: str):
    msg = f"*BOT ERROR*\n`{error_msg}`"
    send_message(msg)


def notify_daily_pnl(date: str, total_pnl: float, num_trades: int, win_rate: float):
    pnl_str = f"+{total_pnl:.4f}" if total_pnl >= 0 else f"{total_pnl:.4f}"
    msg = (
        f"*DAILY PnL REPORT - {date}*\n"
        f"Total PnL: `{pnl_str} BTC`\n"
        f"Trades: `{num_trades}`\n"
        f"Win Rate: `{win_rate:.1f}%`"
    )
    send_message(msg)
