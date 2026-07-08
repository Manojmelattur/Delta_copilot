# =============================================================================
# config.py - All configuration settings
# =============================================================================

# --- Telegram ---
from numpy._core.numerictypes import ushort

TELEGRAM_BOT_TOKEN = "your_telegram_bot_token_here"
TELEGRAM_CHAT_ID = "your_telegram_chat_id_here"

# --- API Credentials ---
USE_TESTNET = False  # Set to False for live trading


API_KEY_TESTNET = "41pSKOuhVSnejj4vMbYaF9w0f9YgyZ"
API_SECRET_TESTNET = "GZ1Xxyc35ausXaGMONJHcwqf6DNeDT4ccIjIy7si33cX55wtlD1lGM2AkQrt"

API_KEY = "Iq55cC0tAFh3XfF5CnEOCWFWcQnFBH"
API_SECRET = "kNoyZWnfdYOLc0VE67hpQkVDcTnczjyNb7aPH8xNkbxFwSAUj8n7a5XVer8R"

# --- Base URLs ---
TESTNET_BASE_URL = "https://cdn-ind.testnet.deltaex.org"
PRODUCTION_BASE_URL = "https://api.india.delta.exchange"

BASE_URL = TESTNET_BASE_URL if USE_TESTNET else PRODUCTION_BASE_URL

# --- Product Settings ---
# PRODUCT_ID is used only as a last-resort fallback when symbol lookup fails.
# get_product_id() in market_data.py resolves the correct ID automatically.
# Confirmed product IDs (from Delta Exchange API):
#   BTCUSD  : 27
#   ETHUSD  : 3136
#   XAUTUSD : 131253
SYMBOL = "BTCUSD"
if USE_TESTNET:
    PRODUCT_ID = 84  # FIX: was 84 (incorrect). Correct BTCUSD product_id is 27.
else:
    PRODUCT_ID = 27  # FIX: was 84 (incorrect). Correct BTCUSD product_id is 27.

# PRODUCT_ID = 27  # FIX: was 84 (incorrect). Correct BTCUSD product_id is 27.

# --- Strategy Settings ---
EMA_FAST = 9
EMA_SLOW = 21
TIMEFRAME = "15m"
POLL_INTERVAL = 60  # seconds between each bot loop

# --- Order Settings ---
ORDER_SIZE = 1  # lots (1 lot = 0.001 BTC for BTCUSD)

# --- Risk Management ---
# These are decimal fractions: 0.01 = 1%, 0.02 = 2%
# Used in backtester.py generic SL/TP fallback only.
# Strategy-specific SL/TP are passed as CLI args (e.g. --bb-stop-loss 0.6 means 0.6%)
STOP_LOSS_PCT = 0.01  # 1%
TAKE_PROFIT_PCT = 0.02  # 2%

# --- Taker Fee ---
TAKER_FEE = 0.0005  # 0.05%

# --- Backtest Settings ---
BACKTEST_DAYS = 30

# --- Logging ---
LOG_FILE = "logs/bot.log"
PNL_CSV_FILE = "logs/pnl_log.csv"
BACKTEST_CHART_FILE = "logs/backtest_report.png"
PNL_CHART_FILE = "logs/pnl_chart.png"


def getbaseUrl():
    return TESTNET_BASE_URL if USE_TESTNET else PRODUCTION_BASE_URL


def getAPI_KEY():
    return API_KEY_TESTNET if USE_TESTNET else API_KEY


def getAPI_SECRET():
    return API_SECRET_TESTNET if USE_TESTNET else API_SECRET
