================================================================
       EMA CROSSOVER TRADING BOT - DELTA EXCHANGE
================================================================

A fully automated Python trading bot implementing an EMA (9/21)
crossover strategy on BTCUSD perpetual futures on Delta Exchange.

----------------------------------------------------------------
PROJECT STRUCTURE
----------------------------------------------------------------

D:\Delta_copilot\
├── config.py           - All settings (API keys, strategy params, SL/TP)
├── auth.py             - HMAC-SHA256 signed API requests
├── market_data.py      - Fetch OHLCV candles from Delta Exchange
├── strategy.py         - EMA crossover signal logic
├── order_manager.py    - Place/cancel market and bracket orders
├── position_manager.py - Query open positions
├── notifier.py         - Telegram notifications
├── pnl_tracker.py      - PnL CSV logging and chart generation
├── backtester.py       - Historical backtest simulation
├── logger.py           - Centralized logging configuration
├── bot.py              - Main bot loop
├── run_backtest.py     - Standalone backtest runner
├── requirements.txt    - Python dependencies
├── logs/               - Auto-created at runtime
│   ├── bot.log         - Full debug log
│   ├── pnl_log.csv     - Trade history with PnL
│   ├── backtest_report.png  - Backtest chart (3 panels)
│   └── pnl_chart.png   - Live trading PnL chart
└── README.md

----------------------------------------------------------------
REQUIREMENTS
----------------------------------------------------------------

- Python 3.11 or 3.12 (recommended)
  Download: https://www.python.org/downloads/

- Do NOT use Python 3.13 or 3.14
  These versions are not yet fully supported by matplotlib,
  pandas, and other dependencies.

----------------------------------------------------------------
INSTALLATION
----------------------------------------------------------------

Step 1: Clone or copy all files to your project folder
        Example: D:\Delta_copilot\

Step 2: Open terminal and navigate to project folder
        cd D:\Delta_copilot

Step 3: Create a virtual environment (recommended)
        py -3.11 -m venv venv          (Windows)
        python3.11 -m venv venv        (Mac/Linux)

Step 4: Activate virtual environment
        venv\Scripts\activate          (Windows)
        source venv/bin/activate       (Mac/Linux)

Step 5: Install all dependencies
        pip install -r requirements.txt

Step 6: Verify installation
        python -c "import matplotlib.dates; print('matplotlib OK')"
        python -c "import pandas; print('pandas OK')"
        python -c "import requests; print('requests OK')"

----------------------------------------------------------------
CONFIGURATION
----------------------------------------------------------------

Open config.py and fill in the following values:

1. API CREDENTIALS
   API_KEY    = "your_api_key_here"
   API_SECRET = "your_api_secret_here"

   Get your API keys from:
   https://www.delta.exchange/app/account/manageapikeys

2. BASE URL
   Testnet (Demo - use this first):
   BASE_URL = "https://cdn-ind.testnet.deltaex.org"

   Production (Live - switch after validation):
   BASE_URL = "https://api.india.delta.exchange"

3. STRATEGY SETTINGS (defaults are pre-configured)
   SYMBOL          = "BTCUSD"
   PRODUCT_ID      = 27
   EMA_FAST        = 9
   EMA_SLOW        = 21
   TIMEFRAME       = "15m"
   ORDER_SIZE      = 10        (lots, 1 lot = 0.001 BTC)
   STOP_LOSS_PCT   = 0.01      (1%)
   TAKE_PROFIT_PCT = 0.02      (2%)
   POLL_INTERVAL   = 60        (seconds)

4. TELEGRAM (optional but recommended)
   TELEGRAM_BOT_TOKEN = "your_telegram_bot_token_here"
   TELEGRAM_CHAT_ID   = "your_telegram_chat_id_here"

5. BACKTEST SETTINGS
   BACKTEST_DAYS = 30          (days of historical data)

----------------------------------------------------------------
IP WHITELISTING (MANDATORY)
----------------------------------------------------------------

Delta Exchange requires IP whitelisting for all API keys.
This cannot be disabled.

Steps:
1. Go to: https://www.delta.exchange/app/account/manageapikeys
2. Find your API key and click Edit
3. Add your server's public IP address
4. Multiple IPs can be added as comma-separated values
   Example: 103.21.45.67, 103.21.45.68
5. IP ranges (CIDR notation) are NOT supported
6. Save changes

To find your public IP:
   Visit: https://whatismyipaddress.com/

----------------------------------------------------------------
TELEGRAM SETUP (OPTIONAL)
----------------------------------------------------------------

Telegram notifications alert you on:
- Trade entry (side, size, entry price, SL, TP)
- Trade exit (PnL per trade)
- Signal detection (EMA values)
- Bot errors
- Daily PnL report (sent at midnight)

Step 1: Create a Telegram Bot
   - Open Telegram
   - Search for @BotFather
   - Send: /newbot
   - Follow the instructions
   - Copy the bot token provided

Step 2: Get your Chat ID
   - Send any message to your new bot
   - Visit this URL in your browser:
     https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
   - Find "chat":{"id": <YOUR_CHAT_ID>}
   - Copy that number

Step 3: Update config.py
   TELEGRAM_BOT_TOKEN = "123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ"
   TELEGRAM_CHAT_ID   = "987654321"

If Telegram is not configured, the bot will skip notifications
and continue trading normally.

----------------------------------------------------------------
RUNNING THE BOT - RECOMMENDED WORKFLOW
----------------------------------------------------------------

STEP 1: RUN BACKTEST FIRST (always do this before going live)

   py run_backtest.py

   This will:
   - Fetch 30 days of real BTCUSD 15m candles from production
   - Simulate EMA 9/21 crossover strategy with 1% SL and 2% TP
   - Print performance summary in terminal
   - Save 3-panel chart to: logs\backtest_report.png

   Review the chart before proceeding.
   If win rate is below 40% or total PnL is negative,
   consider adjusting EMA periods or SL/TP in config.py.

STEP 2: RUN ON TESTNET (paper trading)

   Make sure config.py has:
   BASE_URL = "https://cdn-ind.testnet.deltaex.org"

   Run:
   py bot.py

   Monitor the terminal output and Telegram notifications.
   Let it run for at least 2-3 days to validate behavior.

STEP 3: SWITCH TO PRODUCTION (live trading)

   After testnet validation, update config.py:
   BASE_URL = "https://api.india.delta.exchange"

   Run:
   py bot.py

   To keep the bot running after closing terminal (Windows):
   start /B py bot.py > logs\output.log 2>&1

   To keep the bot running after closing terminal (Mac/Linux):
   nohup python bot.py > logs/output.log 2>&1 &

----------------------------------------------------------------
STRATEGY DETAILS
----------------------------------------------------------------

Parameter        Value
--------------   ------------------
Symbol           BTCUSD
Contract Type    Perpetual Futures
EMA Fast         9 periods
EMA Slow         21 periods
Timeframe        15 minutes
Order Type       Market Order (taker)
Order Size       10 lots
1 Lot            0.001 BTC
Total Size       0.01 BTC per trade
Stop Loss        1% from entry price
Take Profit      2% from entry price
Max Positions    1 at a time
Taker Fee        0.05% per side

Signal Logic:
- BUY  : EMA 9 crosses ABOVE EMA 21
- SELL : EMA 9 crosses BELOW EMA 21



How to run with different configs
Default BB params (0.6 SL, 1.2 TP):


python bot.py --strategy bollinger_bands

Custom BB params:


python bot.py --strategy bollinger_bands --bb-stop-loss 0.7 --bb-take-profit 1.2 --std 2.5

Tighter BB with faster poll:


python bot.py --strategy bollinger_bands --bb-stop-loss 0.6 --bb-take-profit 1.2 --poll-interval 900

EMA crossover:


python bot.py --strategy ema_crossover --fast 9 --slow 21

Triple EMA:


python bot.py --strategy triple_ema --fast 5 --mid 13 --slow 21
