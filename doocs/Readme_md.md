# Run all 4 strategies and compare
py run_backtest.py --all

# Bot
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


# Single strategy with defaults
py run_backtest.py --strategy ema_crossover

py run_backtest.py --strategy vwap
py run_backtest.py --strategy triple_ema

# Custom parameters
py run_backtest.py --strategy ema_crossover --fast 5 --slow 15 --days 60
py run_backtest.py --strategy bollinger_bands --period 20 --std 2.5 --days 45
py run_backtest.py --strategy vwap --period 100 --days 30
py run_backtest.py --strategy triple_ema --fast 5 --mid 13 --slow 34 --days 60

# All strategies with custom days
py run_backtest.py --all --days 60

D:\Delta_copilot\
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py
│   ├── ema_crossover.py
│   ├── bollinger_bands.py
│   ├── vwap.py
│   └── triple_ema.py
├── backtester.py
├── run_backtest.py
├── config.py
├── auth.py
├── market_data.py
├── strategy.py
├── order_manager.py
├── position_manager.py
├── notifier.py
├── pnl_tracker.py
├── logger.py
├── bot.py
├── requirements.txt
└── logs/
