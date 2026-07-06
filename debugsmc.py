# debug_smc.py
import sys

sys.path.insert(0, ".")

import pandas as pd

from backtester import fetch_historical_candles  # or however your project fetches OHLCV
from strategies.smc_strategy import SMCStrategy

# Fetch data - adjust to match your project's fetch function
df = fetch_historical_candles(symbol="ETHUSD", timeframe="15m", days=30)

strategy = SMCStrategy(
    swing_lookback=5,
    atr_length=14,
    atr_multiplier=1.5,
    trail_activation_atr=2.0,
    sl_atr_buffer=0.3,
    ob_strength_period=20,
    ob_strength_mult=0.3,  # CHANGED from 2.0
    max_ob_age_candles=30,
    rr_ratio=2.0,
    trend_ema_period=50,
    min_atr_threshold=0.0,
    bos_type="close",
    choch_type="close",
    structure_lookback=50,
)


df = strategy.calculate_indicators(df)

# Check last candle
report = strategy.debug_signal(df)
print("\n=== SMC Debug Report (last candle) ===")
for k, v in report.items():
    print(f"  {k:<30} {v}")

# Also scan last 50 candles to see if any would have triggered
print("\n=== Scanning last 50 candles for blocked_at distribution ===")
blocked_counts = {}
for i in range(max(strategy.get_min_candles(), len(df) - 50), len(df)):
    r = strategy.debug_signal(df.iloc[: i + 1])
    reason = r.get("blocked_at", "unknown")
    blocked_counts[reason] = blocked_counts.get(reason, 0) + 1

for reason, count in sorted(blocked_counts.items(), key=lambda x: -x[1]):
    print(f"  {reason:<40} {count} candles")

# Add at the bottom of debug_smc.py
print("\n=== CHoCH Deep Debug ===")
choch_report = strategy.debug_choch(df)
for k, v in choch_report.items():
    print(f"  {k:<40} {v}")

# Add at the bottom of debug_smc.py
print("\n=== OB Deep Debug ===")
ob_report = strategy.debug_ob(df)
for k, v in ob_report.items():
    print(f"  {k:<40} {v}")
