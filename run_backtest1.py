# =============================================================================
# run_backtest.py - Standalone Backtest Runner
# =============================================================================

import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtester import fetch_historical_candles, run_backtest

from config import BACKTEST_CHART_FILE, BACKTEST_DAYS
from logger import get_logger

logger = get_logger("run_backtest")


def main():
    print("\n" + "=" * 55)
    print("     EMA CROSSOVER STRATEGY - BACKTEST RUNNER")
    print("=" * 55)
    print(f"  Fetching {BACKTEST_DAYS} days of historical data...")
    print(f"  Symbol  : BTCUSD")
    print(f"  EMA     : Fast=9, Slow=21")
    print(f"  SL/TP   : 1% / 2%")
    print(f"  Lot Size: 10 lots (0.01 BTC)")
    print("=" * 55 + "\n")

    df = fetch_historical_candles(days=BACKTEST_DAYS)

    if df.empty:
        print(
            "[ERROR] Failed to fetch historical data. Check your internet connection."
        )
        sys.exit(1)

    print(
        f"[INFO] Fetched {len(df)} candles from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}\n"
    )

    trades, summary = run_backtest(df=df)

    if not trades:
        print("[WARNING] No trades were generated. Strategy may need adjustment.")
        sys.exit(0)

    print(f"[INFO] Backtest chart saved to: {BACKTEST_CHART_FILE}")
    print("\n[DONE] Backtest complete. Review the chart before going live.\n")


if __name__ == "__main__":
    main()
