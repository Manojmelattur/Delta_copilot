# =============================================================================
# run_sweep.py - Crossover period sweep: 9/21 vs 21/50 with trend=125
# =============================================================================

import argparse

from backtester import fetch_historical_candles, run_all_backtests
from strategies.ema_crossover import EMACrossoverStrategy


def main():
    parser = argparse.ArgumentParser(description="EMA Crossover period sweep")
    parser.add_argument(
        "--timeframe", type=str, default="15m", help="Candle timeframe (e.g. 15m, 1h)"
    )
    args = parser.parse_args()

    strategies = [
        EMACrossoverStrategy(fast=9, slow=21, trend_ema=0),  # original baseline
        EMACrossoverStrategy(
            fast=9, slow=21, trend_ema=125
        ),  # best from previous sweep
        EMACrossoverStrategy(
            fast=21, slow=50, trend_ema=0
        ),  # wider crossover, no filter
        EMACrossoverStrategy(
            fast=21, slow=50, trend_ema=125
        ),  # wider crossover + filter
    ]

    for days in [30, 60, 90]:
        print(f"\n{'#' * 55}")
        print(f"#  VALIDATION WINDOW: {days} DAYS")
        print(f"{'#' * 55}")

        df = fetch_historical_candles(days=days, timeframe=args.timeframe)

        if df.empty:
            print(f"[ERROR] No data fetched for {days} days. Skipping.")
            continue

        print(
            f"[INFO] {len(df)} candles: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}"
        )

        run_all_backtests(strategies, df=df, days=days)


if __name__ == "__main__":
    main()
