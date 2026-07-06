# =============================================================================
# backtester.py - Historical Backtest Simulation
# =============================================================================

import os
import time
from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from auth import public_get
from config import (
    BACKTEST_CHART_FILE,
    BACKTEST_DAYS,
    EMA_FAST,
    EMA_SLOW,
    ORDER_SIZE,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    TAKER_FEE,
)
from logger import get_logger
from strategy import compute_ema

logger = get_logger(__name__)

RESOLUTION_MAP = {
    "5s": "5s",
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "12h": "12h",
    "1d": "1d",
    "1w": "1w",
}

BATCH_SIZE = 1900  # Stay safely under 2000 candle API limit


def fetch_historical_candles(
    days: int = BACKTEST_DAYS, timeframe: str = "15m"
) -> pd.DataFrame:
    """
    Fetch historical candles in batches of BATCH_SIZE.
    Always uses production URL for candle data (testnet has no historical data).
    """
    import requests

    PRODUCTION_CANDLE_URL = "https://api.india.delta.exchange/v2/history/candles"

    resolution = RESOLUTION_MAP.get(timeframe, "15")
    # interval_seconds = int(resolution) * 60 if resolution != "1D" else 86400
    INTERVAL_SECONDS_MAP = {
        "5s": 5,
        "1m": 60,
        "3m": 180,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "2h": 7200,
        "4h": 14400,
        "6h": 21600,
        "12h": 43200,
        "1d": 86400,
        "1w": 604800,
    }
    interval_seconds = INTERVAL_SECONDS_MAP.get(resolution, 900)

    end_time = int(time.time())
    start_time = end_time - (days * 86400)

    all_candles = []
    batch_end = end_time

    logger.info(
        f"Fetching {days} days of {timeframe} candles in batches of {BATCH_SIZE}..."
    )

    while batch_end > start_time:
        batch_start = batch_end - (BATCH_SIZE * interval_seconds)
        if batch_start < start_time:
            batch_start = start_time

        params = {
            "resolution": resolution,
            "symbol": "BTCUSD",
            "start": batch_start,
            "end": batch_end,
        }

        try:
            response = requests.get(
                PRODUCTION_CANDLE_URL, params=params, timeout=(3, 27)
            ).json()

            candles = response.get("result", [])

            if not candles:
                break

            all_candles.extend(candles)
            logger.debug(f"Fetched {len(candles)} candles (batch end={batch_end})")

            batch_end = batch_start - interval_seconds
            time.sleep(0.3)  # Respect rate limits

        except Exception as e:
            logger.error(f"Error fetching batch: {e}")
            break

    if not all_candles:
        logger.error("No historical candles fetched.")
        return pd.DataFrame()

    df = pd.DataFrame(all_candles)
    df = df.rename(columns={"time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates(subset="timestamp")
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(f"Total candles fetched: {len(df)}")
    return df


def run_backtest(df: pd.DataFrame = None, days: int = BACKTEST_DAYS):
    """
    Run EMA crossover backtest simulation.

    Logic:
    - Compute EMA fast and slow on full dataset
    - Iterate candle by candle (from EMA_SLOW+5 onward)
    - Detect crossover using confirmed closed candles (i-1 and i-2)
    - On signal: flip position (close existing, open new)
    - Each candle: check if SL or TP was hit using high/low
    - Record each closed trade with PnL

    Returns:
        trades (list of dicts), summary (dict)
    """
    if df is None or df.empty:
        df = fetch_historical_candles(days=days)

    if df.empty:
        logger.error("No data available for backtest.")
        return [], {}

    df = df.copy()
    df["ema_fast"] = compute_ema(df["close"], EMA_FAST)
    df["ema_slow"] = compute_ema(df["close"], EMA_SLOW)

    trades = []
    position = None  # {"side": "buy"/"sell", "entry_price": float, "entry_idx": int}

    start_idx = EMA_SLOW + 5

    for i in range(start_idx, len(df)):
        candle = df.iloc[i]
        high = candle["high"]
        low = candle["low"]

        # --- Check SL/TP if in position ---
        if position is not None:
            entry = position["entry_price"]
            side = position["side"]

            if side == "buy":
                sl_price = round(entry * (1 - STOP_LOSS_PCT), 1)
                tp_price = round(entry * (1 + TAKE_PROFIT_PCT), 1)
                hit_sl = low <= sl_price
                hit_tp = high >= tp_price
            else:
                sl_price = round(entry * (1 + STOP_LOSS_PCT), 1)
                tp_price = round(entry * (1 - TAKE_PROFIT_PCT), 1)
                hit_sl = high >= sl_price
                hit_tp = low <= tp_price

            exit_price = None
            exit_reason = None

            if hit_tp:
                exit_price = tp_price
                exit_reason = "TP"
            elif hit_sl:
                exit_price = sl_price
                exit_reason = "SL"

            if exit_price is not None:
                size_btc = ORDER_SIZE * 0.001
                if side == "buy":
                    gross_pnl = size_btc * (exit_price - entry)
                else:
                    gross_pnl = size_btc * (entry - exit_price)

                commission = size_btc * (entry + exit_price) * TAKER_FEE
                net_pnl = gross_pnl - commission

                trades.append(
                    {
                        "entry_time": df.iloc[position["entry_idx"]]["timestamp"],
                        "exit_time": candle["timestamp"],
                        "side": side,
                        "entry_price": entry,
                        "exit_price": exit_price,
                        "gross_pnl": round(gross_pnl, 6),
                        "commission": round(commission, 6),
                        "net_pnl": round(net_pnl, 6),
                        "exit_reason": exit_reason,
                    }
                )
                position = None
                continue

        # --- Check for EMA crossover signal ---
        prev_fast = df["ema_fast"].iloc[i - 2]
        prev_slow = df["ema_slow"].iloc[i - 2]
        curr_fast = df["ema_fast"].iloc[i - 1]
        curr_slow = df["ema_slow"].iloc[i - 1]

        signal = None
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            signal = "BUY"
        elif prev_fast >= prev_slow and curr_fast < curr_slow:
            signal = "SELL"

        if signal is None:
            continue

        # --- Close existing position on flip ---
        if position is not None:
            entry = position["entry_price"]
            side = position["side"]
            exit_price = candle["open"]
            size_btc = ORDER_SIZE * 0.001

            if side == "buy":
                gross_pnl = size_btc * (exit_price - entry)
            else:
                gross_pnl = size_btc * (entry - exit_price)

            commission = size_btc * (entry + exit_price) * TAKER_FEE
            net_pnl = gross_pnl - commission

            trades.append(
                {
                    "entry_time": df.iloc[position["entry_idx"]]["timestamp"],
                    "exit_time": candle["timestamp"],
                    "side": side,
                    "entry_price": entry,
                    "exit_price": exit_price,
                    "gross_pnl": round(gross_pnl, 6),
                    "commission": round(commission, 6),
                    "net_pnl": round(net_pnl, 6),
                    "exit_reason": "signal_flip",
                }
            )

        # --- Open new position ---
        position = {
            "side": "buy" if signal == "BUY" else "sell",
            "entry_price": candle["open"],
            "entry_idx": i,
        }

    summary = _print_backtest_summary(trades, df)
    _generate_backtest_chart(df, trades)
    return trades, summary


def _print_backtest_summary(trades: list, df: pd.DataFrame) -> dict:
    """Print and return backtest performance summary."""
    if not trades:
        logger.warning("No trades in backtest.")
        print("\n[BACKTEST] No trades generated.")
        return {}

    total_trades = len(trades)
    wins = [t for t in trades if t["net_pnl"] > 0]
    losses = [t for t in trades if t["net_pnl"] <= 0]
    total_pnl = sum(t["net_pnl"] for t in trades)
    win_rate = len(wins) / total_trades * 100
    avg_win = np.mean([t["net_pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["net_pnl"] for t in losses]) if losses else 0
    max_drawdown = _calculate_max_drawdown(trades)

    summary = {
        "total_trades": total_trades,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 1),
        "total_pnl_btc": round(total_pnl, 6),
        "avg_win_btc": round(avg_win, 6),
        "avg_loss_btc": round(avg_loss, 6),
        "max_drawdown_btc": round(max_drawdown, 6),
        "period_start": str(df["timestamp"].iloc[0]),
        "period_end": str(df["timestamp"].iloc[-1]),
    }

    print("\n" + "=" * 55)
    print("         EMA CROSSOVER BACKTEST RESULTS")
    print("=" * 55)
    print(f"  Period       : {summary['period_start']} to {summary['period_end']}")
    print(f"  Total Trades : {total_trades}")
    print(f"  Wins / Losses: {len(wins)} / {len(losses)}")
    print(f"  Win Rate     : {win_rate:.1f}%")
    print(f"  Total PnL    : {total_pnl:.6f} BTC")
    print(f"  Avg Win      : {avg_win:.6f} BTC")
    print(f"  Avg Loss     : {avg_loss:.6f} BTC")
    print(f"  Max Drawdown : {max_drawdown:.6f} BTC")
    print("=" * 55 + "\n")

    return summary


def _calculate_max_drawdown(trades: list) -> float:
    """Calculate maximum drawdown from trade list."""
    cumulative = np.cumsum([t["net_pnl"] for t in trades])
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0


def _generate_backtest_chart(df: pd.DataFrame, trades: list):
    """
    Generate 3-panel backtest chart:
    Panel 1: Price + EMA lines + trade markers
    Panel 2: Cumulative PnL
    Panel 3: Per-trade PnL bars
    """
    if not trades:
        return

    try:
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle("EMA Crossover Backtest Report", fontsize=14, fontweight="bold")

        # Panel 1: Price + EMAs
        axes[0].plot(
            df["timestamp"],
            df["close"],
            color="gray",
            linewidth=0.8,
            label="Close Price",
            alpha=0.7,
        )
        axes[0].plot(
            df["timestamp"],
            df["ema_fast"],
            color="blue",
            linewidth=1.2,
            label=f"EMA {EMA_FAST}",
        )
        axes[0].plot(
            df["timestamp"],
            df["ema_slow"],
            color="orange",
            linewidth=1.2,
            label=f"EMA {EMA_SLOW}",
        )

        for t in trades:
            color = "green" if t["side"] == "buy" else "red"
            marker = "^" if t["side"] == "buy" else "v"
            axes[0].scatter(
                t["entry_time"],
                t["entry_price"],
                color=color,
                marker=marker,
                s=60,
                zorder=5,
            )
            axes[0].scatter(
                t["exit_time"],
                t["exit_price"],
                color="black",
                marker="x",
                s=40,
                zorder=5,
            )

        axes[0].set_title("Price + EMA Crossover Signals")
        axes[0].set_ylabel("Price (USD)")
        axes[0].legend(loc="upper left", fontsize=8)
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

        # Panel 2: Cumulative PnL
        cum_pnl = np.cumsum([t["net_pnl"] for t in trades])
        trade_times = [t["exit_time"] for t in trades]
        axes[1].plot(
            trade_times, cum_pnl, color="blue", linewidth=1.5, label="Cumulative PnL"
        )
        axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[1].fill_between(
            trade_times, cum_pnl, 0, where=(cum_pnl >= 0), alpha=0.2, color="green"
        )
        axes[1].fill_between(
            trade_times, cum_pnl, 0, where=(cum_pnl < 0), alpha=0.2, color="red"
        )
        axes[1].set_title("Cumulative PnL (BTC)")
        axes[1].set_ylabel("BTC")
        axes[1].legend()
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

        # Panel 3: Per-trade PnL bars
        colors = ["green" if t["net_pnl"] > 0 else "red" for t in trades]
        axes[2].bar(
            range(len(trades)), [t["net_pnl"] for t in trades], color=colors, alpha=0.7
        )
        axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[2].set_title("Per-Trade PnL (BTC)")
        axes[2].set_xlabel("Trade #")
        axes[2].set_ylabel("BTC")

        plt.tight_layout()
        os.makedirs(os.path.dirname(BACKTEST_CHART_FILE), exist_ok=True)
        plt.savefig(BACKTEST_CHART_FILE, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Backtest chart saved to {BACKTEST_CHART_FILE}")
        print(f"[BACKTEST] Chart saved to: {BACKTEST_CHART_FILE}")

    except Exception as e:
        logger.error(f"Error generating backtest chart: {e}")
