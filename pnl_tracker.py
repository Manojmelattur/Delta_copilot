# =============================================================================
# pnl_tracker.py - PnL Tracking, CSV Logging, and Chart Generation
# =============================================================================

import csv
import os
from datetime import date, datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from config import PNL_CHART_FILE, PNL_CSV_FILE, TAKER_FEE
from logger import get_logger
from notifier import notify_daily_pnl

logger = get_logger(__name__)

CSV_HEADERS = [
    "timestamp",
    "side",
    "size_lots",
    "size_btc",
    "entry_price",
    "exit_price",
    "gross_pnl_btc",
    "commission_btc",
    "net_pnl_btc",
    "exit_reason",
]


def _ensure_csv():
    """Create CSV file with headers if it doesn't exist."""
    os.makedirs(os.path.dirname(PNL_CSV_FILE), exist_ok=True)
    if not os.path.exists(PNL_CSV_FILE):
        with open(PNL_CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)


def record_trade(
    side: str,
    size: int,
    entry_price: float,
    exit_price: float,
    exit_reason: str = "signal",
):
    """
    Record a completed trade to the CSV log.

    BTCUSD is an inverse perpetual contract:
      - Contract value: 1 USD per lot
      - PnL is denominated in BTC (the margin asset)

    PnL Calculation (inverse perpetual):
        size_btc = size * contract_value_usd / entry_price
                 = size * 1 / entry_price

        gross_pnl_btc = size * (1/entry_price - 1/exit_price)  for LONG
        gross_pnl_btc = size * (1/exit_price - 1/entry_price)  for SHORT

        commission_btc = size * TAKER_FEE / entry_price   (entry leg)
                       + size * TAKER_FEE / exit_price    (exit leg)

        net_pnl_btc = gross_pnl_btc - commission_btc
    """
    _ensure_csv()

    # BTC value of position at entry (for reference only)
    size_btc = round(size / entry_price, 8)

    # Gross PnL in BTC — inverse perpetual formula
    if side.lower() == "buy":
        gross_pnl = size * (1 / entry_price - 1 / exit_price)
    else:
        gross_pnl = size * (1 / exit_price - 1 / entry_price)

    # Commission in BTC — both entry and exit legs
    commission = size * TAKER_FEE / entry_price + size * TAKER_FEE / exit_price

    net_pnl = gross_pnl - commission

    row = [
        datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        side.upper(),
        size,
        size_btc,
        entry_price,
        exit_price,
        round(gross_pnl, 8),
        round(commission, 8),
        round(net_pnl, 8),
        exit_reason,
    ]

    with open(PNL_CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    logger.info(
        f"Trade recorded | {side.upper()} {size} lots | "
        f"Entry={entry_price} Exit={exit_price} | "
        f"Net PnL={net_pnl:.8f} BTC | Reason={exit_reason}"
    )
    return net_pnl


def get_daily_summary(target_date: date = None):
    """
    Get PnL summary for a specific date (default: today).

    Returns:
        dict with total_pnl, num_trades, wins, win_rate
    """
    if not os.path.exists(PNL_CSV_FILE):
        return {"total_pnl": 0, "num_trades": 0, "wins": 0, "win_rate": 0}

    if target_date is None:
        target_date = date.today()

    date_str = target_date.strftime("%Y-%m-%d")

    try:
        df = pd.read_csv(PNL_CSV_FILE)
        df = df[df["timestamp"].str.startswith(date_str)]

        if df.empty:
            return {"total_pnl": 0, "num_trades": 0, "wins": 0, "win_rate": 0}

        total_pnl = df["net_pnl_btc"].sum()
        num_trades = len(df)
        wins = len(df[df["net_pnl_btc"] > 0])
        win_rate = (wins / num_trades * 100) if num_trades > 0 else 0

        return {
            "total_pnl": round(total_pnl, 8),
            "num_trades": num_trades,
            "wins": wins,
            "win_rate": round(win_rate, 1),
        }

    except Exception as e:
        logger.error(f"Error reading PnL CSV: {e}")
        return {"total_pnl": 0, "num_trades": 0, "wins": 0, "win_rate": 0}


def send_daily_report():
    """Send daily PnL summary via Telegram."""
    summary = get_daily_summary()
    today_str = date.today().strftime("%Y-%m-%d")
    notify_daily_pnl(
        date=today_str,
        total_pnl=summary["total_pnl"],
        num_trades=summary["num_trades"],
        win_rate=summary["win_rate"],
    )
    logger.info(f"Daily report sent for {today_str}: {summary}")


def generate_pnl_chart():
    """
    Generate a cumulative PnL chart from the CSV log.
    Saves to PNL_CHART_FILE (PNG).
    """
    if not os.path.exists(PNL_CSV_FILE):
        logger.warning("No PnL CSV found. Skipping chart generation.")
        return

    try:
        df = pd.read_csv(PNL_CSV_FILE)
        if df.empty:
            logger.warning("PnL CSV is empty. Skipping chart generation.")
            return

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        df["cumulative_pnl"] = df["net_pnl_btc"].cumsum()

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        fig.suptitle("Strategy Bot - PnL Report", fontsize=14, fontweight="bold")

        # Panel 1: Cumulative PnL
        axes[0].plot(
            df["timestamp"],
            df["cumulative_pnl"],
            color="blue",
            linewidth=1.5,
            label="Cumulative PnL",
        )
        axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[0].fill_between(
            df["timestamp"],
            df["cumulative_pnl"],
            0,
            where=(df["cumulative_pnl"] >= 0),
            alpha=0.2,
            color="green",
        )
        axes[0].fill_between(
            df["timestamp"],
            df["cumulative_pnl"],
            0,
            where=(df["cumulative_pnl"] < 0),
            alpha=0.2,
            color="red",
        )
        axes[0].set_title("Cumulative PnL (BTC)")
        axes[0].set_ylabel("BTC")
        axes[0].legend()
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

        # Panel 2: Per-trade PnL bars
        colors = ["green" if p > 0 else "red" for p in df["net_pnl_btc"]]
        axes[1].bar(range(len(df)), df["net_pnl_btc"], color=colors, alpha=0.7)
        axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[1].set_title("Per-Trade PnL (BTC)")
        axes[1].set_xlabel("Trade #")
        axes[1].set_ylabel("BTC")

        plt.tight_layout()
        os.makedirs(os.path.dirname(PNL_CHART_FILE), exist_ok=True)
        plt.savefig(PNL_CHART_FILE, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"PnL chart saved to {PNL_CHART_FILE}")

    except Exception as e:
        logger.error(f"Error generating PnL chart: {e}")
