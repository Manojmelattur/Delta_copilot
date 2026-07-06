# =============================================================================
# sim_bot.py - Simulated Live Bot (Candle Replay Mode)
# =============================================================================
#
# Replays historical candles one by one through the live strategy loop.
# Places real orders on testnet via order_manager (or dry-run with --dry-run).
#
# Usage:
#   python sim_bot.py                  # 1s delay, 30 days, real orders
#   python sim_bot.py --dry-run        # log signals only, no orders placed
#   python sim_bot.py --fast           # 0s delay between candles
#   python sim_bot.py --days 15        # replay last 15 days
#   python sim_bot.py --size 2         # use 2 lots per trade
#   python sim_bot.py --fast --dry-run # fast + dry-run combined
#
# Files created:
#   sim_strategy_state.db  <- isolated from live bot state
#   logs/sim_bot.log       <- separate log file
#
# Zero changes to:
#   strategies/bollinger_bands.py
#   market_data.py
#   live_bot.py
#   backtester.py
#   order_manager.py
# =============================================================================

import argparse
import logging
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from config import PRODUCT_ID, STOP_LOSS_PCT, TAKE_PROFIT_PCT  # FIXED: added import
from order_manager import (
    cancel_all_orders,
    place_bracket_order,
    place_market_order,
)
from strategies.bollinger_bands import BollingerBandsStrategy

# -----------------------------------------------------------------------------
# Logging setup - writes to logs/sim_bot.log AND console
# -----------------------------------------------------------------------------

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("logs/sim_bot.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("sim_bot")

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

TESTNET_BASE_URL = "https://cdn-ind.testnet.deltaex.org"
SYMBOL = "BTCUSD"
RESOLUTION = "15m"
CANDLE_BATCH_SIZE = 1900

# BTCUSD contract specs: 1 lot = 0.001 BTC
CONTRACT_VALUE_BTC = 0.001

# -----------------------------------------------------------------------------
# Candle fetching
# -----------------------------------------------------------------------------


def fetch_historical_candles(days: int) -> list[dict]:
    """
    Fetch historical 15m candles from testnet in batches.
    Returns list of candle dicts sorted oldest to newest.
    """
    logger.info(
        f"Fetching {days} days of 15m candles in batches of {CANDLE_BATCH_SIZE}..."
    )

    end_ts = int(time.time())
    start_ts = end_ts - (days * 24 * 60 * 60)

    all_candles = []
    batch_end = end_ts
    MAX_RETRIES = 3
    RETRY_DELAY = 3  # seconds

    while batch_end > start_ts:
        retries = 0
        batch_done = False  # True = move to next batch; False = stop outer loop

        while retries < MAX_RETRIES:
            try:
                response = requests.get(
                    f"{TESTNET_BASE_URL}/v2/history/candles",
                    params={
                        "symbol": SYMBOL,
                        "resolution": RESOLUTION,
                        "start": str(start_ts),
                        "end": str(batch_end),
                    },
                    timeout=(5, 30),
                )
                response.raise_for_status()
                data = response.json()

                if not data.get("success") or not data.get("result"):
                    logger.warning("Empty or unsuccessful candle response.")
                    # Not a transient error - stop fetching entirely
                    batch_done = False
                    break

                candles = data["result"]
                if not candles:
                    batch_done = False
                    break

                logger.debug(f"Fetched {len(candles)} candles (batch_end={batch_end})")
                all_candles.extend(candles)

                # Move batch window back
                batch_end = min(c["time"] for c in candles) - 1

                if len(candles) < CANDLE_BATCH_SIZE:
                    # Last batch - no more data available
                    batch_done = False
                    break

                # Full batch fetched - continue to next batch
                batch_done = True
                break

            except requests.exceptions.RequestException as e:
                retries += 1
                if retries < MAX_RETRIES:
                    logger.warning(
                        f"Error fetching candle batch (attempt {retries}/{MAX_RETRIES}): {e}. "
                        f"Retrying in {RETRY_DELAY}s..."
                    )
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(
                        f"Error fetching candle batch after {MAX_RETRIES} attempts: {e}"
                    )
                    batch_done = False

        if not batch_done:
            break

    if not all_candles:
        logger.error("No candles fetched. Cannot run simulation.")
        return []

    # Deduplicate and sort oldest to newest
    seen = set()
    unique = []
    for c in all_candles:
        if c["time"] not in seen:
            seen.add(c["time"])
            unique.append(c)

    unique.sort(key=lambda x: x["time"])
    logger.info(f"Total candles fetched: {len(unique)}")

    if unique:
        start_dt = datetime.fromtimestamp(unique[0]["time"], tz=timezone.utc)
        end_dt = datetime.fromtimestamp(unique[-1]["time"], tz=timezone.utc)
        logger.info(f"Candle range: {start_dt} to {end_dt}")

    return unique


# -----------------------------------------------------------------------------
# Order helpers
# -----------------------------------------------------------------------------


def enter_position(side: str, size: int, entry_price: float, dry_run: bool) -> bool:
    """
    Place market entry order + bracket order (SL/TP).
    Returns True if entry order succeeded (or dry-run).
    """
    if dry_run:
        logger.info(
            f"[DRY-RUN] Would place ENTRY: {side.upper()} {size} lot(s) {SYMBOL} "
            f"@ ~{entry_price:.2f}"
        )
        return True

    # Cancel any existing orders first to avoid bracket conflicts
    cancel_all_orders()

    # Place market entry
    response = place_market_order(side, size)
    if not response or not response.get("success"):
        logger.error(f"Entry order failed: {response}")
        return False

    order_result = response.get("result", {})
    order_id = order_result.get("id")

    # Extract actual fill price from market order response
    fill_price_raw = order_result.get("average_fill_price")
    if not fill_price_raw:
        logger.warning(
            f"average_fill_price is None for order {order_id}; "
            "skipping bracket order - strategy will handle exits internally."
        )
        return True  # Entry succeeded; exits handled by strategy logic

    fill_price = float(fill_price_raw)
    logger.info(f"Market order filled at {fill_price:.2f} (order_id={order_id})")

    # Place bracket order using actual fill price (not historical candle price)
    # FIXED: added product_id=PRODUCT_ID; replaced hardcoded 0.7/1.2 with config values
    bracket_response = place_bracket_order(
        product_id=PRODUCT_ID,
        order_id=order_id,
        side=side,
        fill_price=fill_price,
        stop_loss_pct=STOP_LOSS_PCT,
        take_profit_pct=TAKE_PROFIT_PCT,
    )

    if bracket_response and bracket_response.get("success"):
        logger.info(f"Bracket order placed successfully for order {order_id}")
    else:
        logger.warning(
            f"Bracket order failed for order {order_id}; "
            "strategy will handle exits internally."
        )

    return True


def exit_position(side: str, size: int, exit_reason: str, dry_run: bool) -> None:
    """
    Cancel all open orders (clears bracket) then place market exit order.
    """
    if dry_run:
        logger.info(
            f"[DRY-RUN] Would place EXIT ({exit_reason}): "
            f"{'sell' if side == 'buy' else 'buy'} {size} lot(s) {SYMBOL}"
        )
        return

    # Cancel bracket orders before manual exit to avoid double-close
    cancel_all_orders()

    close_side = "sell" if side == "buy" else "buy"
    place_market_order(close_side, size)


# -----------------------------------------------------------------------------
# Simulation loop
# -----------------------------------------------------------------------------


def run_simulation(
    days: int,
    delay: float,
    size: int,
    dry_run: bool,
    strategy: BollingerBandsStrategy,
) -> None:
    """
    Main simulation loop. Replays historical candles one by one,
    runs strategy signal logic, and places orders via order_manager.
    """
    candles = fetch_historical_candles(days)
    if not candles:
        return

    min_candles = strategy.get_min_candles()
    total = len(candles)

    logger.info("=" * 60)
    logger.info("  Simulation Bot Starting")
    logger.info(f"  Symbol   : {SYMBOL}")
    logger.info(f"  Strategy : {strategy.name()}")
    logger.info(f"  Candles  : {total} ({days} days)")
    logger.info(f"  Warmup   : {min_candles} candles")
    logger.info(f"  Lot size : {size} lot(s) = {size * CONTRACT_VALUE_BTC:.3f} BTC")
    logger.info(f"  Delay    : {delay}s per candle")
    logger.info(f"  Dry run  : {dry_run}")
    logger.info("=" * 60)

    position = None  # None = flat, dict = open position
    trade_count = 0
    win_count = 0
    total_pnl = 0.0

    for i, candle in enumerate(candles):
        candle_time = datetime.fromtimestamp(candle["time"], tz=timezone.utc)

        # Build rolling window up to and including current candle (max 200)
        window = candles[max(0, i - 199) : i + 1]
        df = pd.DataFrame(window)

        # Compute indicators on rolling window
        df = strategy.calculate_indicators(df)

        # Skip until warmup complete
        if len(df) < min_candles:
            logger.debug(f"[{candle_time}] Warming up ({len(df)}/{min_candles})")
            if delay:
                time.sleep(delay)
            continue

        # Extract latest indicator values
        latest = df.iloc[-1]
        indicators = {
            "bb_upper": latest.get("bb_upper"),
            "bb_lower": latest.get("bb_lower"),
            "bb_mid": latest.get("bb_mid"),
            "volume_avg": latest.get("volume_avg"),
        }

        candle_dict = {
            "open": candle["open"],
            "high": candle["high"],
            "low": candle["low"],
            "close": candle["close"],
            "volume": candle["volume"],
            "time": str(candle_time),
        }

        # -----------------------------------------------------------------
        # Exit check (if in position)
        # Note: strategy exit logic runs first; bracket order on exchange
        # may have already closed the position via SL/TP. The exit here
        # handles band-flip and trail exits that bracket does not cover.
        # -----------------------------------------------------------------
        if position is not None:
            exit_side, exit_reason = strategy.get_exit_signal(
                candle_dict, indicators, position
            )

            if exit_side is not None:
                entry = position["entry_price"]
                exit_price = candle_dict["close"]

                if position["side"] == "buy":
                    pnl = (exit_price - entry) / entry * size * CONTRACT_VALUE_BTC
                else:
                    pnl = (entry - exit_price) / entry * size * CONTRACT_VALUE_BTC

                total_pnl += pnl
                trade_count += 1
                if pnl > 0:
                    win_count += 1

                logger.info(
                    f"[{candle_time}] EXIT {exit_reason} | "
                    f"side={position['side']} "
                    f"entry={entry:.2f} exit={exit_price:.2f} | "
                    f"pnl={pnl:+.6f} BTC | "
                    f"running_pnl={total_pnl:+.6f} BTC"
                )

                exit_position(position["side"], size, exit_reason, dry_run)
                position = None

        # -----------------------------------------------------------------
        # Entry check (if flat)
        # -----------------------------------------------------------------
        if position is None:
            signal = strategy.get_signal(candle_dict, indicators)

            if signal in ("buy", "sell"):
                entry_price = candle_dict["close"]

                logger.info(
                    f"[{candle_time}] ENTRY {signal.upper()} | "
                    f"price={entry_price:.2f} | "
                    f"bb_upper={indicators['bb_upper']:.2f} | "
                    f"bb_lower={indicators['bb_lower']:.2f} | "
                    f"size={size} lot(s)"
                )

                success = enter_position(signal, size, entry_price, dry_run)

                if success:
                    position = {
                        "side": signal,
                        "entry_price": entry_price,
                        "best_price": entry_price,
                        "size": size,
                        "entry_time": str(candle_time),
                    }
            else:
                logger.debug(
                    f"[{candle_time}] No signal. "
                    f"close={candle_dict['close']:.2f} | "
                    f"bb_upper={indicators['bb_upper']:.2f} | "
                    f"bb_lower={indicators['bb_lower']:.2f}"
                )

        if delay:
            time.sleep(delay)

    # -----------------------------------------------------------------
    # Simulation summary
    # -----------------------------------------------------------------
    win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0.0

    logger.info("=" * 60)
    logger.info("  Simulation Complete")
    logger.info(f"  Total trades  : {trade_count}")
    logger.info(f"  Wins          : {win_count}")
    logger.info(f"  Win rate      : {win_rate:.1f}%")
    logger.info(f"  Total PnL     : {total_pnl:+.6f} BTC")
    logger.info(
        f"  Open position : "
        f"{'YES - ' + position['side'] + ' @ ' + str(position['entry_price']) if position else 'None (flat)'}"
    )
    logger.info("=" * 60)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulated Live Bot - Candle Replay Mode"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Log signals only, do not place real orders (default: False)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=False,
        help="No delay between candles (default: False, uses 1s delay)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of historical candles to replay (default: 30)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1,
        help="Order size in lots (default: 1 lot = 0.001 BTC)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    delay = 0.0 if args.fast else 1.0

    # Isolated DB - never touches live bot state
    strategy = BollingerBandsStrategy(
        period=20,
        std=2.5,
        trail_activation_pct=1.0,
        trail_pct=0.8,
        stop_loss_pct=0.7,  # Fixed: was 1.5, corrected to match config.py
        take_profit_pct=1.2,
        tp_extension_pct=0.3,
        volume_period=20,
        volume_multiplier=1.2,
        db_path="sim_strategy_state.db",
    )

    run_simulation(
        days=args.days,
        delay=delay,
        size=args.size,
        dry_run=args.dry_run,
        strategy=strategy,
    )
