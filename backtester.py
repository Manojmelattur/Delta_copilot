# =============================================================================
# backtester.py - Generic Backtest Engine (works with any strategy)
# =============================================================================

import os
import time

import matplotlib.dates as mdates

# matplotlib.use("Agg")  # Add this BEFORE importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from config import (
    BACKTEST_CHART_FILE,
    BACKTEST_DAYS,
    ORDER_SIZE,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    TAKER_FEE,
)
from logger import get_logger
from report import generate_all_reports, print_comparison_table

logger = get_logger(__name__)

PRODUCTION_CANDLE_URL = "https://api.india.delta.exchange/v2/history/candles"

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

BATCH_SIZE = 1900

# -----------------------------------------------------------------------------
# Contract value per lot per symbol (in BTC equivalent for PnL calculation)
# Add new symbols here as needed.
# -----------------------------------------------------------------------------
CONTRACT_VALUE_MAP = {
    "BTCUSD": 0.001,  # 1 lot = 0.001 BTC
    "ETHUSD": 0.01,  # 1 lot = 0.01 ETH  (verify via getProductBySymbol)
    "XAUTUSD": 1.0,  # 1 lot = 1 unit    (verify via getProductBySymbol)
}


def fetch_historical_candles(
    days: int = BACKTEST_DAYS, timeframe: str = "15m", symbol: str = "BTCUSD"
) -> pd.DataFrame:
    """
    Fetch historical candles from production API in batches.
    Always uses production URL (testnet has no historical data).
    """
    resolution = RESOLUTION_MAP.get(timeframe, "15m")
    interval_seconds = INTERVAL_SECONDS_MAP.get(resolution, 900)

    end_time = int(time.time())
    start_time = end_time - (days * 86400)

    all_candles = []
    batch_end = end_time

    logger.info(
        f"Fetching {days} days of {timeframe} candles for {symbol} "
        f"in batches of {BATCH_SIZE}..."
    )

    while batch_end > start_time:
        batch_start = batch_end - (BATCH_SIZE * interval_seconds)
        if batch_start < start_time:
            batch_start = start_time

        params = {
            "resolution": resolution,
            "symbol": symbol,
            "start": batch_start,
            "end": batch_end,
        }

        try:
            response = requests.get(
                PRODUCTION_CANDLE_URL, params=params, timeout=(3, 27)
            ).json()

            candles = response.get("result", [])

            if not candles:
                logger.warning(f"Empty batch at end={batch_end}. Stopping fetch.")
                break

            all_candles.extend(candles)
            logger.debug(f"Fetched {len(candles)} candles (batch_end={batch_end})")

            batch_end = batch_start - interval_seconds
            time.sleep(0.3)

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


def _get_candle_and_indicators(df: pd.DataFrame, i: int) -> tuple[dict, dict]:
    """
    Extract candle dict and indicators dict from DataFrame row i.

    Passes ALL computed columns as indicators so any strategy can access
    its own indicator columns without requiring backtester changes.

    Special handling:
      - current_idx        : always injected for hold-period guards
      - prev_ema_fast/slow : previous row EMA values for crossover detection
      - prev_close         : previous row close for VWAP crossover detection
      - prev_vwap          : previous row VWAP for VWAP crossover detection
    """
    row = df.iloc[i]
    candle_dict = row.to_dict()

    # ------------------------------------------------------------------
    # Start with ALL DataFrame columns as indicators.
    # This makes the backtester strategy-agnostic - any new strategy
    # (FVG, OB, RSI, MACD, etc.) gets its computed columns automatically
    # without requiring backtester changes.
    # ------------------------------------------------------------------
    indicators_dict = row.to_dict()

    # Always inject current candle index for hold-period guards
    indicators_dict["current_idx"] = i

    if i > 0:
        prev_row = df.iloc[i - 1]

        # EMA crossover: inject previous row values for crossover detection
        for col in ["ema_fast", "ema_slow"]:
            if col in df.columns:
                indicators_dict[f"prev_{col}"] = prev_row[col]

        # VWAP crossover: inject previous close and previous vwap
        indicators_dict["prev_close"] = prev_row["close"]
        if "vwap" in df.columns:
            indicators_dict["prev_vwap"] = prev_row["vwap"]
        else:
            indicators_dict["prev_vwap"] = None

    else:
        for col in ["ema_fast", "ema_slow"]:
            indicators_dict[f"prev_{col}"] = None
        indicators_dict["prev_close"] = None
        indicators_dict["prev_vwap"] = None

    return candle_dict, indicators_dict


def _resolve_exit_price(
    exit_reason: str,
    position: dict,
    candle_dict: dict,
    strategy,
) -> float:
    """
    Derive the actual exit price from the exit reason.

    ATR-based strategies (e.g. OrderBlockStrategy, SMCStrategy):
        SL/TP prices are pre-computed at entry time and stored in the
        position dict under 'sl_price' and 'tp_price'. Read directly.

    Percentage-based strategies (FVG, BB, EMA, VWAP, TripleEMA):
        Strategy attributes (stop_loss_pct, take_profit_pct, trail_pct)
        are already stored as decimals by strategy __init__.
        Do NOT divide by 100 here.
    """
    entry = position["entry_price"]
    side = position["side"]
    best_price = position.get("best_price", entry)

    # ------------------------------------------------------------------
    # ATR-based path: OB/SMC strategies store sl_price/tp_price in
    # position dict at entry time. Read directly instead of recomputing.
    # All other strategies have sl_price=None so this block is skipped.
    # ------------------------------------------------------------------
    if position.get("sl_price") is not None and position.get("tp_price") is not None:
        sl_price = float(position["sl_price"])
        tp_price = float(position["tp_price"])
        atr = float(position.get("atr_at_entry", 0))
        atr_mult = getattr(strategy, "atr_multiplier", 1.5)

        if exit_reason == "EXIT_TP":
            return round(tp_price, 1)

        elif exit_reason == "EXIT_SL":
            return round(sl_price, 1)

        elif exit_reason == "EXIT_TRAIL":
            trail_distance = atr_mult * atr
            if side == "buy":
                return round(best_price - trail_distance, 1)
            else:
                return round(best_price + trail_distance, 1)

        else:
            return candle_dict["close"]

    # ------------------------------------------------------------------
    # Percentage-based path: FVG, BB, EMA, VWAP, TripleEMA all use this.
    # Attributes are stored as decimals (e.g. 0.01 = 1%) by __init__.
    # ------------------------------------------------------------------
    if exit_reason == "EXIT_TP":
        if side == "buy":
            return round(entry * (1 + strategy.take_profit_pct), 1)
        else:
            return round(entry * (1 - strategy.take_profit_pct), 1)

    elif exit_reason == "EXIT_SL":
        if side == "buy":
            return round(entry * (1 - strategy.stop_loss_pct), 1)
        else:
            return round(entry * (1 + strategy.stop_loss_pct), 1)

    elif exit_reason == "EXIT_TRAIL":
        if side == "buy":
            return round(best_price * (1 - strategy.trail_pct), 1)
        else:
            return round(best_price * (1 + strategy.trail_pct), 1)

    else:
        return candle_dict["close"]


def _notify_entry(strategy, idx: int) -> None:
    """Call strategy.notify_entry(idx) if the strategy supports it."""
    if hasattr(strategy, "notify_entry"):
        strategy.notify_entry(idx)


def _notify_exit(strategy) -> None:
    """Call strategy.notify_exit() if the strategy supports it."""
    if hasattr(strategy, "notify_exit"):
        strategy.notify_exit()


def _build_position(signal: str, candle_dict: dict, i: int, strategy=None) -> dict:
    """
    Build a position dict from an entry signal.

    Supports two key formats from get_last_entry_levels():
      - OB strategy  : _ob_sl_price, _ob_tp_price, _ob_atr_at_entry
      - SMC strategy : sl, tp  (no atr_at_entry needed)

    For all other strategies: get_last_entry_levels() is not defined so
    .get() returns None safely - zero impact on existing strategies.
    """
    ob_levels = {}
    if strategy is not None and hasattr(strategy, "get_last_entry_levels"):
        levels = strategy.get_last_entry_levels()
        if levels is not None:
            ob_levels = levels

    # Support OB key format (_ob_sl_price) and SMC key format (sl, tp)
    sl_price = (
        ob_levels.get("_ob_sl_price")
        or candle_dict.get("_ob_sl_price")
        or ob_levels.get("sl")
    )
    tp_price = (
        ob_levels.get("_ob_tp_price")
        or candle_dict.get("_ob_tp_price")
        or ob_levels.get("tp")
    )
    atr_at_entry = (
        ob_levels.get("_ob_atr_at_entry")
        or candle_dict.get("_ob_atr_at_entry")
        or ob_levels.get("atr_at_entry")
    )

    return {
        "side": "buy" if signal == "buy" else "sell",
        "entry_price": candle_dict["close"],
        "entry_idx": i,
        "best_price": candle_dict["close"],
        "peak_price": candle_dict["close"],  # <-- added
        "sl_price": sl_price,
        "tp_price": tp_price,
        "atr_at_entry": atr_at_entry,
    }


def run_backtest(
    strategy,
    df: pd.DataFrame = None,
    days: int = BACKTEST_DAYS,
    timeframe: str = "15m",
    symbol: str = "BTCUSD",
):
    if df is None or df.empty:
        df = fetch_historical_candles(days=days, timeframe=timeframe, symbol=symbol)

    if df.empty:
        logger.error("No data available for backtest.")
        return [], {}

    logger.info(
        f"Running backtest: {strategy.name} | symbol={symbol} | timeframe={timeframe}"
    )

    # -----------------------------------------------------------------
    # IMPORTANT: reset_state() MUST come before calculate_indicators()
    # so that self._df is set by calculate_indicators and not wiped
    # by reset_state() afterwards.
    # -----------------------------------------------------------------
    if hasattr(strategy, "reset_state"):
        strategy.reset_state()
    else:
        if hasattr(strategy, "_pending_signal"):
            strategy._pending_signal = None
        _notify_exit(strategy)

    df = strategy.calculate_indicators(df)

    # Reset hold-period state
    _notify_exit(strategy)

    # Check if strategy has custom exit/re-entry logic
    has_custom_exit = hasattr(strategy, "get_exit_signal")
    has_custom_reentry = hasattr(strategy, "get_reentry_signal")

    trades = []
    position = None
    start_idx = strategy.get_min_candles()

    # --- Re-entry state tracking ---
    last_exit_reason = None
    last_exit_side = None
    candles_since_exit = 0
    reentry_count = 0

    for i in range(start_idx, len(df)):
        candle_dict, indicators_dict = _get_candle_and_indicators(df, i)

        # --- Increment candles since exit counter ---
        if position is None and last_exit_reason is not None:
            candles_since_exit += 1

        # =========================================================
        # EXIT LOGIC
        # =========================================================
        if position is not None:
            if has_custom_exit:
                exit_side, exit_reason = strategy.get_exit_signal(
                    candle_dict, indicators_dict, position
                )

                if exit_side is not None:
                    exit_price = _resolve_exit_price(
                        exit_reason, position, candle_dict, strategy
                    )
                    trades.append(
                        _build_trade(
                            position, df, i, exit_price, exit_reason, symbol=symbol
                        )
                    )
                    last_exit_reason = exit_reason
                    last_exit_side = position["side"]
                    candles_since_exit = 0
                    position = None
                    _notify_exit(strategy)
                    continue

            else:
                # --- Generic SL/TP fallback for simple strategies ---
                entry = position["entry_price"]
                side = position["side"]

                if side == "buy":
                    sl_price = round(entry * (1 - STOP_LOSS_PCT), 1)
                    tp_price = round(entry * (1 + TAKE_PROFIT_PCT), 1)
                    hit_sl = candle_dict["low"] <= sl_price
                    hit_tp = candle_dict["high"] >= tp_price
                else:
                    sl_price = round(entry * (1 + STOP_LOSS_PCT), 1)
                    tp_price = round(entry * (1 - TAKE_PROFIT_PCT), 1)
                    hit_sl = candle_dict["high"] >= sl_price
                    hit_tp = candle_dict["low"] <= tp_price

                exit_price = None
                exit_reason = None

                if hit_tp:
                    exit_price = tp_price
                    exit_reason = "TP"
                elif hit_sl:
                    exit_price = sl_price
                    exit_reason = "SL"

                if exit_price is not None:
                    trades.append(
                        _build_trade(
                            position, df, i, exit_price, exit_reason, symbol=symbol
                        )
                    )
                    last_exit_reason = exit_reason
                    last_exit_side = position["side"]
                    candles_since_exit = 0
                    position = None
                    _notify_exit(strategy)
                    continue

        # =========================================================
        # ENTRY LOGIC
        # =========================================================
        if position is None:
            signal = None

            # --- Check re-entry first (higher priority than fresh signal) ---
            if (
                has_custom_reentry
                and last_exit_reason is not None
                and candles_since_exit > 0
            ):
                # signal = strategy.get_reentry_signal(
                #     df,
                #     i,
                #     last_exit_reason,
                #     last_exit_side,
                #     candles_since_exit,
                #     reentry_count,
                # )
                # AFTER (correct - passing candle_dict and indicators_dict)
                signal = strategy.get_reentry_signal(
                    candle_dict,
                    indicators_dict,
                    last_exit_reason,
                    last_exit_side,
                    candles_since_exit,
                    reentry_count,
                )

                if signal is not None:
                    reentry_count += 1
                    last_exit_reason = None
                    candles_since_exit = 0

            # --- Fresh signal if no re-entry ---
            if signal is None:
                signal = strategy.get_signal(candle_dict, indicators_dict)
                if signal is not None:
                    reentry_count = 0
                    last_exit_reason = None
                    candles_since_exit = 0

            if signal is not None:
                position = _build_position(signal, candle_dict, i, strategy=strategy)
                _notify_entry(strategy, i)

        # --- Close existing on signal flip (only for fresh signals) ---
        elif position is not None:
            signal = strategy.get_signal(candle_dict, indicators_dict)
            if signal is not None:
                current_side = position["side"]
                signal_side = "buy" if signal == "buy" else "sell"

                if current_side != signal_side:
                    exit_price = candle_dict["open"]
                    trades.append(
                        _build_trade(
                            position, df, i, exit_price, "signal_flip", symbol=symbol
                        )
                    )
                    last_exit_reason = "signal_flip"
                    last_exit_side = current_side
                    candles_since_exit = 0
                    reentry_count = 0
                    _notify_exit(strategy)

                    position = _build_position(
                        signal, candle_dict, i, strategy=strategy
                    )
                    _notify_entry(strategy, i)

    summary = _print_summary(trades, df, strategy)
    return trades, summary


def run_all_backtests(
    strategies: list,
    df: pd.DataFrame = None,
    days: int = BACKTEST_DAYS,
    timeframe: str = "15m",
    symbol: str = "BTCUSD",
):
    if df is None or df.empty:
        print("\n[INFO] Fetching historical data (shared across all strategies)...")
        df = fetch_historical_candles(days=days, timeframe=timeframe, symbol=symbol)

    if df.empty:
        logger.error("No data available.")
        return {}

    # FIXED - _get_strategy_name() resolves clean name before storing
    results = {}
    for strategy in strategies:
        name = _get_strategy_name(strategy)[:15].strip()  # <-- resolve ONCE here
        print(f"\n{'=' * 55}")
        print(f"  Running: {name}")  # <-- use resolved name
        print(f"{'=' * 55}")
        trades, summary = run_backtest(
            strategy, df=df.copy(), days=days, timeframe=timeframe, symbol=symbol
        )
        results[name] = (trades, summary)  # <-- store with resolved name

    # --- Replace old _print_comparison_table call with full report generator ---
    generate_all_reports(results, symbol=symbol, timeframe=timeframe)
    _generate_comparison_chart(results, df, symbol=symbol, timeframe=timeframe)
    print_comparison_table(results)

    return results


# =============================================================================
# Internal Helpers
# =============================================================================


def _build_trade(
    position: dict,
    df: pd.DataFrame,
    exit_idx: int,
    exit_price: float,
    exit_reason: str,
    symbol: str = "BTCUSD",
) -> dict:
    """
    Build a trade record dict. PnL is calculated using the symbol-specific
    contract value from CONTRACT_VALUE_MAP.
    Falls back to BTCUSD contract value (0.001) if symbol not found in map.
    """
    entry = position["entry_price"]
    side = position["side"]

    contract_value = CONTRACT_VALUE_MAP.get(symbol, 0.001)
    size_units = ORDER_SIZE * contract_value

    if side == "buy":
        gross_pnl_usd = size_units * (exit_price - entry)
    else:
        gross_pnl_usd = size_units * (entry - exit_price)

    commission_usd = size_units * (entry + exit_price) * TAKER_FEE

    avg_price = (entry + exit_price) / 2
    gross_pnl_btc = gross_pnl_usd / avg_price
    commission_btc = commission_usd / avg_price
    net_pnl_btc = gross_pnl_btc - commission_btc

    return {
        "entry_time": df.iloc[position["entry_idx"]]["timestamp"],
        "exit_time": df.iloc[exit_idx]["timestamp"],
        "side": side,
        "entry_price": entry,
        "exit_price": exit_price,
        "gross_pnl": round(gross_pnl_btc, 6),
        "commission": round(commission_btc, 6),
        "net_pnl": round(net_pnl_btc, 6),
        "exit_reason": exit_reason,
    }


def _calculate_max_drawdown(trades: list) -> float:
    if not trades:
        return 0.0
    cumulative = np.cumsum([t["net_pnl"] for t in trades])
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    return float(np.max(drawdown))


def _print_summary(trades: list, df: pd.DataFrame, strategy) -> dict:
    """Print and return backtest performance summary with exit reason breakdown."""
    if not trades:
        print(f"\n[{strategy.name}] No trades generated.")
        return {}

    total_trades = len(trades)
    wins = [t for t in trades if t["net_pnl"] > 0]
    losses = [t for t in trades if t["net_pnl"] <= 0]
    total_pnl = sum(t["net_pnl"] for t in trades)
    win_rate = len(wins) / total_trades * 100
    avg_win = np.mean([t["net_pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["net_pnl"] for t in losses]) if losses else 0
    max_drawdown = _calculate_max_drawdown(trades)

    # --- Exit reason breakdown ---
    all_reasons = sorted(set(t["exit_reason"] for t in trades))
    exit_breakdown = {}
    for reason in all_reasons:
        reason_trades = [t for t in trades if t["exit_reason"] == reason]
        reason_wins = [t for t in reason_trades if t["net_pnl"] > 0]
        reason_losses = [t for t in reason_trades if t["net_pnl"] <= 0]
        reason_pnl = sum(t["net_pnl"] for t in reason_trades)
        reason_avg_win = (
            np.mean([t["net_pnl"] for t in reason_wins]) if reason_wins else 0
        )
        reason_avg_loss = (
            np.mean([t["net_pnl"] for t in reason_losses]) if reason_losses else 0
        )
        exit_breakdown[reason] = {
            "count": len(reason_trades),
            "wins": len(reason_wins),
            "losses": len(reason_losses),
            "total_pnl": round(reason_pnl, 6),
            "avg_win": round(reason_avg_win, 6),
            "avg_loss": round(reason_avg_loss, 6),
        }

    summary = {
        "strategy": str(strategy),
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
        "exit_breakdown": exit_breakdown,
    }

    print(f"\n{'=' * 55}")
    print(f"  {strategy.name} - BACKTEST RESULTS")
    print(f"  Params : {strategy.get_params()}")
    print(f"{'=' * 55}")
    print(f"  Period       : {summary['period_start']} to {summary['period_end']}")
    print(f"  Total Trades : {total_trades}")
    print(f"  Wins / Losses: {len(wins)} / {len(losses)}")
    print(f"  Win Rate     : {win_rate:.1f}%")
    print(f"  Total PnL    : {total_pnl:.6f} BTC")
    print(f"  Avg Win      : {avg_win:.6f} BTC")
    print(f"  Avg Loss     : {avg_loss:.6f} BTC")
    print(f"  Max Drawdown : {max_drawdown:.6f} BTC")
    print(f"{'=' * 55}")

    # --- Print exit reason breakdown table ---
    print(f"\n  EXIT REASON BREAKDOWN")
    print(f"  {'-' * 71}")
    print(
        f"  {'Reason':<16} {'Count':>6} {'Wins':>6} {'Losses':>7} "
        f"{'Total PnL':>12} {'Avg Win':>10} {'Avg Loss':>10}"
    )
    print(f"  {'-' * 71}")
    for reason, stats in exit_breakdown.items():
        pnl_str = f"{stats['total_pnl']:+.4f}"
        avg_win_str = f"{stats['avg_win']:+.4f}" if stats["wins"] > 0 else "   N/A"
        avg_loss_str = f"{stats['avg_loss']:+.4f}" if stats["losses"] > 0 else "   N/A"
        print(
            f"  {reason:<16} {stats['count']:>6} {stats['wins']:>6} {stats['losses']:>7} "
            f"{pnl_str:>12} {avg_win_str:>10} {avg_loss_str:>10}"
        )
    print(f"  {'-' * 71}")
    print(
        f"  {'TOTAL':<16} {total_trades:>6} {len(wins):>6} {len(losses):>7} "
        f"{total_pnl:>+12.4f}"
    )
    print(f"{'=' * 55}\n")

    return summary


import re


def _get_strategy_name(strategy) -> str:
    """
    Safely resolve strategy name whether it is a @property or plain method.
    Extracts clean name from bound method string if needed.
    """
    name = strategy.name

    # If it is a callable (missing @property), call it first
    if callable(name):
        name = name()

    name = str(name)

    # If still looks like a bound method string, extract from it
    if name.startswith("<bound method"):
        match = re.search(r"<bound method (\w+)\.name", name)
        if match:
            class_name = match.group(1)
            class_name = re.sub(r"Strategy$", "", class_name)
            name = re.sub(r"(?<!^)(?=[A-Z])", " ", class_name).strip()

    return name


def _print_comparison_table(results: dict):
    # Delegated to report.py — kept for backward compatibility

    print_comparison_table(results)


def _generate_comparison_chart(
    results: dict,
    df: pd.DataFrame,
    symbol: str = "BTCUSD",
    timeframe: str = "15m",
):
    try:
        fig, axes = plt.subplots(4, 1, figsize=(16, 18))
        fig.suptitle(
            f"Multi-Strategy Backtest Comparison - {symbol} {timeframe}",
            fontsize=14,
            fontweight="bold",
        )

        colors = ["blue", "green", "orange", "red"]

        # --- Panel 1: Price Chart ---
        axes[0].plot(
            df["timestamp"],
            df["close"],
            color="gray",
            linewidth=0.8,
            alpha=0.8,
            label=f"{symbol} Close",
        )
        axes[0].set_title(f"{symbol} Price")
        axes[0].set_ylabel("Price (USD)")
        axes[0].legend(fontsize=8)
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

        # --- Panel 2: Equity Curves ---
        for idx, (name, (trades, summary)) in enumerate(results.items()):
            if not trades:
                continue
            color = colors[idx % len(colors)]
            cum_pnl = np.cumsum([t["net_pnl"] for t in trades])
            trade_times = [t["exit_time"] for t in trades]
            pnl_label = f"{str(name)} ({summary['total_pnl_btc']:+.4f} BTC)"  # <-- Fix
            axes[1].plot(
                trade_times, cum_pnl, color=color, linewidth=1.5, label=pnl_label
            )

        axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[1].set_title("Cumulative PnL - All Strategies (BTC)")
        axes[1].set_ylabel("BTC")
        axes[1].legend(fontsize=8)
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

        # --- Panel 3: Win Rate Bars ---
        names = [str(n) for n in results.keys()]  # <-- Fix: cast all keys to str
        win_rates = [results[n][1].get("win_rate", 0) for n in results.keys()]
        bar_colors = [colors[i % len(colors)] for i in range(len(names))]
        bars = axes[2].bar(names, win_rates, color=bar_colors, alpha=0.7)
        axes[2].axhline(
            50, color="black", linewidth=0.8, linestyle="--", label="50% line"
        )
        axes[2].set_title("Win Rate by Strategy (%)")
        axes[2].set_ylabel("Win Rate (%)")
        axes[2].set_ylim(0, 100)
        for bar, val in zip(bars, win_rates):
            axes[2].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # --- Panel 4: Total PnL Bars ---
        total_pnls = [results[n][1].get("total_pnl_btc", 0) for n in results.keys()]
        pnl_colors = ["green" if p >= 0 else "red" for p in total_pnls]
        bars2 = axes[3].bar(
            names, total_pnls, color=pnl_colors, alpha=0.7
        )  # <-- uses fixed names
        axes[3].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[3].set_title("Total Net PnL by Strategy (BTC)")
        axes[3].set_ylabel("BTC")
        for bar, val in zip(bars2, total_pnls):
            label = f"{val:+.4f}"
            y_pos = bar.get_height() + 0.0001 if val >= 0 else bar.get_height() - 0.0003
            axes[3].text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        os.makedirs(os.path.dirname(BACKTEST_CHART_FILE), exist_ok=True)
        plt.savefig(BACKTEST_CHART_FILE, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Comparison chart saved to {BACKTEST_CHART_FILE}")
        print(f"[CHART] Saved to: {BACKTEST_CHART_FILE}")

    except Exception as e:
        logger.error(f"Error generating comparison chart: {e}")
