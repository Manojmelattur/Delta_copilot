# =============================================================================
# bot.py - Strategy-Aware Live Bot Loop (with CLI support)
# =============================================================================

import argparse
import os
import sys
import time
from datetime import date, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging

import config
from config import (
    BACKTEST_DAYS,
    ORDER_SIZE,
    POLL_INTERVAL,
    PRODUCTION_BASE_URL,
    STOP_LOSS_PCT,
    SYMBOL,
    TAKE_PROFIT_PCT,
    TAKER_FEE,
    TESTNET_BASE_URL,
)
from logger import get_logger
from market_data import fetch_candles
from notifier import (
    notify_error,
    notify_trade_entry,
    notify_trade_exit,
)
from order_manager import cancel_all_orders, place_market_order
from pnl_tracker import generate_pnl_chart, record_trade, send_daily_report
from position_manager import get_open_position
from strategies import STRATEGY_REGISTRY, get_strategy

logging.getLogger("order_manager").setLevel(logging.DEBUG)
logger = get_logger(__name__)

CANDLE_LIMIT = 200

VALID_TIMEFRAMES = [
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "12h",
    "1d",
    "1w",
    "1s",
    "5s",
]

_OHLCV_COLUMNS = {"open", "high", "low", "close", "volume", "time", "timestamp"}


# =============================================================================
# CLI Argument Parser
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Delta Exchange - Strategy-Aware Live Bot",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python bot.py --strategy bollinger_bands --symbol BTCUSD
  python bot.py --strategy bollinger_bands --symbol ETHUSD --timeframe 1h
  python bot.py --strategy bollinger_bands --symbol BTCUSD --timeframe 4h --bb-take-profit 1.2 --bb-stop-loss 0.6
  python bot.py --strategy bollinger_bands --symbol BTCUSD --timeframe 15m --period 20 --std 2.5 --bb-stop-loss 0.7
  python bot.py --strategy ema_crossover --symbol ETHUSD --timeframe 1h --fast 9 --slow 21
  python bot.py --strategy triple_ema --symbol BTCUSD --timeframe 4h --fast 5 --mid 13 --slow 21
  python bot.py --strategy fvg --symbol BTCUSD --timeframe 1h --fvg-stop-loss 1.0 --fvg-trail-activation 1.0 --fvg-trail 0.15 --fvg-take-profit 2.0 --fvg-min-size 0.10
  python bot.py --strategy ob --symbol BTCUSD --timeframe 15m
  python bot.py --strategy smc --symbol BTCUSD --timeframe 15m

Available strategies:
  bollinger_bands  - Bollinger Bands mean reversion
  ema_crossover    - EMA fast/slow crossover
  triple_ema       - Triple EMA alignment
  vwap             - Rolling VWAP crossover
  fvg              - Fair Value Gap retracement
  ob               - Order Block trend continuation
  smc              - Smart Money Concepts

Available timeframes:
  1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 1w
        """,
    )

    # --- API ---
    parser.add_argument(
        "--api-key",
        type=str,
        required=False,
        help="API key for trading -- if not provided, credentials are read from environment variables",
    )
    parser.add_argument(
        "--api-secret",
        type=str,
        required=False,
        help="API secret for trading -- if not provided, credentials are read from environment variables",
    )

    # --- Mode and lot size ---
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["live", "paper"],
        help="Trade mode: live (production) or paper (testnet)",
    )
    parser.add_argument(
        "--lot",
        type=int,
        required=True,
        help="Lot size",
    )

    # --- Strategy selection ---
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=list(STRATEGY_REGISTRY.keys()),
        help="Strategy to run",
    )

    # --- SMC Strategy arguments ---
    parser.add_argument(
        "--smc-swing-lookback",
        type=int,
        default=5,
        help="Candles each side for swing detection",
    )
    parser.add_argument("--smc-atr-length", type=int, default=14, help="ATR period")
    parser.add_argument(
        "--smc-atr-mult", type=float, default=1.5, help="ATR trailing multiplier"
    )
    parser.add_argument(
        "--smc-trail-activation",
        type=float,
        default=2.0,
        help="ATR units profit before trail activates",
    )
    parser.add_argument(
        "--smc-sl-buffer",
        type=float,
        default=0.3,
        help="ATR buffer added to OB boundary for SL",
    )
    parser.add_argument(
        "--smc-strength-period", type=int, default=20, help="Period for OB strength ATR"
    )
    parser.add_argument(
        "--smc-strength-mult", type=float, default=2.0, help="OB strength multiplier"
    )
    parser.add_argument(
        "--smc-max-age", type=int, default=30, help="Max OB age in candles"
    )
    parser.add_argument(
        "--smc-rr-ratio", type=float, default=2.0, help="Risk:Reward ratio for TP"
    )
    parser.add_argument(
        "--smc-ema-period", type=int, default=50, help="Trend EMA period"
    )
    parser.add_argument(
        "--smc-min-atr", type=float, default=0.0, help="Minimum ATR threshold to trade"
    )
    parser.add_argument(
        "--smc-bos-type",
        type=str,
        default="close",
        choices=["close", "wick"],
        help="BOS confirmation type",
    )
    parser.add_argument(
        "--smc-choch-type",
        type=str,
        default="close",
        choices=["close", "wick"],
        help="CHoCH confirmation type",
    )
    parser.add_argument(
        "--smc-structure-lookback",
        type=int,
        default=50,
        help="Candles back to track BOS structure",
    )
    parser.add_argument(
        "--smc-ob-proximity-atr",
        type=float,
        default=3.0,
        help="ATR units proximity for OB entry",
    )

    # --- General ---
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=None,
        help="Poll interval in seconds (default: POLL_INTERVAL from config)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="15m",
        choices=VALID_TIMEFRAMES,
        help="Candle timeframe (default: 15m)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help=f"Trading symbol to override config e.g. BTCUSD, ETHUSD (default: {SYMBOL} from config)",
    )

    # --- Bollinger Bands params ---
    parser.add_argument(
        "--period", type=int, default=20, help="BB/VWAP period (default: 20)"
    )
    parser.add_argument(
        "--std", type=float, default=2.2, help="BB std deviation (default: 2.2)"
    )
    parser.add_argument(
        "--bb-stop-loss",
        type=float,
        default=0.6,
        help="BB hard stop loss pct (default: 0.6)",
    )
    parser.add_argument(
        "--bb-take-profit",
        type=float,
        default=1.2,
        help="BB take profit pct (default: 1.2)",
    )
    parser.add_argument(
        "--bb-trail-activation",
        type=float,
        default=1.0,
        help="BB trail activation pct (default: 1.0)",
    )
    parser.add_argument(
        "--bb-trail",
        type=float,
        default=0.8,
        help="BB trailing stop pct (default: 0.8)",
    )
    parser.add_argument(
        "--bb-tp-extension",
        type=float,
        default=0.3,
        help="BB TP extension pct (default: 0.3)",
    )
    parser.add_argument(
        "--bb-volume-period",
        type=int,
        default=20,
        help="BB volume period (default: 20)",
    )
    parser.add_argument(
        "--bb-volume-multiplier",
        type=float,
        default=1.2,
        help="BB volume multiplier (default: 1.2)",
    )
    parser.add_argument(
        "--debug-log",
        action="store_true",
        default=False,
        help="Enable per-poll debug logging of band proximity and signal state.",
    )

    # --- EMA Crossover params ---
    parser.add_argument(
        "--fast", type=int, default=9, help="EMA fast period (default: 9)"
    )
    parser.add_argument(
        "--slow", type=int, default=21, help="EMA slow period (default: 21)"
    )

    # --- Triple EMA params ---
    parser.add_argument(
        "--mid", type=int, default=13, help="Triple EMA mid period (default: 13)"
    )

    # --- FVG params ---
    parser.add_argument(
        "--fvg-min-size",
        type=float,
        default=0.10,
        help="FVG minimum gap size as pct of price (default: 0.10)",
    )
    parser.add_argument(
        "--fvg-max-age",
        type=int,
        default=50,
        help="FVG maximum age in candles before expiry (default: 50)",
    )
    parser.add_argument(
        "--fvg-trail-activation",
        type=float,
        default=1.0,
        help="FVG trail activation pct (default: 1.0)",
    )
    parser.add_argument(
        "--fvg-trail",
        type=float,
        default=0.15,
        help="FVG trailing stop pct (default: 0.15)",
    )
    parser.add_argument(
        "--fvg-stop-loss",
        type=float,
        default=1.0,
        help="FVG hard stop loss pct (default: 1.0)",
    )
    parser.add_argument(
        "--fvg-take-profit",
        type=float,
        default=2.0,
        help="FVG take profit pct (default: 2.0)",
    )
    parser.add_argument(
        "--fvg-tp-extension",
        type=float,
        default=0.3,
        help="FVG TP extension pct once trail activates (default: 0.3)",
    )

    # --- OB params ---
    parser.add_argument(
        "--ob-atr-length",
        type=int,
        default=14,
        help="OB ATR period (default: 14)",
    )
    parser.add_argument(
        "--ob-atr-mult",
        type=float,
        default=1.5,
        help="OB ATR trailing stop multiplier (default: 1.5)",
    )
    parser.add_argument(
        "--ob-trail-activation",
        type=float,
        default=1.5,
        help="OB trail activation in ATR units (default: 1.5)",
    )
    parser.add_argument(
        "--ob-strength-period",
        type=int,
        default=20,
        help="OB strength lookback period (default: 20)",
    )
    parser.add_argument(
        "--ob-strength-mult",
        type=float,
        default=1.5,
        help="OB body strength multiplier (default: 1.5)",
    )
    parser.add_argument(
        "--ob-rr-ratio",
        type=float,
        default=3.0,
        help="OB risk:reward ratio (default: 3.0)",
    )
    parser.add_argument(
        "--ob-sl-buffer",
        type=float,
        default=0.1,
        help="OB SL ATR buffer (default: 0.1)",
    )
    parser.add_argument(
        "--ob-max-age",
        type=int,
        default=50,
        help="OB max age in candles before expiry (default: 50)",
    )

    return parser.parse_args()


def build_strategy_params(args) -> dict:
    """Build strategy params dict from parsed CLI args."""
    if args.strategy == "bollinger_bands":
        return {
            "period": args.period,
            "std": args.std,
            "stop_loss_pct": args.bb_stop_loss,
            "take_profit_pct": args.bb_take_profit,
            "trail_activation_pct": args.bb_trail_activation,
            "trail_pct": args.bb_trail,
            "tp_extension_pct": args.bb_tp_extension,
            "volume_period": args.bb_volume_period,
            "volume_multiplier": args.bb_volume_multiplier,
        }
    elif args.strategy == "ema_crossover":
        return {
            "fast": args.fast,
            "slow": args.slow,
        }
    elif args.strategy == "triple_ema":
        return {
            "fast": args.fast,
            "mid": args.mid,
            "slow": args.slow,
        }
    elif args.strategy == "vwap":
        return {
            "period": args.period,
        }
    elif args.strategy == "fvg":
        return {
            "min_fvg_size_pct": args.fvg_min_size,
            "max_fvg_age_candles": args.fvg_max_age,
            "trail_activation_pct": args.fvg_trail_activation,
            "trail_pct": args.fvg_trail,
            "stop_loss_pct": args.fvg_stop_loss,
            "take_profit_pct": args.fvg_take_profit,
            "tp_extension_pct": args.fvg_tp_extension,
        }
    elif args.strategy == "smc":
        return {
            "swing_lookback": args.smc_swing_lookback,
            "atr_length": args.smc_atr_length,
            "atr_multiplier": args.smc_atr_mult,
            "trail_activation_atr": args.smc_trail_activation,
            "sl_atr_buffer": args.smc_sl_buffer,
            "ob_strength_period": args.smc_strength_period,
            "ob_strength_mult": args.smc_strength_mult,
            "max_ob_age_candles": args.smc_max_age,
            "rr_ratio": args.smc_rr_ratio,
            "trend_ema_period": args.smc_ema_period,
            "min_atr_threshold": args.smc_min_atr,
            "bos_type": args.smc_bos_type,
            "choch_type": args.smc_choch_type,
            "structure_lookback": args.smc_structure_lookback,
            "ob_proximity_atr": args.smc_ob_proximity_atr,
        }
    elif args.strategy == "ob":
        return {
            "atr_length": args.ob_atr_length,
            "atr_multiplier": args.ob_atr_mult,
            "trail_activation_atr": args.ob_trail_activation,
            "ob_strength_period": args.ob_strength_period,
            "ob_strength_mult": args.ob_strength_mult,
            "rr_ratio": args.ob_rr_ratio,
            "sl_atr_buffer": args.ob_sl_buffer,
            "max_ob_age_candles": args.ob_max_age,
        }
    return {}


# =============================================================================
# Main Bot Loop
# =============================================================================


def run_bot(
    strategy,
    poll_interval: int,
    timeframe: str = "15m",
    symbol: str = SYMBOL,
    lot: int = 1,
    papertrade: bool = True,
):
    logger.info("=" * 55)
    logger.info(f"  Strategy Bot Starting: {strategy.name()}")
    logger.info(f"  Symbol       : {symbol}")
    logger.info(f"  Poll Interval: {poll_interval}s")
    logger.info(f"  Timeframe    : {timeframe}")
    config.ORDER_SIZE = lot
    logger.info(f"  Lot Size     : {config.ORDER_SIZE}")
    t = "paper" if papertrade else "live"
    logger.info(f"  Live/Paper   : {t}")
    logger.info(f"  Base URL     : {config.getbaseUrl()}")
    logger.info(f"  API Key      : {config.getAPI_KEY()[:6]}******")
    logger.info("=" * 55)

    internal_position = _restore_position_on_startup(symbol)
    if internal_position is not None:
        if hasattr(strategy, "notify_entry"):
            strategy.notify_entry(0)
        logger.warning(
            f"[{symbol}] Restored open {internal_position['side'].upper()} position "
            f"from exchange @ {internal_position['entry_price']}. Resuming management."
        )

    last_report_date = None
    candle_dict = {}
    last_candle_time = None

    MAX_CONSECUTIVE_FAILURES = 3
    consecutive_failures = 0

    while True:
        try:
            loop_start = time.time()

            # --- Daily Report at Midnight ---
            today = date.today()
            if last_report_date != today:
                if last_report_date is not None:
                    send_daily_report()
                last_report_date = today

            # --- Fetch Candles ---
            df = fetch_candles(symbol=symbol, timeframe=timeframe, limit=CANDLE_LIMIT)
            if df.empty:
                consecutive_failures += 1
                logger.warning(
                    f"[{symbol}] No candle data. Skipping iteration. "
                    f"(consecutive failures: {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})"
                )
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    if internal_position is not None:
                        logger.error(
                            f"[{symbol}] Too many consecutive fetch failures with open position. "
                            "Executing emergency exit."
                        )
                        notify_error(
                            f"[{symbol}] Emergency exit triggered: candle fetch failed "
                            f"{MAX_CONSECUTIVE_FAILURES} times with open position."
                        )
                        internal_position = _execute_exit(
                            "EMERGENCY_NETWORK_FAILURE",
                            internal_position,
                            candle_dict,
                            symbol,
                            strategy,
                        )
                    else:
                        logger.error(
                            f"[{symbol}] Too many consecutive fetch failures. "
                            "No open position. Stopping bot to prevent runaway retries."
                        )
                        notify_error(
                            f"[{symbol}] Bot stopping: candle fetch failed "
                            f"{MAX_CONSECUTIVE_FAILURES} consecutive times."
                        )
                    break
                _sleep(loop_start, poll_interval)
                continue

            consecutive_failures = 0

            # --- Compute Indicators ---
            df = strategy.calculate_indicators(df)

            # ------------------------------------------------------------------
            # Use second-to-last candle (index -2):
            #   df.iloc[-1] = currently forming candle (incomplete, skip it)
            #   df.iloc[-2] = last fully closed candle (use this for signals)
            # ------------------------------------------------------------------
            i = len(df) - 2

            if i < strategy.get_min_candles():
                logger.warning(
                    f"[{symbol}] Not enough candles ({i}) for indicators "
                    f"(need {strategy.get_min_candles()}). Waiting..."
                )
                _sleep(loop_start, poll_interval)
                continue

            candle_dict, indicators_dict = _get_candle_and_indicators(df, i)

            # Skip re-evaluation if this candle has already been processed
            current_candle_time = candle_dict.get("timestamp")
            if current_candle_time == last_candle_time:
                logger.debug(
                    f"[{symbol}] Candle unchanged ({current_candle_time}). Skipping evaluation."
                )
                _sleep(loop_start, poll_interval)
                continue
            last_candle_time = current_candle_time

            # =========================================================
            # EXIT LOGIC
            # =========================================================
            if internal_position is not None:
                exit_side, exit_reason = strategy.get_exit_signal(
                    candle_dict, indicators_dict, internal_position
                )

                if exit_side is not None:
                    internal_position = _execute_exit(
                        exit_reason, internal_position, candle_dict, symbol, strategy
                    )
                    _sleep(loop_start, poll_interval)
                    continue

                else:
                    current_price = float(
                        candle_dict.get("close", internal_position["best_price"])
                    )
                    if internal_position["side"] == "buy":
                        if current_price > internal_position["best_price"]:
                            internal_position["best_price"] = current_price
                    else:
                        if current_price < internal_position["best_price"]:
                            internal_position["best_price"] = current_price

                    logger.info(
                        f"[{symbol}] Holding {internal_position['side'].upper()} @ "
                        f"{internal_position['entry_price']} | "
                        f"best_price={internal_position['best_price']:.1f}"
                    )

            # =========================================================
            # ENTRY LOGIC
            # =========================================================
            if internal_position is None:
                signal = strategy.get_signal(candle_dict, indicators_dict)

                if signal is not None:
                    internal_position = _execute_entry(
                        signal, candle_dict, strategy, symbol
                    )
                else:
                    logger.info(f"[{symbol}] No signal. Waiting for next candle.")

        except KeyboardInterrupt:
            logger.warning(
                f"[{symbol}] Bot stopped by user. Closing open position if any..."
            )
            if internal_position is not None:
                logger.warning(
                    f"[{symbol}] Open {internal_position['side'].upper()} position detected. "
                    f"Placing emergency exit order..."
                )
                _execute_exit(
                    "MANUAL_SHUTDOWN", internal_position, candle_dict, symbol, strategy
                )
            generate_pnl_chart()
            send_daily_report()
            break

        except Exception as e:
            consecutive_failures += 1
            logger.error(
                f"[{symbol}] Unexpected error in main loop: {e} "
                f"(consecutive failures: {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})"
            )
            notify_error(str(e))

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                if internal_position is not None:
                    logger.error(
                        f"[{symbol}] Too many consecutive errors with open position. "
                        "Executing emergency exit."
                    )
                    notify_error(
                        f"[{symbol}] Emergency exit triggered after "
                        f"{MAX_CONSECUTIVE_FAILURES} consecutive errors."
                    )
                    try:
                        _execute_exit(
                            "EMERGENCY_ERROR",
                            internal_position,
                            candle_dict,
                            symbol,
                            strategy,
                        )
                    except Exception as exit_err:
                        logger.error(
                            f"[{symbol}] Emergency exit also failed: {exit_err}"
                        )
                else:
                    logger.error(
                        f"[{symbol}] Too many consecutive errors. "
                        "No open position. Stopping bot."
                    )
                notify_error(
                    f"[{symbol}] Bot stopping after "
                    f"{MAX_CONSECUTIVE_FAILURES} consecutive errors."
                )
                break

            time.sleep(poll_interval)
            continue

        _sleep(loop_start, poll_interval)


# =============================================================================
# Entry / Exit Execution
# =============================================================================


def _execute_entry(
    signal: str, candle_dict: dict, strategy, symbol: str
) -> dict | None:
    side = signal
    response = place_market_order(side, config.ORDER_SIZE, symbol)

    if response and response.get("result"):
        entry_price = float(
            response["result"].get("average_fill_price", candle_dict["close"])
        )

        position = {
            "side": side,
            "entry_price": entry_price,
            "best_price": entry_price,
            "entry_time": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "size": config.ORDER_SIZE,
        }

        if hasattr(strategy, "get_last_entry_levels"):
            ob_levels = strategy.get_last_entry_levels()
            if ob_levels:
                position["sl_price"] = ob_levels.get("_ob_sl_price")
                position["tp_price"] = ob_levels.get("_ob_tp_price")
                position["atr_at_entry"] = ob_levels.get("_ob_atr_at_entry")

                sl_val = position.get("sl_price")
                tp_val = position.get("tp_price")
                atr_val = position.get("atr_at_entry")

                if sl_val is not None and tp_val is not None and atr_val is not None:
                    logger.info(
                        f"[{symbol}] OB/SMC entry levels injected: "
                        f"SL={sl_val:.2f} "
                        f"TP={tp_val:.2f} "
                        f"ATR={atr_val:.4f}"
                    )
                else:
                    logger.warning(
                        f"[{symbol}] OB/SMC entry levels incomplete "
                        f"(sl={sl_val}, tp={tp_val}, atr={atr_val}). "
                        f"Raw levels: {ob_levels}"
                    )

        if hasattr(strategy, "notify_entry"):
            strategy.notify_entry(0)

        if position.get("sl_price") is not None:
            sl = round(position["sl_price"], 1)
            tp = round(position["tp_price"], 1)
        else:
            sl_pct = getattr(strategy, "stop_loss_pct", STOP_LOSS_PCT)
            tp_pct = getattr(strategy, "take_profit_pct", TAKE_PROFIT_PCT)
            if side == "buy":
                sl = round(entry_price * (1 - sl_pct / 100), 1)
                tp = round(entry_price * (1 + tp_pct / 100), 1)
            else:
                sl = round(entry_price * (1 + sl_pct / 100), 1)
                tp = round(entry_price * (1 - tp_pct / 100), 1)

        notify_trade_entry(side, config.ORDER_SIZE, entry_price, sl, tp)
        logger.info(
            f"[{symbol}] Entry executed: {side.upper()} @ {entry_price} | SL={sl} TP={tp}"
        )
        return position

    else:
        logger.error(f"[{symbol}] Failed to place entry order.")
        notify_error(f"[{symbol}] Failed to place entry order.")
        return None


def _execute_exit(
    exit_reason: str,
    internal_position: dict,
    candle_dict: dict,
    symbol: str = SYMBOL,
    strategy=None,
) -> None:
    current_side = internal_position["side"]
    entry_price = internal_position["entry_price"]
    exit_side = "sell" if current_side == "buy" else "buy"
    size = internal_position.get("size", config.ORDER_SIZE)

    cancel_all_orders(symbol)
    time.sleep(0.3)

    response = place_market_order(exit_side, size, symbol)

    if response and response.get("result"):
        exit_price = float(
            response["result"].get(
                "average_fill_price", candle_dict.get("close", entry_price)
            )
        )

        net_pnl = record_trade(
            current_side,
            size,
            entry_price,
            exit_price,
            exit_reason=exit_reason,
        )

        notify_trade_exit(current_side, size, entry_price, exit_price, net_pnl)
        logger.info(
            f"[{symbol}] Exit executed: {exit_reason} | {current_side.upper()} "
            f"entry={entry_price} exit={exit_price} | PnL={net_pnl:.6f} BTC"
        )

    else:
        logger.error(f"[{symbol}] Failed to place exit order: {exit_reason}")
        notify_error(f"[{symbol}] Failed to exit position: {exit_reason}")

    if strategy is not None and hasattr(strategy, "notify_exit"):
        strategy.notify_exit()

    return None


# =============================================================================
# Helpers
# =============================================================================


def _get_candle_and_indicators(df, i: int) -> tuple[dict, dict]:
    """
    Extract candle OHLCV and all computed indicator columns.

    IMPORTANT: indicators_dict contains ALL columns (including OHLCV) so
    strategies that read 'close', 'high', 'low' from indicators_dict work
    correctly. This matches the backtester behaviour exactly.

    current_idx is injected so hold-period guards work in the live bot.

    prev_ema_fast / prev_ema_slow are injected for EMA crossover detection.
    """
    row = df.iloc[i]
    candle_dict = row.to_dict()

    indicators_dict = row.to_dict()
    indicators_dict["current_idx"] = i

    if i > 0:
        prev_row = df.iloc[i - 1]
        for col in ["ema_fast", "ema_slow"]:
            if col in df.columns:
                indicators_dict[f"prev_{col}"] = prev_row[col]
    else:
        for col in ["ema_fast", "ema_slow"]:
            indicators_dict[f"prev_{col}"] = None

    return candle_dict, indicators_dict


def _restore_position_on_startup(symbol: str) -> dict | None:
    """
    Check the exchange for an open position on startup.
    Passes symbol to get_open_position() so the correct product ID is resolved
    dynamically for whichever environment (testnet/production) is active.
    Returns a position dict if open position found, None otherwise.
    """
    try:
        # FIX: Pass symbol so get_open_position resolves the correct product_id
        # via get_product_id() rather than using the static config.PRODUCT_ID fallback.
        position_data = get_open_position(symbol)
        if position_data is None:
            return None

        size = float(position_data.get("size", 0))
        if size == 0:
            return None

        entry_price = float(position_data.get("entry_price", 0))
        if entry_price == 0:
            return None

        side = "buy" if size > 0 else "sell"

        return {
            "side": side,
            "entry_price": entry_price,
            "best_price": entry_price,
            "entry_time": position_data.get(
                "created_at", datetime.utcnow().isoformat()
            ),
            "symbol": symbol,
            "size": abs(size),
        }

    except Exception as e:
        logger.warning(f"[{symbol}] Could not restore position on startup: {e}")
        return None


def _sleep(loop_start: float, interval: int):
    elapsed = time.time() - loop_start
    sleep_time = max(0, interval - elapsed)
    logger.debug(f"Sleeping for {sleep_time:.1f}s")
    try:
        time.sleep(sleep_time)
    except KeyboardInterrupt:
        raise


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    args = parse_args()

    # -------------------------------------------------------------------------
    # FIX: Apply environment switching BEFORE run_bot() is called and BEFORE
    # any submodule makes an API call. All submodules use config.getbaseUrl(),
    # config.getAPI_KEY(), config.getAPI_SECRET() which read USE_TESTNET at
    # call time — so setting it here ensures every subsequent call uses the
    # correct environment.
    # -------------------------------------------------------------------------
    #
    #

    config.USE_TESTNET = args.mode == "paper"
    import market_data

    market_data.PRODUCT_ID_MAP.clear()

    # Override credentials from CLI if provided
    if args.api_key:
        if config.USE_TESTNET:
            config.API_KEY_TESTNET = args.api_key
        else:
            config.API_KEY = args.api_key

    if args.api_secret:
        if config.USE_TESTNET:
            config.API_SECRET_TESTNET = args.api_secret
        else:
            config.API_SECRET = args.api_secret

    strategy_params = build_strategy_params(args)
    strategy = get_strategy(args.strategy, **strategy_params)
    poll_interval = args.poll_interval if args.poll_interval else config.POLL_INTERVAL
    timeframe = args.timeframe
    symbol = args.symbol.upper() if args.symbol else config.SYMBOL
    lot = args.lot
    mode = config.USE_TESTNET

    run_bot(strategy, poll_interval, timeframe, symbol, lot, mode)
