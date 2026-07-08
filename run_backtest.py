# =============================================================================
# run_backtest.py - CLI Backtest Runner (single or all strategies)
# =============================================================================

import argparse
import os
import sys

from strategies.smc_strategy import SMCStrategy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtester import fetch_historical_candles, run_all_backtests, run_backtest
from config import BACKTEST_CHART_FILE, BACKTEST_DAYS
from logger import get_logger
from strategies import STRATEGY_REGISTRY, get_strategy

logger = get_logger("run_backtest")


def print_header(
    strategy_name: str, symbol: str, days: int, timeframe: str, params: dict = None
):
    print("\n" + "=" * 55)
    print(f"    {strategy_name} - BACKTEST RUNNER")
    print("=" * 55)
    print(f"  Strategy  : {strategy_name}")
    print(f"  Symbol    : {symbol}")
    print(f"  Timeframe : {timeframe}")
    print(f"  Days      : {days}")
    if params:
        for key, value in params.items():
            print(f"  {key:<10} : {value}")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(
        description="Delta Exchange - Customizable Strategy Backtester",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  py run_backtest.py --all
  py run_backtest.py --strategy ema_crossover
  py run_backtest.py --strategy ema_crossover --fast 5 --slow 15 --days 60
  py run_backtest.py --strategy ema_crossover --symbol ETHUSD --timeframe 1h
  py run_backtest.py --strategy bollinger_bands --period 20 --std 2.2
  py run_backtest.py --strategy bollinger_bands --period 20 --std 2.2 --bb-stop-loss 1.0 --bb-trail 0.8 --bb-take-profit 1.5
  py run_backtest.py --strategy bollinger_bands --symbol XAUTUSD --timeframe 15m
  py run_backtest.py --strategy vwap --period 50
  py run_backtest.py --strategy triple_ema --fast 5 --mid 13 --slow 21
  py run_backtest.py --strategy fvg
  py run_backtest.py --strategy fvg --fvg-min-size 0.05 --fvg-max-age 50
  py run_backtest.py --strategy fvg --fvg-stop-loss 1.0 --fvg-take-profit 1.5
  py run_backtest.py --strategy fvg --symbol ETHUSD --timeframe 1h --days 60
  py run_backtest.py --strategy fvg --symbol BTCUSD --timeframe 4h --fvg-trail 0.5
  py run_backtest.py --strategy ob
  py run_backtest.py --strategy ob --ob-atr-length 14 --ob-max-age 50
  py run_backtest.py --strategy ob --ob-strength-mult 1.5 --ob-rr-ratio 3.0
  py run_backtest.py --strategy ob --symbol ETHUSD --timeframe 1h --days 60
  py run_backtest.py --strategy ob --symbol BTCUSD --timeframe 4h --ob-atr-mult 1.5
  py run_backtest.py --strategy smc --symbol ETHUSD --timeframe 15m --days 90
  py run_backtest.py --strategy smc --symbol ETHUSD --timeframe 15m --smc-proximity-atr 2.0
  py run_backtest.py --all --symbol ETHUSD --timeframe 1h

  Parameter	CLI Flag
  atr_multiplier	--ob-atr-mult
  trail_activation_atr	--ob-trail-activation
  ob_strength_mult	--ob-strength-mult
  ob_strength_period	--ob-strength-period
  rr_ratio	--ob-rr-ratio
  sl_atr_buffer	--ob-sl-buffer
  max_ob_age_candles	--ob-max-age
  trend_ema_period	--ob-trend-ema
  min_atr_threshold	--ob-min-atr
  atr_length	--ob-atr-length
  stop_loss_pct (BB)	--bb-stop-loss
  trail_activation_pct (BB)	--bb-trail-activation
  trail_pct (BB)	--bb-trail
  take_profit_pct (BB)	--bb-take-profit
  tp_extension_pct (BB)	--bb-tp-extension
  volume_period (BB)	--bb-volume-period
  volume_multiplier (BB)	--bb-volume-multiplier

Available strategies:
  ema_crossover    - EMA fast/slow crossover
  bollinger_bands  - Bollinger Bands mean reversion
  vwap             - Rolling VWAP crossover
  triple_ema       - Triple EMA alignment
  fvg              - Fair Value Gap (SMC) retracement
  ob               - Order Block (SMC) retracement with ATR TSL
  smc              - Smart Money Concepts (CHoCH + BOS + OB)

Available timeframes:
  5s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 1w

FVG timeframe guidance:
  1m  : --fvg-min-size 0.03 --fvg-max-age 20  --fvg-stop-loss 0.5  --fvg-take-profit 0.8
  5m  : --fvg-min-size 0.05 --fvg-max-age 30  --fvg-stop-loss 0.8  --fvg-take-profit 1.2
  15m : --fvg-min-size 0.05 --fvg-max-age 50  --fvg-stop-loss 1.0  --fvg-take-profit 1.5
  1h  : --fvg-min-size 0.10 --fvg-max-age 50  --fvg-stop-loss 1.0  --fvg-take-profit 1.5
  4h  : --fvg-min-size 0.15 --fvg-max-age 30  --fvg-stop-loss 1.5  --fvg-take-profit 2.5
  1d  : --fvg-min-size 0.20 --fvg-max-age 20  --fvg-stop-loss 2.0  --fvg-take-profit 4.0

OB timeframe guidance:
  1m  : --ob-strength-mult 1.2 --ob-max-age 20 --ob-rr-ratio 2.0
  5m  : --ob-strength-mult 1.3 --ob-max-age 30 --ob-rr-ratio 2.5
  15m : --ob-strength-mult 1.5 --ob-max-age 50 --ob-rr-ratio 3.0
  1h  : --ob-strength-mult 1.5 --ob-max-age 50 --ob-rr-ratio 3.0
  4h  : --ob-strength-mult 1.8 --ob-max-age 30 --ob-rr-ratio 3.5
  1d  : --ob-strength-mult 2.0 --ob-max-age 20 --ob-rr-ratio 4.0
        """,
    )

    # =========================================================
    # OB (Order Block) argument group
    # =========================================================
    parser.add_argument(
        "--ob-atr-length",
        type=int,
        default=14,
        help="OB ATR period for SL/TP/TSL calculation (default: 14)",
    )
    parser.add_argument(
        "--ob-atr-mult",
        type=float,
        default=1.5,
        help="OB ATR multiplier for trailing stop distance (default: 1.5)",
    )
    parser.add_argument(
        "--ob-trail-activation",
        type=float,
        default=1.5,
        help="OB trail activation in ATR units from entry (default: 1.5)",
    )
    parser.add_argument(
        "--ob-strength-period",
        type=int,
        default=20,
        help="OB avg body lookback period for strong move filter (default: 20)",
    )
    parser.add_argument(
        "--ob-strength-mult",
        type=float,
        default=1.5,
        help="OB body multiplier threshold for strong move detection (default: 1.5)",
    )

    # =========================================================
    # SMC Strategy argument group
    # =========================================================
    parser.add_argument(
        "--smc-swing-lookback",
        type=int,
        default=5,
        help="Candles each side for swing detection",
    )
    parser.add_argument(
        "--smc-atr-length",
        type=int,
        default=14,
        help="ATR period",
    )
    parser.add_argument(
        "--smc-atr-mult",
        type=float,
        default=1.5,
        help="ATR trailing multiplier",
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
        "--smc-strength-period",
        type=int,
        default=20,
        help="OB strength ATR period",
    )
    parser.add_argument(
        "--smc-strength-mult",
        type=float,
        default=0.3,
        help="OB strength multiplier",
    )
    parser.add_argument(
        "--smc-max-age",
        type=int,
        default=30,
        help="Max OB age in candles",
    )
    parser.add_argument(
        "--smc-rr-ratio",
        type=float,
        default=2.0,
        help="Risk:Reward ratio for TP",
    )
    parser.add_argument(
        "--smc-ema-period",
        type=int,
        default=50,
        help="Trend EMA period",
    )
    parser.add_argument(
        "--smc-min-atr",
        type=float,
        default=0.0,
        help="Minimum ATR threshold",
    )
    parser.add_argument(
        "--smc-bos-type",
        type=str,
        default="close",
        choices=["close", "wick"],
        help="BOS type",
    )
    parser.add_argument(
        "--smc-choch-type",
        type=str,
        default="close",
        choices=["close", "wick"],
        help="CHoCH type",
    )
    parser.add_argument(
        "--smc-structure-lookback",
        type=int,
        default=50,
        help="BOS structure lookback candles",
    )
    parser.add_argument(
        "--smc-proximity-atr",
        type=float,
        default=3.0,
        help="Max ATR distance from current price to OB for entry filter",
    )

    parser.add_argument(
        "--ob-rr-ratio",
        type=float,
        default=3.0,
        help="OB risk:reward ratio for take profit (default: 3.0)",
    )
    parser.add_argument(
        "--ob-sl-buffer",
        type=float,
        default=0.1,
        help="OB SL ATR buffer beyond OB wick (default: 0.1)",
    )
    parser.add_argument(
        "--ob-max-age",
        type=int,
        default=50,
        help="OB max candles before zone expires (default: 50)",
    )
    parser.add_argument(
        "--ob-trend-ema",
        type=int,
        default=50,
        help="OB trend EMA period for directional filter (default: 50, set 0 to disable)",
    )
    parser.add_argument(
        "--ob-min-atr",
        type=float,
        default=0.0,
        help="OB minimum ATR value to allow entry, filters low-volatility candles (default: 0.0 = disabled)",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=list(STRATEGY_REGISTRY.keys()),
        help="Strategy to backtest",
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all strategies and compare"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSD",
        help="Trading symbol to backtest (default: BTCUSD)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="15m",
        choices=[
            "5s",
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
        ],
        help="Candle timeframe (default: 15m)",
    )
    parser.add_argument(
        "--bb-tp-extension",
        type=float,
        default=0.3,
        help="BB TP extension beyond midline in percent (default: 0.3)",
    )
    parser.add_argument(
        "--bb-take-profit",
        type=float,
        default=2.0,
        help="BB fixed take profit percentage from entry (default: 2.0)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=BACKTEST_DAYS,
        help=f"Days of historical data (default: {BACKTEST_DAYS})",
    )
    parser.add_argument(
        "--bb-volume-period",
        type=int,
        default=20,
        help="BB volume average period (default: 20)",
    )
    parser.add_argument(
        "--bb-volume-multiplier",
        type=float,
        default=1.2,
        help="BB volume multiplier threshold (default: 1.2)",
    )
    parser.add_argument(
        "--fast", type=int, default=9, help="EMA fast period (default: 9)"
    )
    parser.add_argument(
        "--slow", type=int, default=21, help="EMA slow period (default: 21)"
    )
    parser.add_argument(
        "--period", type=int, default=20, help="BB/VWAP period (default: 20)"
    )
    parser.add_argument(
        "--std", type=float, default=2.2, help="BB std deviation (default: 2.2)"
    )
    parser.add_argument(
        "--bb-stop-loss",
        type=float,
        default=1.0,
        help="BB hard stop loss percentage (default: 1.0)",
    )
    parser.add_argument(
        "--bb-trail-activation",
        type=float,
        default=1.0,
        help="BB trail activation percentage (default: 1.0)",
    )
    parser.add_argument(
        "--bb-trail",
        type=float,
        default=0.8,
        help="BB trailing stop percentage (default: 0.8)",
    )
    parser.add_argument(
        "--mid", type=int, default=13, help="Triple EMA mid period (default: 13)"
    )

    # =========================================================
    # FVG argument group
    # =========================================================
    parser.add_argument(
        "--fvg-min-size",
        type=float,
        default=0.05,
        help="FVG minimum gap size as %% of price to filter noise (default: 0.05)",
    )
    parser.add_argument(
        "--fvg-max-age",
        type=int,
        default=50,
        help="FVG max candles to wait for retracement before expiry (default: 50)",
    )
    parser.add_argument(
        "--fvg-trail-activation",
        type=float,
        default=0.8,
        help="FVG trail activation percentage from entry (default: 0.8)",
    )
    parser.add_argument(
        "--fvg-trail",
        type=float,
        default=0.5,
        help="FVG trailing stop percentage from best price (default: 0.5)",
    )
    parser.add_argument(
        "--fvg-stop-loss",
        type=float,
        default=1.0,
        help="FVG hard stop loss percentage from entry (default: 1.0)",
    )
    parser.add_argument(
        "--fvg-take-profit",
        type=float,
        default=1.5,
        help="FVG fixed take profit percentage from entry (default: 1.5)",
    )
    parser.add_argument(
        "--fvg-tp-extension",
        type=float,
        default=0.3,
        help="FVG TP extension once trailing stop activates (default: 0.3)",
    )

    args = parser.parse_args()

    if not args.all and args.strategy is None:
        parser.print_help()
        print("\n[ERROR] Please provide --strategy <name> or --all\n")
        sys.exit(1)

    # --- Run All Strategies ---
    if args.all:
        print_header("ALL STRATEGIES", args.symbol, args.days, args.timeframe)
        print("\n[INFO] Fetching shared historical data...")

        df = fetch_historical_candles(
            days=args.days, timeframe=args.timeframe, symbol=args.symbol
        )
        if df.empty:
            print("[ERROR] Failed to fetch historical data.")
            sys.exit(1)

        print(
            f"[INFO] {len(df)} candles fetched: "
            f"{df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}\n"
        )

        strategies = [
            get_strategy(
                "ob",
                atr_length=args.ob_atr_length,
                atr_multiplier=args.ob_atr_mult,
                trail_activation_atr=args.ob_trail_activation,
                ob_strength_period=args.ob_strength_period,
                ob_strength_mult=args.ob_strength_mult,
                rr_ratio=args.ob_rr_ratio,
                sl_atr_buffer=args.ob_sl_buffer,
                max_ob_age_candles=args.ob_max_age,
                trend_ema_period=args.ob_trend_ema,
                min_atr_threshold=args.ob_min_atr,
                is_backtest=True,
            ),
            get_strategy("ema_crossover", fast=args.fast, slow=args.slow),
            get_strategy(
                "bollinger_bands",
                period=args.period,
                std=args.std,
                stop_loss_pct=args.bb_stop_loss,
                trail_activation_pct=args.bb_trail_activation,
                trail_pct=args.bb_trail,
                tp_extension_pct=args.bb_tp_extension,
                take_profit_pct=args.bb_take_profit,
                volume_period=args.bb_volume_period,
                volume_multiplier=args.bb_volume_multiplier,
            ),
            get_strategy("vwap", period=args.period),
            get_strategy("triple_ema", fast=args.fast, mid=args.mid, slow=args.slow),
            get_strategy(
                "fvg",
                min_fvg_size_pct=args.fvg_min_size,
                max_fvg_age_candles=args.fvg_max_age,
                trail_activation_pct=args.fvg_trail_activation,
                trail_pct=args.fvg_trail,
                stop_loss_pct=args.fvg_stop_loss,
                take_profit_pct=args.fvg_take_profit,
                tp_extension_pct=args.fvg_tp_extension,
            ),
            # --- SMC added to --all ---
            get_strategy(
                "smc",
                swing_lookback=args.smc_swing_lookback,
                atr_length=args.smc_atr_length,
                atr_multiplier=args.smc_atr_mult,
                trail_activation_atr=args.smc_trail_activation,
                sl_atr_buffer=args.smc_sl_buffer,
                ob_strength_period=args.smc_strength_period,
                ob_strength_mult=args.smc_strength_mult,
                max_ob_age_candles=args.smc_max_age,
                rr_ratio=args.smc_rr_ratio,
                trend_ema_period=args.smc_ema_period,
                min_atr_threshold=args.smc_min_atr,
                bos_type=args.smc_bos_type,
                choch_type=args.smc_choch_type,
                structure_lookback=args.smc_structure_lookback,
                ob_proximity_atr=args.smc_proximity_atr,
            ),
        ]

        run_all_backtests(
            strategies,
            df=df,
            days=args.days,
            timeframe=args.timeframe,
            symbol=args.symbol,
        )
        print(f"\n[DONE] Comparison chart saved to: {BACKTEST_CHART_FILE}\n")

    # --- Run Single Strategy ---
    else:
        strategy_params = {}

        if args.strategy == "ema_crossover":
            strategy_params = {
                "fast": args.fast,
                "slow": args.slow,
            }

        # -----------------------------------------------------------------
        # FIX: SMC now uses strategy_params dict like all other strategies
        # so get_strategy() at the bottom receives the correct values.
        # Previously SMCStrategy() was instantiated directly here but then
        # overwritten by get_strategy(args.strategy, **{}) below.
        # -----------------------------------------------------------------
        elif args.strategy == "smc":
            strategy_params = {
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
                "ob_proximity_atr": args.smc_proximity_atr,
            }

        elif args.strategy == "bollinger_bands":
            strategy_params = {
                "period": args.period,
                "std": args.std,
                "stop_loss_pct": args.bb_stop_loss,
                "trail_activation_pct": args.bb_trail_activation,
                "trail_pct": args.bb_trail,
                "tp_extension_pct": args.bb_tp_extension,
                "take_profit_pct": args.bb_take_profit,
                "volume_period": args.bb_volume_period,
                "volume_multiplier": args.bb_volume_multiplier,
            }

        elif args.strategy == "vwap":
            strategy_params = {
                "period": args.period,
            }

        elif args.strategy == "triple_ema":
            strategy_params = {
                "fast": args.fast,
                "mid": args.mid,
                "slow": args.slow,
            }

        elif args.strategy == "ob":
            strategy_params = {
                "atr_length": args.ob_atr_length,
                "atr_multiplier": args.ob_atr_mult,
                "trail_activation_atr": args.ob_trail_activation,
                "ob_strength_period": args.ob_strength_period,
                "ob_strength_mult": args.ob_strength_mult,
                "rr_ratio": args.ob_rr_ratio,
                "sl_atr_buffer": args.ob_sl_buffer,
                "max_ob_age_candles": args.ob_max_age,
                "trend_ema_period": args.ob_trend_ema,
                "min_atr_threshold": args.ob_min_atr,
                "is_backtest": True,
            }

        elif args.strategy == "fvg":
            strategy_params = {
                "min_fvg_size_pct": args.fvg_min_size,
                "max_fvg_age_candles": args.fvg_max_age,
                "trail_activation_pct": args.fvg_trail_activation,
                "trail_pct": args.fvg_trail,
                "stop_loss_pct": args.fvg_stop_loss,
                "take_profit_pct": args.fvg_take_profit,
                "tp_extension_pct": args.fvg_tp_extension,
            }

        strategy = get_strategy(args.strategy, **strategy_params)
        print_header(
            strategy.name(),
            args.symbol,
            args.days,
            args.timeframe,
            params=strategy_params,
        )

        print("\n[INFO] Fetching historical data...")
        df = fetch_historical_candles(
            days=args.days, timeframe=args.timeframe, symbol=args.symbol
        )

        if df.empty:
            print("[ERROR] Failed to fetch historical data.")
            sys.exit(1)

        print(
            f"[INFO] {len(df)} candles fetched: "
            f"{df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}\n"
        )

        trades, summary = run_backtest(
            strategy,
            df=df,
            days=args.days,
            timeframe=args.timeframe,
            symbol=args.symbol,
        )

        if not trades:
            print(
                "[WARNING] No trades generated. "
                "Try adjusting parameters or increasing --days."
            )
            sys.exit(0)

        print(f"\n[DONE] Chart saved to: {BACKTEST_CHART_FILE}\n")


if __name__ == "__main__":
    main()
