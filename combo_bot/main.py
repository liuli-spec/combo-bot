from __future__ import annotations
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("combo_bot")


def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text())


def cmd_backtest(args):
    from combo_bot.backtest import BacktestConfig, Backtester
    from combo_bot.grid_engine import GridConfig
    from combo_bot.trend_signal import TrendConfig
    from combo_bot.merger import MergerConfig
    from combo_bot.risk import RiskConfig
    from combo_bot.data import load_cached_data

    cfg = load_config(args.config)

    # Map a "timeframe" string ("1m", "5m", "1h", ...) to bar_interval_minutes
    # so users can put it in their config alongside the candle download
    # timeframe and have funding / volatility-EMA / Sharpe annualization
    # scale correctly. Explicit ``bar_interval_minutes`` in the config
    # takes priority if set.
    _TF_TO_MIN = {
        "1m": 1.0, "3m": 3.0, "5m": 5.0, "15m": 15.0, "30m": 30.0,
        "1h": 60.0, "2h": 120.0, "4h": 240.0, "6h": 360.0, "8h": 480.0,
        "12h": 720.0, "1d": 1440.0,
    }
    bar_min = cfg.get("bar_interval_minutes")
    if bar_min is None:
        bar_min = _TF_TO_MIN.get(str(cfg.get("timeframe", "1m")), 1.0)

    bt_config = BacktestConfig(
        starting_balance=cfg.get("starting_balance", 10000),
        maker_fee=cfg.get("maker_fee", 0.0002),
        taker_fee=cfg.get("taker_fee", 0.0005),
        bar_interval_minutes=float(bar_min),
        symbols=cfg.get("symbols", []),
        grid=GridConfig(**cfg.get("grid", {})),
        trend=TrendConfig(**cfg.get("trend", {})),
        merger=MergerConfig(**cfg.get("merger", {})),
        risk=RiskConfig(**cfg.get("risk", {})),
    )

    candle_data = load_cached_data(bt_config.symbols, data_dir=cfg.get("data_dir", "data"))
    if not candle_data:
        logger.error("No cached data found. Run 'download' first.")
        sys.exit(1)

    logger.info("Running backtest on %d symbols, %d candles each",
                len(candle_data), min(len(v) for v in candle_data.values()))

    bt = Backtester(bt_config)
    result = bt.run(candle_data)

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Duration:        {result.duration_days:.1f} days")
    print(f"Final Balance:   {result.final_balance:.2f}")
    print(f"Total PnL:       {result.total_pnl:.2f}")
    print(f"Total Fees:      {result.total_fees:.2f}")
    print(f"Total Funding:   {result.total_funding:.2f}")
    print(f"Trades:          {result.n_trades}")
    print(f"Win Rate:        {result.win_rate:.2%}")
    print(f"ADG:             {result.adg:.4%}")
    print(f"Max Drawdown:    {result.max_drawdown:.2%}")
    print(f"Sharpe Ratio:    {result.sharpe_ratio:.3f}")
    print(f"Sortino Ratio:   {result.sortino_ratio:.3f}")
    print(f"Calmar Ratio:    {result.calmar_ratio:.3f}")
    print(f"Grid PnL:        {result.grid_pnl:.2f}")
    print(f"Trend PnL:       {result.trend_pnl:.2f}")
    print("=" * 60)

    if args.output:
        import numpy as np
        out = {
            "final_balance": result.final_balance,
            "total_pnl": result.total_pnl,
            "total_fees": result.total_fees,
            "n_trades": result.n_trades,
            "win_rate": result.win_rate,
            "adg": result.adg,
            "max_drawdown": result.max_drawdown,
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "calmar_ratio": result.calmar_ratio,
            "grid_pnl": result.grid_pnl,
            "trend_pnl": result.trend_pnl,
            "duration_days": result.duration_days,
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        logger.info("Results saved to %s", args.output)


def cmd_download(args):
    from combo_bot.data import download_backtest_data

    cfg = load_config(args.config)
    symbols = cfg.get("symbols", [])
    data_dir = cfg.get("data_dir", "data")

    if not symbols:
        logger.error("No symbols in config")
        sys.exit(1)

    logger.info("Downloading data for %s", symbols)
    asyncio.run(download_backtest_data(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        timeframe=args.timeframe,
        data_dir=data_dir,
    ))
    logger.info("Download complete")


def cmd_optimize(args):
    from combo_bot.data import load_cached_data
    from combo_bot.optimize import OptimizeConfig, Optimizer

    cfg = load_config(args.config)
    symbols = cfg.get("symbols", [])
    candle_data = load_cached_data(symbols, data_dir=cfg.get("data_dir", "data"))
    if not candle_data:
        logger.error("No cached data. Run 'download' first.")
        sys.exit(1)

    opt_cfg = OptimizeConfig(
        n_trials=args.trials,
        n_jobs=args.jobs,
    )

    optimizer = Optimizer(opt_cfg, candle_data)
    best = optimizer.run()

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(json.dumps(best, indent=2, default=str))

    if args.output:
        Path(args.output).write_text(json.dumps(best, indent=2, default=str))
        logger.info("Best params saved to %s", args.output)


def cmd_live(args):
    from combo_bot.data import create_exchange
    from combo_bot.live import LiveConfig, LiveTrader
    from combo_bot.grid_engine import GridConfig
    from combo_bot.trend_signal import TrendConfig
    from combo_bot.merger import MergerConfig
    from combo_bot.risk import RiskConfig

    cfg = load_config(args.config)

    live_cfg = LiveConfig(
        symbols=cfg.get("symbols", []),
        leverage=cfg.get("leverage", 5),
        dry_run=not args.real,
        grid=GridConfig(**cfg.get("grid", {})),
        trend=TrendConfig(**cfg.get("trend", {})),
        merger=MergerConfig(**cfg.get("merger", {})),
        risk=RiskConfig(**cfg.get("risk", {})),
    )

    if args.real:
        logger.warning("=" * 40)
        logger.warning("  REAL TRADING MODE - USE AT OWN RISK")
        logger.warning("=" * 40)
        confirm = input("Type 'YES' to confirm: ")
        if confirm != "YES":
            logger.info("Aborted")
            return

    exchange = create_exchange(testnet=args.testnet)

    trader = LiveTrader(live_cfg, exchange)

    async def run():
        try:
            await trader.start()
        except KeyboardInterrupt:
            await trader.stop()
        finally:
            await exchange.close()

    asyncio.run(run())


def main():
    parser = argparse.ArgumentParser(description="Combo Bot: Grid + Trend Futures Trading")
    sub = parser.add_subparsers(dest="command")

    p_bt = sub.add_parser("backtest", help="Run backtest")
    p_bt.add_argument("-c", "--config", default="config.json")
    p_bt.add_argument("-o", "--output", help="Save results to file")

    p_dl = sub.add_parser("download", help="Download historical data")
    p_dl.add_argument("-c", "--config", default="config.json")
    p_dl.add_argument("--start", default="2024-01-01")
    p_dl.add_argument("--end", default="2025-12-31")
    p_dl.add_argument("--timeframe", default="1m")

    p_opt = sub.add_parser("optimize", help="Optimize parameters")
    p_opt.add_argument("-c", "--config", default="config.json")
    p_opt.add_argument("--trials", type=int, default=500)
    p_opt.add_argument("--jobs", type=int, default=4)
    p_opt.add_argument("-o", "--output", help="Save best params")

    p_live = sub.add_parser("live", help="Live trading")
    p_live.add_argument("-c", "--config", default="config.json")
    p_live.add_argument("--real", action="store_true", help="Real trading (not dry run)")
    p_live.add_argument("--testnet", action="store_true", help="Use testnet")

    args = parser.parse_args()
    if args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "live":
        cmd_live(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
