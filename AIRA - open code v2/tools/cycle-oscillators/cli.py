#!/usr/bin/env python3
"""
Freqtrade Replay Harness — CLI entry point.

Example (inside container):
    python /freqtrade/user_data/freqtrade_replay/cli.py \
        --timerange 20241101-20241115

    # override config, pairs, strategy as needed:
    python /freqtrade/user_data/freqtrade_replay/cli.py \
        --config /freqtrade/user_data/config.json \
        --pairs "BTC/USDT:USDT" "ETH/USDT:USDT" \
        --timerange 20241101-20241115

Docker run (from project root):
    docker compose --profile replay run --rm replay --timerange 20241101-20241115
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

# Ensure freqtrade and our package are importable inside the container
sys.path.insert(0, "/freqtrade")
sys.path.insert(0, "/freqtrade/user_data")

from freqtrade_replay.runner import run_replay

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Suppress noisy subsystems
for noisy in (
    "freqtrade.exchange",
    "freqtrade.rpc",
    "freqtrade.plugins",
    "freqtrade.resolvers",
    "urllib3",
    "asyncio",
    "ccxt",
):
    logging.getLogger(noisy).setLevel(logging.WARNING)

DEFAULT_CONFIG = "/freqtrade/user_data/config.json"


def _parse_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d").replace(tzinfo=timezone.utc)


def _pairs_from_config(config_path: str) -> list[str]:
    with open(config_path) as f:
        cfg = json.load(f)
    pairs = cfg.get("exchange", {}).get("pair_whitelist", [])
    if not pairs:
        raise SystemExit(f"No pairs specified and pair_whitelist in {config_path} is empty.")
    return pairs


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run Freqtrade dry-run loop against historical data at max speed."
    )
    p.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help=f"Path to freqtrade config.json (default: {DEFAULT_CONFIG})",
    )
    p.add_argument(
        "--pairs", nargs="+", default=None,
        help='Trading pairs, e.g. "BTC/USDT:USDT" "ETH/USDT:USDT". '
             "Defaults to pair_whitelist in config.",
    )
    p.add_argument(
        "--timerange", required=True,
        help="Date range as YYYYMMDD-YYYYMMDD, e.g. 20241101-20241115",
    )
    p.add_argument(
        "--strategy", default="GKD_FisherTransformV4",
        help="Strategy class name (default: GKD_FisherTransformV4)",
    )
    p.add_argument(
        "--slippage", type=float, default=0.0005,
        help="Bid-ask half-spread fraction (default 0.0005 = 0.05%%). "
             "Controls order fill price and entry/exit pricing.",
    )
    p.add_argument(
        "--db", default="sqlite:////freqtrade/user_data/tradesv3.sqlite",
        help="SQLite DB URL for replay results. Compatible with freqtrade analysis tools.",
    )
    # --datadir will be auto-detected AFTER parsing args from the actual config file
    p.add_argument(
        "--datadir", default="/freqtrade/user_data/data/binance/futures",
        help="Directory containing .feather data files. Auto-detected from config exchange.",
    )
    p.add_argument(
        "--no-fresh", action="store_true",
        help="Keep existing DB instead of deleting it before the run.",
    )
    p.add_argument(
        "--report",
        default=None,
        metavar="PATH",
        help="Write a standalone HTML report (equity curve + per-pair P&L) to PATH. "
             "Example: /freqtrade/user_data/replay_report.html",
    )

    args = p.parse_args()

    # Detect exchange from ACTUAL config (not DEFAULT_CONFIG) for correct datadir
    try:
        with open(args.config) as f:
            cfg = json.load(f)
        exchange_name = cfg.get("exchange", {}).get("name", "binance").lower()
        trading_mode = cfg.get("trading_mode", "futures")
        # When data is mounted directly as /freqtrade/user_data/data (shared folder), use it flat
        default_datadir = f"/freqtrade/user_data/data/{exchange_name}/{trading_mode}"
        if not os.path.exists(default_datadir):
            # Fallback: shared data mounted flat at /freqtrade/user_data/data
            flat_datadir = "/freqtrade/user_data/data"
            if os.path.exists(flat_datadir):
                default_datadir = flat_datadir
    except Exception:
        default_datadir = "/freqtrade/user_data/data"

    # Override datadir if user didn't specify
    if args.datadir == "/freqtrade/user_data/data/binance/futures":
        args.datadir = default_datadir

    try:
        start_str, end_str = args.timerange.split("-")
        start_dt = _parse_dt(start_str)
        end_dt = _parse_dt(end_str)
    except ValueError:
        p.error("--timerange must be YYYYMMDD-YYYYMMDD")

    if start_dt >= end_dt:
        p.error("Start date must be before end date")

    pairs = args.pairs or _pairs_from_config(args.config)

    run_replay(
        config_path=args.config,
        pairs=pairs,
        start_dt=start_dt,
        end_dt=end_dt,
        strategy=args.strategy,
        slippage_pct=args.slippage,
        db_url=args.db,
        datadir=args.datadir,
        fresh=not args.no_fresh,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
