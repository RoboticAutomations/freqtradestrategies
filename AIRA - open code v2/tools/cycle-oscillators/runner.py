"""
ReplayRunner — drives FreqtradeBot.process() candle-by-candle at maximum speed.

Patch surface beyond ReplayExchange:
  ExchangeResolver.load_exchange  → return ReplayExchange (constructor injection)
  Worker._sleep                   → advance virtual clock instead of wall-clock sleep
"""

import logging
import subprocess
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

from freqtrade.configuration import Configuration  # returns dict directly in 2026.x
from freqtrade.enums import RunMode
from freqtrade.exchange import timeframe_to_seconds
from freqtrade.enums import State
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.persistence import Trade, init_db
from freqtrade.resolvers import ExchangeResolver
from freqtrade.wallets import Wallets
from freqtrade.worker import Worker

from .clock import VirtualClock
from .data_store import ReplayDataStore
from .exchange import ReplayExchange

logger = logging.getLogger(__name__)


def _drop_db(db_url: str) -> None:
    """Delete the SQLite file (+ WAL pair) before a fresh run."""
    if not db_url.startswith("sqlite:///"):
        return
    import os
    path = db_url.replace("sqlite:///", "")
    for suffix in ("", "-shm", "-wal"):
        p = path + suffix
        if os.path.exists(p):
            os.remove(p)
            logger.info("Removed existing DB file: %s", p)


def _download_data(
    config_path: str,
    pairs: list[str],
    start_dt: datetime,
    end_dt: datetime,
    datadir: str,
    trading_mode: str = "futures",
) -> None:
    """Auto-download missing/stale OHLCV data via the freqtrade CLI."""
    # freqtrade download-data appends /<trading_mode> to --datadir automatically.
    # Our datadir already ends in /futures, so strip it to avoid double-nesting.
    dl_datadir = Path(datadir)
    if dl_datadir.name == trading_mode:
        dl_datadir = dl_datadir.parent

    # Go back 90 days before start to cover any startup_candle_count warmup
    dl_start = (start_dt - timedelta(days=90)).strftime("%Y%m%d")
    dl_end = (end_dt + timedelta(days=1)).strftime("%Y%m%d")
    cmd = [
        "freqtrade", "download-data",
        "--config", config_path,
        "--pairs", *pairs,
        "--timeframes", "1m", "5m", "15m", "1h", "4h",
        "--timerange", f"{dl_start}-{dl_end}",
        "--trading-mode", trading_mode,
        "--datadir", str(dl_datadir),
    ]
    logger.info("Auto-downloading data for %s (%s → %s) …", pairs, dl_start, dl_end)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.warning("download-data exited with code %d — proceeding anyway", result.returncode)



def _load_informative_pairs(
    bot,
    store: "ReplayDataStore",
    exchange: "ReplayExchange",
    config_path: str,
    start_dt: datetime,
    end_dt: datetime,
    datadir: str,
    trading_mode: str,
) -> None:
    """Detect and load informative pairs declared by the strategy."""
    try:
        inf_pairs = bot.strategy.informative_pairs()
    except Exception as exc:
        logger.warning("Could not read informative_pairs() from strategy: %s", exc)
        return

    extra: set[str] = set()
    for item in inf_pairs:
        inf_pair = item[0]  # (pair, tf) or (pair, tf, candle_type)
        if inf_pair not in store._candles:
            extra.add(inf_pair)

    if not extra:
        return

    logger.info("Detected informative pairs not in whitelist: %s", sorted(extra))

    missing: list[str] = []
    for pair in sorted(extra):
        found = store.load_extra_pair(pair)
        if not found:
            missing.append(pair)

    if missing:
        logger.info("Downloading missing informative pair data: %s", missing)
        _download_data(config_path, missing, start_dt, end_dt, datadir, trading_mode)
        for pair in missing:
            store.load_extra_pair(pair)

    # Register informative pairs in the exchange market map so Freqtrade
    # doesn't reject them as unknown symbols.
    for pair in extra:
        if pair not in exchange._markets:
            exchange._markets[pair] = exchange._make_market(pair)


def run_replay(
    config_path: str,
    pairs: list[str],
    start_dt: datetime,
    end_dt: datetime,
    strategy: str = "GKD_FisherTransformV4",
    slippage_pct: float = 0.0005,
    db_url: str = "sqlite:////freqtrade/user_data/tradesv3.sqlite",
    datadir: str | None = None,
    fresh: bool = True,
    report_path: str | None = None,
) -> None:
    if fresh:
        _drop_db(db_url)

    # ------------------------------------------------------------------ #
    # 1. Build config                                                       #
    # ------------------------------------------------------------------ #
    config = deepcopy(Configuration.from_files([config_path]))

    # Derive datadir from config exchange if not explicitly provided
    if datadir is None:
        exchange_name = config.get("exchange", {}).get("name", "binance").lower()
        trading_mode = config.get("trading_mode", "futures")
        datadir = f"/freqtrade/user_data/data/{exchange_name}/{trading_mode}"

    config["dry_run"] = True
    config["runmode"] = RunMode.DRY_RUN
    config["db_url"] = db_url
    config["pairs"] = pairs
    config["strategy"] = strategy

    # Strip live credentials
    config["exchange"]["key"] = ""
    config["exchange"]["secret"] = ""

    # Disable all notification channels (remove rather than disable to skip schema validation)
    config.pop("telegram", None)
    config.pop("api_server", None)

    tf: str = config.get("timeframe", "1h")
    tf_secs: int = timeframe_to_seconds(tf)

    # ------------------------------------------------------------------ #
    # 2. Load and validate historical data — auto-download if stale/missing #
    # ------------------------------------------------------------------ #
    startup_count: int = config.get("startup_candle_count", 50)
    data_start = start_dt - timedelta(seconds=startup_count * tf_secs)

    trading_mode = config.get("trading_mode", "futures")
    store = ReplayDataStore(datadir, pairs, trading_mode=trading_mode)

    missing: list[str] = []
    for pair in pairs:
        try:
            store.validate(pair, tf, start_dt, end_dt, startup_count)
        except ValueError as exc:
            logger.warning("Data gap detected — will auto-download: %s", exc)
            missing.append(pair)

    if missing:
        _download_data(config_path, missing, start_dt, end_dt, datadir, trading_mode)
        # Reload store with fresh files
        store = ReplayDataStore(datadir, pairs, trading_mode=trading_mode)
        # Re-validate — skip pairs that STILL don't have data after download
        still_missing = []
        for pair in pairs:
            try:
                store.validate(pair, tf, start_dt, end_dt, startup_count)
            except ValueError as exc:
                logger.warning("Still no data after download — skipping pair: %s", exc)
                still_missing.append(pair)
        
        # Remove still-missing pairs from the pairlist so replay can continue
        if still_missing:
            logger.info("Removing %d pairs with no data: %s", len(still_missing), still_missing)
            pairs = [p for p in pairs if p not in still_missing]
            # Rebuild store with only valid pairs
            store = ReplayDataStore(datadir, pairs, trading_mode=trading_mode)
            for pair in pairs:
                store.validate(pair, tf, start_dt, end_dt, startup_count)

    # ------------------------------------------------------------------ #
    # 3. Virtual clock                                                      #
    # ------------------------------------------------------------------ #
    clock = VirtualClock()
    clock.start(data_start)

    # ------------------------------------------------------------------ #
    # 4. Build ReplayExchange                                               #
    # ------------------------------------------------------------------ #
    exchange = ReplayExchange(config, store, clock, slippage_pct=slippage_pct)

    # ------------------------------------------------------------------ #
    # 5. Patch ExchangeResolver so FreqtradeBot receives our exchange       #
    # ------------------------------------------------------------------ #
    original_load_exchange = ExchangeResolver.load_exchange
    ExchangeResolver.load_exchange = staticmethod(lambda cfg, **kw: exchange)

    # ------------------------------------------------------------------ #
    # 6. Patch Worker._sleep to advance virtual clock instead of sleeping   #
    # ------------------------------------------------------------------ #
    original_sleep = Worker._sleep
    Worker._sleep = staticmethod(
        lambda duration: clock.advance_to(clock.now() + timedelta(seconds=max(duration, 0)))
    )

    # ------------------------------------------------------------------ #
    # 6b. Patch Wallets.record_wallet_state to use upsert semantics        #
    #     The method floors its timestamp to the day, so multiple calls     #
    #     within the same replay day all share the same (timestamp,         #
    #     currency) key and hit the UNIQUE constraint.                      #
    # ------------------------------------------------------------------ #
    original_record_wallet_state = Wallets.record_wallet_state

    def _upsert_record_wallet_state(self) -> None:
        from freqtrade.persistence import WalletHistory
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        if self._is_backtest:
            return
        from freqtrade.util.datetime_helpers import dt_floor_day, dt_now
        timestamp = dt_floor_day(dt_now())

        position_collaterals = 0.0
        open_assets: dict = {t.safe_base_currency: t for t in Trade.get_open_trades()}
        wallet_records = []

        for pos in self.get_all_positions().values():
            base = self._exchange.get_pair_base_currency(pos.symbol)
            rate = self._exchange.get_conversion_rate(base, self._stake_currency)
            leverage = pos.leverage or 1.0
            total_quote = None
            if rate:
                total_quote = (
                    rate * pos.position - pos.collateral * (leverage - 1)
                    if pos.side == "long"
                    else pos.collateral * (1 + leverage) - rate * pos.position
                )
            position_collaterals += pos.collateral
            wallet_records.append({
                "timestamp": timestamp,
                "currency": pos.symbol,
                "quote_currency": self._stake_currency,
                "rate": rate,
                "balance": pos.position,
                "total_quote": total_quote,
                "total_position_value": rate * pos.position if rate else None,
                "collateral": pos.collateral,
                "leverage": leverage,
                "bot_managed": base in open_assets,
            })

        for wallet in self.get_all_balances().values():
            rate = self._exchange.get_conversion_rate(wallet.currency, self._stake_currency)
            balance = wallet.total - (
                position_collaterals if wallet.currency == self._stake_currency else 0
            )
            wallet_records.append({
                "timestamp": timestamp,
                "currency": wallet.currency,
                "quote_currency": self._stake_currency,
                "rate": rate,
                "balance": balance,
                "total_quote": rate * balance if rate else None,
                "total_position_value": None,
                "collateral": None,
                "leverage": 1.0,
                "bot_managed": self._stake_currency == wallet.currency or wallet.currency in open_assets,
            })

        if not wallet_records:
            return
        try:
            stmt = sqlite_insert(WalletHistory.__table__).values(wallet_records)
            update_cols = {
                col.name: stmt.excluded[col.name]
                for col in WalletHistory.__table__.columns
                if col.name != "id"
            }
            stmt = stmt.on_conflict_do_update(
                index_elements=["timestamp", "currency"],
                set_=update_cols,
            )
            WalletHistory.session.execute(stmt)
            WalletHistory.session.commit()
        except Exception as e:
            WalletHistory.session.rollback()
            logger.error("Error saving wallet balance records: %s", e)

    Wallets.record_wallet_state = _upsert_record_wallet_state

    try:
        # ---------------------------------------------------------------- #
        # 7. Initialise FreqtradeBot                                         #
        # ---------------------------------------------------------------- #
        bot = FreqtradeBot(config)
        bot.state = State.RUNNING  # Worker normally does this; we bypass Worker

        # Load informative pairs declared by the strategy (e.g. BTC as a filter).
        # These are not in the user's --pairs list so the store doesn't have them yet.
        _load_informative_pairs(bot, store, exchange, config_path, start_dt, end_dt, datadir, trading_mode)

        total_candles = int((end_dt - start_dt).total_seconds() / tf_secs)
        logger.info(
            "Replay ready: %s → %s  |  %d candles × %d pairs  |  tf=%s  |  slippage=%.4f%%",
            start_dt.date(), end_dt.date(), total_candles, len(pairs), tf, slippage_pct * 100,
        )

        # ---------------------------------------------------------------- #
        # 8. Main loop — one process() call per candle                       #
        # ---------------------------------------------------------------- #
        current = start_dt
        processed = 0

        while current < end_dt:
            clock.advance_to(current)
            try:
                bot.process()
            except Exception as exc:
                logger.warning("bot.process() raised at %s: %s", current, exc, exc_info=True)

            processed += 1
            if processed % 24 == 0:
                n_open = len(Trade.get_open_trades())
                n_closed = Trade.get_trades_proxy(is_open=False)
                logger.info(
                    "[%d/%d] %s  open=%d  closed=%d",
                    processed, total_candles, current.strftime("%Y-%m-%d %H:%M"), n_open, len(n_closed),
                )

            current += timedelta(seconds=tf_secs)

        # ---------------------------------------------------------------- #
        # 9. Summary + view config                                           #
        # ---------------------------------------------------------------- #
        _print_summary(db_url, report_path=report_path)
        print("  View in FreqUI: docker compose up freqtrade")
        print()

    finally:
        ExchangeResolver.load_exchange = original_load_exchange
        Worker._sleep = original_sleep
        Wallets.record_wallet_state = original_record_wallet_state
        clock.stop()


def _print_summary(db_url: str, report_path: str | None = None) -> None:
    closed = Trade.get_trades_proxy(is_open=False)
    open_trades = Trade.get_trades_proxy(is_open=True)

    W = 60
    print()
    print("=" * W)
    print("  REPLAY SUMMARY")
    print("=" * W)
    print(f"  DB           : {db_url}")
    print(f"  Closed trades: {len(closed)}")
    print(f"  Open trades  : {len(open_trades)}")

    if not closed:
        print("  No closed trades.")
        print("=" * W)
        return

    profits_pct = [t.close_profit for t in closed if t.close_profit is not None]
    profits_abs = [t.close_profit_abs for t in closed if t.close_profit_abs is not None]

    wins_pct = [p for p in profits_pct if p > 0]
    losses_pct = [p for p in profits_pct if p <= 0]
    wins_abs = [p for p in profits_abs if p > 0]
    losses_abs = [p for p in profits_abs if p <= 0]

    total_pct = sum(profits_pct)
    total_abs = sum(profits_abs)
    win_rate = 100 * len(wins_pct) / len(profits_pct)
    avg_win = 100 * sum(wins_pct) / len(wins_pct) if wins_pct else 0.0
    avg_loss = 100 * sum(losses_pct) / len(losses_pct) if losses_pct else 0.0
    best = 100 * max(profits_pct)
    worst = 100 * min(profits_pct)

    gross_profit = sum(wins_abs)
    gross_loss = abs(sum(losses_abs))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown on running absolute equity
    sorted_trades = sorted(
        (t for t in closed if t.close_date and t.close_profit_abs is not None),
        key=lambda t: t.close_date,
    )
    equity, peak, max_dd_abs = 0.0, 0.0, 0.0
    for t in sorted_trades:
        equity += t.close_profit_abs
        peak = max(peak, equity)
        max_dd_abs = max(max_dd_abs, peak - equity)

    print(f"  Win rate     : {len(wins_pct)}/{len(profits_pct)} ({win_rate:.1f}%)")
    print(f"  Avg win      : {avg_win:+.3f}%")
    print(f"  Avg loss     : {avg_loss:+.3f}%")
    print(f"  Best trade   : {best:+.3f}%")
    print(f"  Worst trade  : {worst:+.3f}%")
    print(f"  Profit factor: {profit_factor:.2f}")
    print(f"  Max drawdown : {max_dd_abs:+.2f} USDT")
    print(f"  Total P&L    : {total_abs:+.2f} USDT  ({100 * total_pct:+.3f}%)")

    # Per-pair breakdown
    pairs: dict[str, dict] = {}
    for t in closed:
        if t.close_profit is None:
            continue
        d = pairs.setdefault(t.pair, {"n": 0, "wins": 0, "pnl_abs": 0.0})
        d["n"] += 1
        d["wins"] += int(t.close_profit > 0)
        d["pnl_abs"] += t.close_profit_abs or 0.0

    print()
    print(f"  {'Pair':<22} {'Trades':>6}  {'Win%':>6}  {'P&L (USDT)':>12}")
    print(f"  {'-'*22} {'-'*6}  {'-'*6}  {'-'*12}")
    for pair, d in sorted(pairs.items(), key=lambda x: x[1]["pnl_abs"], reverse=True):
        wr = 100 * d["wins"] / d["n"] if d["n"] else 0
        print(f"  {pair:<22} {d['n']:>6}  {wr:>5.1f}%  {d['pnl_abs']:>+12.2f}")

    print("=" * W)
    print()
    print(f"  Plot with freqtrade (inside container):")
    print(f"    freqtrade plot-profit --db-url {db_url} \\")
    print(f"      --config /freqtrade/user_data/config_backtest_static.json \\")
    print(f"      --datadir /freqtrade/user_data/data/{{exchange_name}}/futures")
    print()

    if report_path:
        _export_html_report(sorted_trades, open_trades, report_path)
        print(f"  HTML report  : {report_path}")
        print()


def _export_html_report(
    closed: list,
    open_trades: list,
    path: str,
) -> None:
    """Standalone plotly HTML: equity curve + per-pair trade count bar."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.warning("plotly not installed — skipping HTML report")
        return

    # ---- equity curve ----
    # close_date may be FakeDatetime (freezegun) — convert to ISO string so
    # orjson can serialize it.
    dates, equity_vals = [], []
    running = 0.0
    for t in closed:
        running += t.close_profit_abs or 0.0
        dates.append(t.close_date.isoformat())
        equity_vals.append(round(running, 4))

    # ---- per-pair bar ----
    pair_data: dict[str, float] = {}
    for t in closed:
        pair_data[t.pair] = pair_data.get(t.pair, 0.0) + (t.close_profit_abs or 0.0)

    pairs_sorted = sorted(pair_data.items(), key=lambda x: x[1])
    pair_names = [p for p, _ in pairs_sorted]
    pair_pnl = [round(v, 2) for _, v in pairs_sorted]
    bar_colors = ["#ef5350" if v < 0 else "#26a69a" for v in pair_pnl]

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Equity Curve (cumulative P&L, USDT)", "P&L by Pair (USDT)"),
        vertical_spacing=0.12,
    )

    # equity line
    fig.add_trace(
        go.Scatter(
            x=dates, y=equity_vals,
            mode="lines+markers",
            name="Equity",
            line=dict(color="#42a5f5", width=2),
            marker=dict(size=4),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>P&L: %{y:+.2f} USDT<extra></extra>",
        ),
        row=1, col=1,
    )
    # zero line
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)

    # per-pair bar
    fig.add_trace(
        go.Bar(
            x=pair_pnl, y=pair_names,
            orientation="h",
            marker_color=bar_colors,
            name="P&L by pair",
            hovertemplate="%{y}: %{x:+.2f} USDT<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        title="Freqtrade Replay Report",
        template="plotly_dark",
        showlegend=False,
        height=700,
        margin=dict(l=60, r=40, t=80, b=40),
    )
    fig.update_xaxes(title_text="Close date", row=1, col=1)
    fig.update_yaxes(title_text="USDT", row=1, col=1)
    fig.update_xaxes(title_text="P&L (USDT)", row=2, col=1)

    fig.write_html(path, include_plotlyjs="cdn")
    logger.info("HTML report written to %s", path)
