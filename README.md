# freqtradestrategies

A collection of [Freqtrade](https://www.freqtrade.io) strategies, FreqAI models and
supporting tooling, gathered from public sources and community channels.

> **Use at your own risk.** Nothing here has been verified as profitable. Most of
> these files are experiments, forks, or works in progress by third parties.
> Backtest and dry-run everything before it touches real money. See
> [Known problems](#known-problems) before you trust any result.

## What is in here

| | |
|---|---:|
| Python files | **1,286** |
| Freqtrade strategies / FreqAI models | **909** |
| Supporting tools, bots and libraries | **377** |
| Duplicate files collapsed by SHA-256 | **515** |
| 🔴 Flagged for lookahead bias | **137** |
| 🟡 Worth a second look | **90** |

Every file was parsed with Python's `ast` module to pull out its strategy class,
timeframe, stoploss, hyperopt parameters and indicator usage. **No file in this
repository was executed** in order to document it, and you should extend the same
suspicion to anything you run from here.

## Layout

```
AIRA - open code v2/         617 files
dirty LA/                    137 files
strategies/                  530 files
tools/                         2 files
```

- **`strategies/`** — the long-standing collection, now grouped by family
  (`ichimoku/`, `bollinger/`, `nostalgia-for-infinity/`, …) instead of one flat
  directory. Hyperopt `.json` parameter files travel with their strategy, which
  is what Freqtrade expects.
- **`AIRA - open code v2/`** — strategies and tooling collected from the
  A.I.R.A. Freqtrade Strategies Telegram channel. Same grouping.
- **`dirty LA/`** — strategies whose source contains a high-confidence
  lookahead-bias pattern. They are kept, not deleted, because several are useful
  once the offending line is fixed. **Do not trust their backtests.**
- **`tools/`**, **`*/tools/`** — scanners, Telegram bots, pairlist helpers,
  forecasting libraries and strategies for *other* bots (PyCryptoBot and
  friends). Not loadable by Freqtrade.
- **`*/configs/`** — Freqtrade configuration files. **All credentials have been
  stripped**; see [docs/SECURITY.md](docs/SECURITY.md).

## Where to start

- **[CATALOG.md](CATALOG.md)** — every strategy, with timeframe, indicators and
  lookahead status.
- **[docs/LOOKAHEAD.md](docs/LOOKAHEAD.md)** — how the bias detection works and
  what it found, line by line.
- **[docs/ORGANIZATION.md](docs/ORGANIZATION.md)** — the layout, and a map from
  every old path to its new one.
- **[docs/SECURITY.md](docs/SECURITY.md)** — what was redacted before publishing.

## Most common timeframes

- `5m` — 475 strategies
- `15m` — 151 strategies
- `1h` — 149 strategies
- `1m` — 42 strategies
- `4h` — 31 strategies
- `30m` — 13 strategies
- `1d` — 10 strategies
- `3m` — 5 strategies

## Most common indicators

- RSI — 658
- EMA — 594
- SMA — 477
- Bollinger — 367
- ATR — 339
- ADX — 236
- MACD — 223
- Stochastic — 199
- Heikin-Ashi — 198
- Elliott Wave — 190
- CCI — 136
- Williams %R — 111

## Known problems

1. **Lookahead bias.** 137 strategies use future data. A backtest on these
   will look extraordinary and will not survive live trading. They live in
   `dirty LA/`.
2. **Unverified third-party code.** These files came from public channels.
   Read a strategy before you run it. Several contained hard-coded API keys,
   which have been removed — treat that as a sign of the general standard of care.
3. **Duplicates and forks.** 515 byte-identical copies were collapsed.
   Many remaining files are near-identical forks that differ only in tuned
   parameters.
4. **Mixed Freqtrade versions.** Strategies here target different interface
   versions; older ones use `populate_buy_trend`/`populate_sell_trend` and need
   updating for current Freqtrade.

## Licence and attribution

These files were collected from public sources; individual authors retain their
rights. Where a file carries an author or licence header it has been preserved.
If you own something here and want it removed, open an issue.
