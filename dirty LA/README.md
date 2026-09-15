# dirty LA

Strategies whose source contains a **high-confidence lookahead-bias pattern** —
they read data from candles that had not closed at the moment they claim to
trade. A backtest of anything in this folder is not achievable in live trading.

They are kept rather than deleted because the bias is often a single line, and
the rest of the strategy may be sound. See
[docs/LOOKAHEAD.md](../docs/LOOKAHEAD.md) for the exact line flagged in each file.

- `from-repo/` — 48 files that were already in this repository.
- `from-AIRA/` — 89 files from the Telegram collection.

Detection is static and pattern-based. It is good at finding the common mistakes
and it is not a proof of either guilt or innocence. Before trading any strategy
from this repository, run `freqtrade lookahead-analysis` on it yourself, in a
sandbox.
