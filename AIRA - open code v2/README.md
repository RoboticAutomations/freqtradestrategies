# AIRA - open code v2

Strategies, FreqAI models and tooling collected from the **A.I.R.A. Freqtrade
Strategies** Telegram channel, indexed from the channel's full history
(64,429 messages) and deduplicated against the rest of this repository.

| | |
|---|---:|
| Files | 706 |
| Strategies / models | 331 |
| Flagged for lookahead → `../dirty LA/from-AIRA/` | 89 |

## Provenance

Files are named after the strategy; the Telegram message id each came from was
stripped from the filename but is preserved in
[`../catalog.json`](../catalog.json) under `source_message`.

Source archives contributing here: `TSPredict`, `AlexFreqAlphaDashboardV12`,
`Good_strategies_moutonneux`, `AlexCryptoKingTOP5`, `freqtrade_RUST_OBv2`,
`No.2_PumpBot-master` and others. Two bulk scrapes of already-public collections
(`strategies.zip`, 2,847 files and a 470-file sibling) were excluded as
redundant, and four Kraken trade-history archives (1.8 GB of CSV, no strategies)
were excluded as market data.

## Warning

This is other people's code, shared informally. It has not been reviewed for
correctness or for safety. Credentials found in the original files have been
removed — see [../docs/SECURITY.md](../docs/SECURITY.md).
