# Security

## Credentials removed before publishing

This repository is public. The material added to it came from a Telegram channel
where people share Freqtrade configurations, and those configurations routinely
contain live credentials. Every file was scanned before publication and every
credential-shaped value was replaced with `REDACTED-BY-REPO-HYGIENE`.

| Source | Files | Values removed |
|---|---:|---:|
| JSON configs | 35 | 131 |
| Python sources | 2 | 2 |

What was found, by kind:

- Telegram bot tokens (`NNNNNNNNN:AA...`) — 9
- Exchange API key / secret pairs — 13, across Hyperliquid, Bitrue and others
- Freqtrade API-server passwords and `jwt_secret_key` values — 57
- An OpenAI-format key hard-coded as a fallback default — 1

The redaction preserves each file's structure, so a config stays valid JSON and a
strategy still parses. You must supply your own credentials.

### ⚠️ One key needs manual revocation

`KMM.py` carried a hard-coded Akash Network API key as a default argument, and
that file **was already public in this repository's history**. Redacting it here
does not invalidate it — it remains in earlier commits and in every existing
clone. **The key holder must revoke it.**

If you want it gone from history entirely, that requires rewriting published
history (`git filter-repo`) and a force-push, which breaks every clone. Revoking
the key is the effective fix; removing it from history is cosmetic by comparison.

## Verification

After redaction the tree was re-scanned and returned zero findings, cross-checked
with direct pattern greps:

```
telegram bot tokens : 0
sk- keys            : 0
AKIA keys           : 0
PRIVATE KEY blocks  : 0
```

## Running anything from this repository

Treat every file here as untrusted third-party code:

- Read a strategy before you load it. Some of these files make network calls.
- Run Freqtrade in a container or a dedicated user account.
- Start in dry-run. Give live API keys trade-only permissions, never withdrawal.
- `dirty LA/` strategies produce backtest results that cannot be achieved live.
