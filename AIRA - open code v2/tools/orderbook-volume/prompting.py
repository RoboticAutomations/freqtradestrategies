"""
Prompt templates for DarkProphetLLMAIV2 trading model.
Contains system prompts and user prompt templates for LLM-based trading decisions.
"""

SYSTEM_PROMPT_INDICATORS_ONLY = """You are an expert cryptocurrency trading analyst. Your job is to analyze technical indicators and make CONSERVATIVE trading decisions.

**CRITICAL TRADING PHILOSOPHY**:
1. **PATIENCE IS KEY** - Wait for high-probability setups with multiple confirmations
2. **HOLD BY DEFAULT** - When in doubt, HOLD the position (return NEUTRAL)
3. **AVOID OVERTRADING** - Small price movements are NORMAL, don't panic exit
4. **MULTIPLE CONFIRMATIONS** - Need at least 5 aligned signals for entry, 3 for exit
5. **RESPECT THE TREND** - SuperTrend is your primary guide

**OUTPUT RULES**:
- Return ONLY valid JSON: {"action": "...", "summary": "..."}
- Actions: LONG_ENTER, SHORT_ENTER, LONG_EXIT, SHORT_EXIT, NEUTRAL
- NEUTRAL means: no entry OR hold current position (do NOT exit)
- Summary: cite specific indicator values (max 300 chars)

**BIAS TOWARD HOLDING**:
- If position is open and SuperTrend supports it → NEUTRAL (hold)
- If loss is small (< 70% of stop loss) → NEUTRAL (hold)
- Exit ONLY when multiple strong reversal signals confirm

NO markdown, NO explanations outside JSON."""

USER_PROMPT_ENTER_FULL = """# ENTRY ANALYSIS
**Targets**: Profit {target_profit}% | Duration {target_duration} candles | Stop {stop_loss}%

## INDICATORS:
{indicators}

##  CRITICAL RULES - BE SELECTIVE:

**DEFAULT ACTION IS NEUTRAL**. Only enter when MULTIPLE strong signals align.
Bad entries lead to losses. Wait for high-probability setups.

### WHEN TO STAY OUT (return NEUTRAL):
1. **Market Regime** = CHAOTIC, TRANSITIONAL, or RANGING_VOLATILE → NO ENTRY
2. **Choppiness > 55** → Market is ranging, avoid trend entries
3. **ADX < 20** → No clear trend, avoid entry
4. **Mixed signals** → SuperTrend says one thing, RSI says another
5. **Regime Confidence < 60%** → Uncertain market conditions
6. **Extreme readings** → RSI < 20 or > 80 (wait for pullback)

**If ANY of above is true → NEUTRAL (no entry)**

### LONG ENTRY (return LONG_ENTER):
Enter LONG only when **AT LEAST 5** of these are TRUE:

1.  SuperTrend = BULLISH (+1) ← MANDATORY
2.  RSI between 35-60 (not overbought, room to grow)
3.  ADX > 20 (trend exists)
4.  DI+ > DI- (bullish momentum)
5.  CMF > 0 (buying pressure / accumulation)
6.  EMA alignment: Price > EMA_fast > EMA_slow (uptrend structure)
7.  Choppiness < 50 (trending, not ranging)
8.  Market Regime = STRONG_TREND or WEAK_TREND
9.  Trend Direction = BULLISH
10. VWAP distance < 1.5 ATR (not overextended)

**LONG requires: SuperTrend +1 (mandatory) + 4 other signals = 5 minimum**

### SHORT ENTRY (return SHORT_ENTER):
Enter SHORT only when **AT LEAST 5** of these are TRUE:

1.  SuperTrend = BEARISH (-1) ← MANDATORY
2.  RSI between 40-65 (not oversold, room to fall)
3.  ADX > 20 (trend exists)
4.  DI- > DI+ (bearish momentum)
5.  CMF < 0 (selling pressure / distribution)
6.  EMA alignment: Price < EMA_fast < EMA_slow (downtrend structure)
7.  Choppiness < 50 (trending, not ranging)
8.  Market Regime = STRONG_TREND or WEAK_TREND
9.  Trend Direction = BEARISH
10. VWAP distance > -1.5 ATR (not overextended)

**SHORT requires: SuperTrend -1 (mandatory) + 4 other signals = 5 minimum**

### DECISION LOGIC:
```
IF Market Regime in [CHAOTIC, TRANSITIONAL, RANGING_VOLATILE] → NEUTRAL
IF Choppiness > 55 OR ADX < 20 → NEUTRAL
IF SuperTrend +1 AND long_signals >= 5 → LONG_ENTER
IF SuperTrend -1 AND short_signals >= 5 → SHORT_ENTER
ELSE → NEUTRAL
```

###  DO NOT ENTER IF:
- SuperTrend not aligned with intended direction
- Less than 5 confirming signals
- Choppy/ranging market conditions
- Extreme RSI (overbought/oversold)
- Conflicting indicator readings

**Response format** (ONLY valid JSON):
```json
{{
  "action": "LONG_ENTER" or "SHORT_ENTER" or "NEUTRAL",
  "summary": "<count signals found, list key ones, explain decision>"
}}
```

**Examples:**
- NO ENTRY: {{"action": "NEUTRAL", "summary": "3/10 signals only. ST+1 but ADX 18<20, Chop 58>55, Regime TRANSITIONAL. Insufficient confirmation."}}
- LONG: {{"action": "LONG_ENTER", "summary": "7/10 signals: ST+1, RSI 45, ADX 28, DI+>DI-, CMF 0.12, EMA aligned, Chop 38, Regime STRONG_TREND. High probability setup."}}
- SHORT: {{"action": "SHORT_ENTER", "summary": "6/10 signals: ST-1, RSI 58, ADX 32, DI->DI+, CMF -0.09, Chop 42, Regime WEAK_TREND. Confirmed bearish."}}

Respond with ONLY valid JSON."""

USER_PROMPT_EXIT_FULL = """# EXIT OR HOLD ANALYSIS
**Position**: {side} | **Profit**: {profit:.2f}% | **Duration**: {duration} candles
**Targets**: Profit {target_profit}% | Duration {target_duration} candles | Stop {stop_loss}%

## INDICATORS:
{indicators}

##  CRITICAL RULES - READ CAREFULLY:

**DEFAULT ACTION IS HOLD (NEUTRAL)**. Only exit when MULTIPLE strong signals confirm.
Small price movements against position are NORMAL - do NOT exit on minor fluctuations.

### WHEN TO HOLD (return NEUTRAL):
1. **SuperTrend still aligned** with position (ST=+1 for LONG, ST=-1 for SHORT)
2. **Loss is within tolerance** (loss < 70% of stop loss)
3. **No extreme indicator readings** (RSI between 25-75)
4. **Trend still intact** (ADX > 15 with DI aligned to position)
5. **Duration < 80%** of target duration
6. **Market Regime** is STRONG_TREND or WEAK_TREND (aligned with position direction)

**If ANY of above is true → HOLD (return NEUTRAL)**

### WHEN TO EXIT (return {side}_EXIT):
Exit ONLY when **AT LEAST 3** of these conditions are TRUE simultaneously:

**For LONG positions - Exit signals:**
1. SuperTrend flipped to BEARISH (ST = -1) ← CRITICAL
2. RSI > 75 (extreme overbought)
3. ADX > 25 AND DI- > DI+ (bearish momentum confirmed)
4. CMF < -0.15 (strong distribution)
5. Bearish divergence detected
6. Loss >= 80% of stop loss (risk management)
7. Profit >= 80% of target (secure gains)
8. Market Regime = CHAOTIC or RANGING_VOLATILE

**For SHORT positions - Exit signals:**
1. SuperTrend flipped to BULLISH (ST = +1) ← CRITICAL
2. RSI < 25 (extreme oversold)
3. ADX > 25 AND DI+ > DI- (bullish momentum confirmed)
4. CMF > 0.15 (strong accumulation)
5. Bullish divergence detected
6. Loss >= 80% of stop loss (risk management)
7. Profit >= 80% of target (secure gains)
8. Market Regime = CHAOTIC or RANGING_VOLATILE

### DECISION LOGIC:
```
IF SuperTrend aligned with position AND loss < 70% stop → NEUTRAL (HOLD)
IF exit_signals_count >= 3 → {side}_EXIT
ELSE → NEUTRAL (HOLD)
```

###  DO NOT EXIT IF:
- Only 1-2 exit signals (wait for confirmation)
- Small temporary loss (< 50% of stop loss)
- RSI moved slightly against but not extreme
- SuperTrend still supports position
- Just noise/normal volatility

**Response format** (ONLY valid JSON):
```json
{{
  "action": "{side}_EXIT" or "NEUTRAL",
  "summary": "<count exit signals found, list them, explain decision>"
}}
```

**Examples:**
- HOLD: {{"action": "NEUTRAL", "summary": "2/8 exit signals (RSI 68, CMF -0.08). ST+1 aligned, ADX 22 DI+>DI-, loss -0.8% within tolerance. HOLD."}}
- EXIT: {{"action": "LONG_EXIT", "summary": "5/8 exit signals: ST flipped -1, RSI 78, DI->DI+ ADX 32, CMF -0.22, profit 2.1%>=80% target. EXIT confirmed."}}

Respond with ONLY valid JSON."""