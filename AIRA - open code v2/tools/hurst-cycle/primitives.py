"""
SMC / Structure-Based Primitives
=================================
All functions are vectorized and look-ahead-free.

Each docstring states:
  "Valid at bar i+k" — meaning the output at index j can only be used by a strategy
  at bar j (it contains no information from bars after j).

Look-ahead audit for every primitive is documented inline.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder ATR. Valid at bar i (uses only close[i-1] and earlier).
    No look-ahead: ewm with adjust=False is a causal filter.
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Causal EMA. Valid at bar i."""
    return series.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Swing Highs / Lows
# ---------------------------------------------------------------------------

def swing_hl(high: pd.Series, low: pd.Series, n: int = 2):
    """
    Confirmed fractal swing high/low.

    A swing high at bar i requires: high[i] > high[i-k] for all k in 1..n
                                AND high[i] > high[i+k] for all k in 1..n
    The confirmation of bar i's swing is only known at bar i+n (after n right-side
    bars have closed).

    Output:
        sh[j] = price of the swing high whose confirmation bar is j  (NaN otherwise)
        sl[j] = price of the swing low  whose confirmation bar is j  (NaN otherwise)

    Valid at bar j = i+n.  Bars 0..n-1 and the last n bars are NaN.

    Look-ahead audit: bar i+n is the EARLIEST bar at which all n right-side comparisons
    are available. The output index is shifted by +n, so no future information leaks
    into earlier bars.
    """
    h = high.values
    l = low.values
    N = len(h)

    is_sh = np.zeros(N, dtype=bool)
    is_sl = np.zeros(N, dtype=bool)

    for i in range(n, N - n):
        left_h = h[i - n:i]
        right_h = h[i + 1:i + n + 1]
        if np.all(h[i] > left_h) and np.all(h[i] > right_h):
            is_sh[i] = True

        left_l = l[i - n:i]
        right_l = l[i + 1:i + n + 1]
        if np.all(l[i] < left_l) and np.all(l[i] < right_l):
            is_sl[i] = True

    # Shift output forward by n to represent the confirmation bar.
    sh_out = np.full(N, np.nan)
    sl_out = np.full(N, np.nan)
    for i in range(N - n):
        if is_sh[i]:
            sh_out[i + n] = h[i]
        if is_sl[i]:
            sl_out[i + n] = l[i]

    return pd.Series(sh_out, index=high.index), pd.Series(sl_out, index=low.index)


def last_swing(sh_confirmed: pd.Series, sl_confirmed: pd.Series):
    """
    Running last-known confirmed swing high and low at each bar.
    Forward-fills the confirmed swing arrays.

    Valid at bar i: contains the most recent confirmed swing as of bar i.
    No look-ahead: ffill only propagates backward-confirmed values forward.
    """
    last_sh = sh_confirmed.ffill()
    last_sl = sl_confirmed.ffill()
    return last_sh, last_sl


# ---------------------------------------------------------------------------
# BOS / CHoCH  (state machine — causal loop)
# ---------------------------------------------------------------------------

def bos_choch(close: pd.Series, sh_confirmed: pd.Series, sl_confirmed: pd.Series):
    """
    Break-of-Structure and Change-of-Character state machine.

    Structural logic:
      Bullish bias  → watch last swing high for BOS_BULL (continuation)
                    → watch last swing low  for CHoCH_BEAR (reversal)
      Bearish bias  → watch last swing low  for BOS_BEAR  (continuation)
                    → watch last swing high for CHoCH_BULL (reversal)
      Unclear       → first break in either direction sets bias (MSS_BULL / MSS_BEAR)

    Each level is 'consumed' after it is broken — no re-trigger on the same level.
    A new confirmed swing of that type must form before the next event can fire.

    Output:
        bias[i]  : int8 — 1 (bullish), -1 (bearish), 0 (unclear)
        event[i] : str  — 'BOS_BULL', 'BOS_BEAR', 'CHoCH_BULL', 'CHoCH_BEAR',
                          'MSS_BULL', 'MSS_BEAR', or ''

    Valid at bar i (close[i] is fully known; swing inputs are already shifted by n).
    Look-ahead audit: uses only close[i] and sh/sl values that were confirmed at or
    before bar i. The state is updated sequentially — no future information.
    """
    c = close.values
    sh = sh_confirmed.values
    sl = sl_confirmed.values
    N = len(c)

    bias_arr = np.zeros(N, dtype=np.int8)
    event_arr = np.full(N, '', dtype=object)

    current_bias = 0         # 0=unclear, 1=bull, -1=bear
    watch_sh = np.nan        # active ceiling level
    watch_sl = np.nan        # active floor level

    for i in range(N):
        # Absorb any newly confirmed swing levels
        if not np.isnan(sh[i]):
            watch_sh = sh[i]
        if not np.isnan(sl[i]):
            watch_sl = sl[i]

        if current_bias == 1:
            if not np.isnan(watch_sh) and c[i] > watch_sh:
                event_arr[i] = 'BOS_BULL'
                watch_sh = np.nan          # consumed; wait for next swing high
            elif not np.isnan(watch_sl) and c[i] < watch_sl:
                event_arr[i] = 'CHoCH_BEAR'
                current_bias = -1
                watch_sl = np.nan

        elif current_bias == -1:
            if not np.isnan(watch_sl) and c[i] < watch_sl:
                event_arr[i] = 'BOS_BEAR'
                watch_sl = np.nan
            elif not np.isnan(watch_sh) and c[i] > watch_sh:
                event_arr[i] = 'CHoCH_BULL'
                current_bias = 1
                watch_sh = np.nan

        else:  # unclear — first break establishes bias
            if not np.isnan(watch_sh) and c[i] > watch_sh:
                event_arr[i] = 'MSS_BULL'
                current_bias = 1
                watch_sh = np.nan
            elif not np.isnan(watch_sl) and c[i] < watch_sl:
                event_arr[i] = 'MSS_BEAR'
                current_bias = -1
                watch_sl = np.nan

        bias_arr[i] = current_bias

    return pd.Series(bias_arr, index=close.index), pd.Series(event_arr, index=close.index)


# ---------------------------------------------------------------------------
# Fair Value Gaps (FVG)
# ---------------------------------------------------------------------------

def detect_fvg(high: pd.Series, low: pd.Series):
    """
    3-bar imbalance zones (Fair Value Gaps).

    Bullish FVG at bar i:  low[i]  > high[i-2]  →  zone = [high[i-2], low[i]]
    Bearish FVG at bar i:  high[i] < low[i-2]   →  zone = [high[i],   low[i-2]]

    Confirmed at bar i (all 3 candles are closed). No look-ahead.

    Output: bull_top, bull_bot, bear_top, bear_bot — all NaN when no FVG at that bar.

    Look-ahead audit: only uses high[i-2] and low[i-2] (past bars) plus current
    bar's high/low (bar i is already closed when this is computed). No future bars used.
    """
    h = high.values
    l = low.values
    N = len(h)

    bull_top = np.full(N, np.nan)
    bull_bot = np.full(N, np.nan)
    bear_top = np.full(N, np.nan)
    bear_bot = np.full(N, np.nan)

    for i in range(2, N):
        if l[i] > h[i - 2]:                  # bullish gap
            bull_top[i] = l[i]
            bull_bot[i] = h[i - 2]
        if h[i] < l[i - 2]:                  # bearish gap
            bear_top[i] = l[i - 2]
            bear_bot[i] = h[i]

    return (
        pd.Series(bull_top, index=high.index),
        pd.Series(bull_bot, index=high.index),
        pd.Series(bear_top, index=high.index),
        pd.Series(bear_bot, index=high.index),
    )


def active_fvg(bull_top: pd.Series, bull_bot: pd.Series,
               bear_top: pd.Series, bear_bot: pd.Series,
               low: pd.Series, high: pd.Series):
    """
    Track currently-active (unmitigated) FVG zones.

    A bullish FVG is active from the bar it forms until price trades into it
    (low[i] <= bull_top, i.e. CE touched). Similarly for bearish.

    Returns the most recently active FVG bounds at each bar (NaN if none active).
    This is used for 'price returning to FVG' entry conditions.

    Look-ahead audit: mitigation is checked BEFORE absorbing new FVGs each bar,
    and the state is recorded before new FVGs are added. This ensures a FVG formed
    at bar i is first visible at bar i+1 — the formation bar never self-mitigates.
    Fully causal.
    """
    N = len(low)
    bt = bull_top.values
    bb = bull_bot.values
    brt = bear_top.values
    brb = bear_bot.values
    l = low.values
    h = high.values

    act_bull_top = np.full(N, np.nan)
    act_bull_bot = np.full(N, np.nan)
    act_bear_top = np.full(N, np.nan)
    act_bear_bot = np.full(N, np.nan)

    cur_bull_top = np.nan
    cur_bull_bot = np.nan
    cur_bear_top = np.nan
    cur_bear_bot = np.nan

    for i in range(N):
        # 1. Mitigate existing zones with this bar's price action (before absorbing
        #    new FVGs). This ensures the formation bar never self-mitigates.
        if not np.isnan(cur_bull_top) and l[i] <= cur_bull_top:
            cur_bull_top = np.nan
            cur_bull_bot = np.nan
        if not np.isnan(cur_bear_top) and h[i] >= cur_bear_bot:
            cur_bear_top = np.nan
            cur_bear_bot = np.nan

        # 2. Record the active state for this bar (post-mitigation, pre-new-FVG).
        #    New FVGs formed at bar i are only visible from bar i+1 onward.
        act_bull_top[i] = cur_bull_top
        act_bull_bot[i] = cur_bull_bot
        act_bear_top[i] = cur_bear_top
        act_bear_bot[i] = cur_bear_bot

        # 3. Absorb new FVG formed this bar — available to strategies next bar.
        if not np.isnan(bt[i]):
            cur_bull_top = bt[i]
            cur_bull_bot = bb[i]
        if not np.isnan(brt[i]):
            cur_bear_top = brt[i]
            cur_bear_bot = brb[i]

    return (
        pd.Series(act_bull_top, index=low.index),
        pd.Series(act_bull_bot, index=low.index),
        pd.Series(act_bear_top, index=low.index),
        pd.Series(act_bear_bot, index=low.index),
    )


# ---------------------------------------------------------------------------
# Displacement
# ---------------------------------------------------------------------------

def detect_displacement(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_series: pd.Series,
    body_ratio: float = 0.6,
    atr_mult: float = 1.5,
):
    """
    Displacement candle: strong directional move that signals 'intent'.

    Conditions:
      body / range >= body_ratio   (strong close relative to wick)
      range > atr_mult * ATR       (above-average size)

    Returns:
      disp_bull: bool Series — bullish displacement (close > open)
      disp_bear: bool Series — bearish displacement (close < open)

    Valid at bar i (bar is closed). No look-ahead.
    """
    o = open_.values
    h = high.values
    l = low.values
    c = close.values
    a = atr_series.values

    rng = h - l
    body = np.abs(c - o)
    body_r = np.where(rng > 0, body / rng, 0.0)

    is_disp = (body_r >= body_ratio) & (rng > atr_mult * a)
    disp_bull = pd.Series(is_disp & (c > o), index=open_.index)
    disp_bear = pd.Series(is_disp & (c < o), index=open_.index)
    return disp_bull, disp_bear


# ---------------------------------------------------------------------------
# Liquidity Sweeps
# ---------------------------------------------------------------------------

def detect_sweep(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    sh_confirmed: pd.Series,
    sl_confirmed: pd.Series,
):
    """
    Single-bar liquidity sweep detection.

    SSL sweep (bullish):
      wick below the most recently confirmed swing low AND bar closes above it.
      Interpretation: sell-side liquidity taken, potential long opportunity.

    BSL sweep (bearish):
      wick above the most recently confirmed swing high AND bar closes below it.
      Interpretation: buy-side liquidity taken, potential short opportunity.

    Returns:
      sweep_ssl: bool Series — SSL swept this bar (bullish implication)
      sweep_bsl: bool Series — BSL swept this bar (bearish implication)
      sweep_ssl_level: float Series — the swing low level that was swept (NaN if no sweep)
      sweep_bsl_level: float Series — the swing high level that was swept (NaN if no sweep)

    Valid at bar i (uses bar's high/low/close, all closed). No look-ahead.
    Look-ahead audit: sh_confirmed/sl_confirmed are already shifted by n bars in
    swing_hl(), so they only contain information available at or before bar i.
    """
    h = high.values
    l = low.values
    c = close.values
    sh = sh_confirmed.values
    sl = sl_confirmed.values
    N = len(c)

    sweep_ssl = np.zeros(N, dtype=bool)
    sweep_bsl = np.zeros(N, dtype=bool)
    ssl_level = np.full(N, np.nan)
    bsl_level = np.full(N, np.nan)

    last_sh = np.nan
    last_sl = np.nan

    for i in range(N):
        if not np.isnan(sh[i]):
            last_sh = sh[i]
        if not np.isnan(sl[i]):
            last_sl = sl[i]

        # SSL sweep: wick below last swing low, close back above
        if not np.isnan(last_sl) and l[i] < last_sl and c[i] > last_sl:
            sweep_ssl[i] = True
            ssl_level[i] = last_sl
            last_sl = np.nan          # level swept; wait for new swing low

        # BSL sweep: wick above last swing high, close back below
        if not np.isnan(last_sh) and h[i] > last_sh and c[i] < last_sh:
            sweep_bsl[i] = True
            bsl_level[i] = last_sh
            last_sh = np.nan

    return (
        pd.Series(sweep_ssl, index=high.index),
        pd.Series(sweep_bsl, index=high.index),
        pd.Series(ssl_level, index=high.index),
        pd.Series(bsl_level, index=high.index),
    )


# ---------------------------------------------------------------------------
# Order Blocks
# ---------------------------------------------------------------------------

def detect_order_block(
    open_: pd.Series,
    close: pd.Series,
    disp_bull: pd.Series,
    disp_bear: pd.Series,
    lookback: int = 5,
):
    """
    Order block: the last opposite-color candle immediately preceding a displacement.

    Bullish OB: last bearish candle (close < open) in the lookback bars before a
                bullish displacement.  Zone = [low[ob_bar], high[ob_bar]].
    Bearish OB: last bullish candle (close > open) in the lookback bars before a
                bearish displacement.  Zone = [low[ob_bar], high[ob_bar]].

    Output:
      bull_ob_top, bull_ob_bot: bullish OB zone bounds (NaN if no OB at bar i)
      bear_ob_top, bear_ob_bot: bearish OB zone bounds (NaN if no OB at bar i)

    Confirmed at the displacement bar i. Look-ahead audit: all candles in the lookback
    window (i-lookback .. i-1) are fully closed before bar i closes. No future bars used.
    """
    o = open_.values
    c = close.values
    db = disp_bull.values
    dbe = disp_bear.values
    N = len(o)

    # We need high/low for the OB candle — accept them via the same df later.
    # For now return the OB bar index so callers can extract high/low.
    bull_ob_idx = np.full(N, -1, dtype=np.int32)
    bear_ob_idx = np.full(N, -1, dtype=np.int32)

    for i in range(lookback, N):
        if db[i]:  # bullish displacement at bar i → find last bearish candle before
            for j in range(i - 1, max(i - lookback - 1, -1), -1):
                if c[j] < o[j]:           # bearish candle
                    bull_ob_idx[i] = j
                    break
        if dbe[i]:  # bearish displacement → find last bullish candle before
            for j in range(i - 1, max(i - lookback - 1, -1), -1):
                if c[j] > o[j]:           # bullish candle
                    bear_ob_idx[i] = j
                    break

    return pd.Series(bull_ob_idx, index=open_.index), pd.Series(bear_ob_idx, index=open_.index)


def ob_zones(high: pd.Series, low: pd.Series,
             bull_ob_idx: pd.Series, bear_ob_idx: pd.Series):
    """
    Resolve order-block candle indices to price zones.
    Returns top/bottom for bull and bear OBs (NaN where no OB).
    """
    h = high.values
    l = low.values
    bi = bull_ob_idx.values
    bri = bear_ob_idx.values
    N = len(h)

    bull_top = np.full(N, np.nan)
    bull_bot = np.full(N, np.nan)
    bear_top = np.full(N, np.nan)
    bear_bot = np.full(N, np.nan)

    for i in range(N):
        if bi[i] >= 0:
            bull_top[i] = h[bi[i]]
            bull_bot[i] = l[bi[i]]
        if bri[i] >= 0:
            bear_top[i] = h[bri[i]]
            bear_bot[i] = l[bri[i]]

    return (
        pd.Series(bull_top, index=high.index),
        pd.Series(bull_bot, index=high.index),
        pd.Series(bear_top, index=high.index),
        pd.Series(bear_bot, index=high.index),
    )


# ---------------------------------------------------------------------------
# OTE Zone  (Optimal Trade Entry)
# ---------------------------------------------------------------------------

def ote_zone(sh_confirmed: pd.Series, sl_confirmed: pd.Series,
             fib_lo: float = 0.62, fib_hi: float = 0.79):
    """
    OTE retracement zone derived from the most recent confirmed impulse leg.

    The impulse leg is defined as: the distance from the last confirmed swing low
    to the last confirmed swing high (for bullish OTE), or vice versa.

    Bullish OTE zone = impulse_low + fib_lo*(range) .. impulse_low + fib_hi*(range)
    (i.e. 62%–79% retracement of the up-leg, still above the swing low)

    Returns:
      ote_bull_hi, ote_bull_lo: upper and lower OTE bounds for long entries
      ote_bear_hi, ote_bear_lo: upper and lower OTE bounds for short entries

    Valid at bar i. Look-ahead audit: uses forward-filled confirmed swings which
    only reflect information confirmed at or before bar i.
    """
    last_sh, last_sl = last_swing(sh_confirmed, sl_confirmed)
    sh_v = last_sh.values
    sl_v = last_sl.values
    N = len(sh_v)

    ote_bull_hi = np.full(N, np.nan)
    ote_bull_lo = np.full(N, np.nan)
    ote_bear_hi = np.full(N, np.nan)
    ote_bear_lo = np.full(N, np.nan)

    for i in range(N):
        if not np.isnan(sh_v[i]) and not np.isnan(sl_v[i]):
            rng = sh_v[i] - sl_v[i]
            if rng > 0:
                # Bull OTE: retrace into the up-leg
                ote_bull_hi[i] = sl_v[i] + (1 - fib_lo) * rng   # 38% from top
                ote_bull_lo[i] = sl_v[i] + (1 - fib_hi) * rng   # 21% from top
                # Bear OTE: retrace into the down-leg
                ote_bear_hi[i] = sh_v[i] - (1 - fib_hi) * rng
                ote_bear_lo[i] = sh_v[i] - (1 - fib_lo) * rng

    return (
        pd.Series(ote_bull_hi, index=sh_confirmed.index),
        pd.Series(ote_bull_lo, index=sh_confirmed.index),
        pd.Series(ote_bear_hi, index=sh_confirmed.index),
        pd.Series(ote_bear_lo, index=sh_confirmed.index),
    )


# ---------------------------------------------------------------------------
# Regime — Hurst Exponent (R/S method)
# ---------------------------------------------------------------------------

def hurst_rs(series: pd.Series, window: int = 100, min_periods: int = 50) -> pd.Series:
    """
    Rolling Hurst exponent via rescaled-range (R/S) analysis.

    H > 0.55 → trending (momentum regime)
    H < 0.45 → mean-reverting
    0.45 ≤ H ≤ 0.55 → random walk / unclear

    Uses a single lag equal to window//2 for the R/S estimate (fast approximation).
    Valid at bar i (uses only past `window` bars). No look-ahead.
    """
    vals = series.values
    N = len(vals)
    H = np.full(N, np.nan)

    for i in range(min_periods, N):
        start = max(0, i - window + 1)
        seg = vals[start:i + 1]
        n = len(seg)
        if n < min_periods:
            continue

        mean = np.mean(seg)
        deviations = np.cumsum(seg - mean)
        R = deviations.max() - deviations.min()
        S = np.std(seg, ddof=1)
        if S == 0:
            continue

        rs = R / S
        if rs <= 0:
            continue

        # H = log(R/S) / log(n/2) — single-lag approximation
        H[i] = np.log(rs) / np.log(n / 2.0)

    return pd.Series(H, index=series.index)


# ---------------------------------------------------------------------------
# Structure Health Score
# ---------------------------------------------------------------------------

def structure_health(
    sh_confirmed: pd.Series,
    sl_confirmed: pd.Series,
    bias: pd.Series,
    window: int = 10,
) -> pd.Series:
    """
    Continuous score [0, 1] reflecting how 'clean' the current trend structure is.

    In a bullish trend:
      Score = fraction of recent confirmed swing highs that are higher than the
              previous one (Higher Highs) AND recent swing lows higher than previous
              (Higher Lows).

    In bearish: mirror (Lower Highs + Lower Lows).

    A score near 1 = textbook trending structure; near 0 = choppy/unclear.
    Valid at bar i. No look-ahead.
    """
    sh = sh_confirmed.values
    sl = sl_confirmed.values
    b = bias.values
    N = len(sh)

    score = np.full(N, np.nan)

    sh_prices = []
    sl_prices = []

    for i in range(N):
        if not np.isnan(sh[i]):
            sh_prices.append(sh[i])
        if not np.isnan(sl[i]):
            sl_prices.append(sl[i])

        recent_sh = sh_prices[-window:] if len(sh_prices) >= 2 else []
        recent_sl = sl_prices[-window:] if len(sl_prices) >= 2 else []

        if len(recent_sh) < 2 or len(recent_sl) < 2:
            score[i] = 0.5
            continue

        if b[i] == 1:   # bullish — want HH + HL
            hh = sum(1 for a, c in zip(recent_sh, recent_sh[1:]) if c > a)
            hl = sum(1 for a, c in zip(recent_sl, recent_sl[1:]) if c > a)
            total = (len(recent_sh) - 1) + (len(recent_sl) - 1)
            score[i] = (hh + hl) / total if total > 0 else 0.5
        elif b[i] == -1:  # bearish — want LH + LL
            lh = sum(1 for a, c in zip(recent_sh, recent_sh[1:]) if c < a)
            ll = sum(1 for a, c in zip(recent_sl, recent_sl[1:]) if c < a)
            total = (len(recent_sh) - 1) + (len(recent_sl) - 1)
            score[i] = (lh + ll) / total if total > 0 else 0.5
        else:
            score[i] = 0.5

    return pd.Series(score, index=sh_confirmed.index)


# ---------------------------------------------------------------------------
# Imbalance / Displacement Intensity
# ---------------------------------------------------------------------------

def displacement_intensity(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_series: pd.Series,
) -> pd.Series:
    """
    Continuous displacement intensity score per bar.

    Combines:
      - body/range ratio (0..1)
      - range / ATR (normalized size)
      - directional sign: positive = bullish, negative = bearish

    Score = sign(close - open) * (body/range) * (range / ATR)

    Positive strong value = strong bullish displacement bar.
    Negative strong value = strong bearish displacement bar.
    Valid at bar i. No look-ahead.
    """
    o = open_.values
    h = high.values
    l = low.values
    c = close.values
    a = atr_series.values

    rng = h - l
    body = c - o                          # signed body
    body_ratio = np.where(rng > 0, np.abs(body) / rng, 0.0)
    size_ratio = np.where(a > 0, rng / a, 0.0)
    direction = np.sign(body)

    intensity = direction * body_ratio * size_ratio

    return pd.Series(intensity, index=open_.index)


# ---------------------------------------------------------------------------
# Convenience: compute all primitives on a DataFrame
# ---------------------------------------------------------------------------

def compute_all(df: pd.DataFrame, swing_n: int = 2, atr_period: int = 14) -> pd.DataFrame:
    """
    Run all primitives on a standard OHLCV DataFrame.
    Returns a new DataFrame with all indicator columns added.

    Expects columns: open, high, low, close, volume (standard Freqtrade names).
    """
    out = df.copy()

    atr_s = atr(df['high'], df['low'], df['close'], atr_period)
    out['atr'] = atr_s
    out['ema200'] = ema(df['close'], 200)

    sh, sl = swing_hl(df['high'], df['low'], swing_n)
    out['sh_confirmed'] = sh
    out['sl_confirmed'] = sl
    out['last_sh'], out['last_sl'] = last_swing(sh, sl)

    bias, event = bos_choch(df['close'], sh, sl)
    out['struct_bias'] = bias
    out['struct_event'] = event

    out['hurst'] = hurst_rs(df['close'])
    out['struct_health'] = structure_health(sh, sl, bias)
    out['disp_intensity'] = displacement_intensity(
        df['open'], df['high'], df['low'], df['close'], atr_s)

    db, dbe = detect_displacement(df['open'], df['high'], df['low'], df['close'], atr_s)
    out['disp_bull'] = db
    out['disp_bear'] = dbe

    fbt, fbb, fbrt, fbrb = detect_fvg(df['high'], df['low'])
    out['fvg_bull_top'] = fbt
    out['fvg_bull_bot'] = fbb
    out['fvg_bear_top'] = fbrt
    out['fvg_bear_bot'] = fbrb

    sweep_ssl, sweep_bsl, ssl_lvl, bsl_lvl = detect_sweep(
        df['high'], df['low'], df['close'], sh, sl)
    out['sweep_ssl'] = sweep_ssl
    out['sweep_bsl'] = sweep_bsl
    out['sweep_ssl_level'] = ssl_lvl
    out['sweep_bsl_level'] = bsl_lvl

    ob_bull_idx, ob_bear_idx = detect_order_block(df['open'], df['close'], db, dbe)
    bt, bb, brt, brb = ob_zones(df['high'], df['low'], ob_bull_idx, ob_bear_idx)
    out['ob_bull_top'] = bt
    out['ob_bull_bot'] = bb
    out['ob_bear_top'] = brt
    out['ob_bear_bot'] = brb

    obe_bh, obe_bl, obe_beh, obe_bel = ote_zone(sh, sl)
    out['ote_bull_hi'] = obe_bh
    out['ote_bull_lo'] = obe_bl
    out['ote_bear_hi'] = obe_beh
    out['ote_bear_lo'] = obe_bel

    return out
