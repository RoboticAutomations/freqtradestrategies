#!/usr/bin/env python3
"""With Love from DarkReaper."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


def next_pow2(n: int) -> int:
    n = max(1, int(n))
    return 1 << (n - 1).bit_length()


def linear_detrend(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    L = len(x)
    if L < 3:
        return x - np.nanmean(x)
    j = np.arange(L, dtype=float)
    jc = j - (L - 1) / 2.0
    denom = float(np.dot(jc, jc))
    if denom <= 1e-12:
        return x - np.nanmean(x)
    mu = float(np.nanmean(x))
    b = float(np.dot(jc, x - mu)) / denom
    a = mu - b * (L - 1) / 2.0
    return x - (a + b * j)


def robust_z(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if np.isfinite(mad) and mad > eps:
        z = (x - med) / (1.4826 * mad + eps)
    else:
        sigma = np.nanstd(x, ddof=1)
        z = (x - np.nanmean(x)) / (sigma + eps) if np.isfinite(sigma) and sigma > eps else x - np.nanmean(x)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z - float(np.mean(z))


def quinn2_delta(spectrum: np.ndarray, k: int) -> float:
    """Quinn's second estimator. Returns fractional bin offset in [-0.75, 0.75]."""
    if k <= 0 or k >= len(spectrum) - 1:
        return 0.0
    xk = spectrum[k]
    denom = abs(xk)
    if denom < 1e-18:
        return 0.0
    am = (spectrum[k - 1] / xk).real
    ap = (spectrum[k + 1] / xk).real
    dm = am / (am - 1.0) if abs(am - 1.0) > 1e-12 else 0.0
    dp = ap / (1.0 - ap) if abs(1.0 - ap) > 1e-12 else 0.0

    def kappa(x: float) -> float:
        x = float(np.clip(x, 0.0, 0.999))
        num = x + 1.0 - np.sqrt(2.0 / 3.0)
        den = x + 1.0 + np.sqrt(2.0 / 3.0)
        return 0.25 * np.log(3.0 * x * x + 6.0 * x + 1.0) - (np.sqrt(6.0) / 24.0) * np.log(num / den)

    delta = 0.5 * (dm + dp) + kappa(dm * dm) - kappa(dp * dp)
    if not np.isfinite(delta):
        return 0.0
    # Sign convention: δ > 0 when the true frequency sits above bin k.
    return float(np.clip(-delta, -0.75, 0.75))


def parabolic_delta(power: np.ndarray, k: int) -> float:
    if k <= 0 or k >= len(power) - 1:
        return 0.0
    a, b, c = float(power[k - 1]), float(power[k]), float(power[k + 1])
    den = a - 2.0 * b + c
    if abs(den) < 1e-18:
        return 0.0
    return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))


def ehlers_hilbert_iq(series: np.ndarray, period: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ehlers Hilbert transformer (6-bar quadrature, in-phase delayed 3)."""
    n = len(series)
    in_phase = np.zeros(n)
    quad = np.zeros(n)
    x = np.asarray(series, dtype=float)
    p = np.asarray(period, dtype=float)
    for i in range(n):
        if i < 6:
            in_phase[i] = 0.0
            quad[i] = 0.0
            continue
        pr = float(np.clip(p[i] if np.isfinite(p[i]) else 32.0, 12.0, 128.0))
        scale = 0.075 * pr + 0.54
        s0, s2, s4, s6 = x[i], x[i - 2], x[i - 4], x[i - 6]
        if not np.all(np.isfinite([s0, s2, s4, s6, x[i - 3]])):
            in_phase[i] = in_phase[i - 1]
            quad[i] = quad[i - 1]
            continue
        quad[i] = (0.0962 * s0 + 0.5769 * s2 - 0.5769 * s4 - 0.0962 * s6) * scale
        in_phase[i] = x[i - 3]
    return in_phase, quad


def causal_cycle_state(
    price: Series,
    window: int = 192,
    step: int = 2,
    min_p: float = 16.0,
    max_p: float = 64.0,
    default_p: float = 32.0,
) -> DataFrame:
    """
    cycle period (Quinn II) + Hilbert phase.

    Phase convention (Ehlers):
      atan2(Q, I) ≈ 0        → cycle peak (price high of the oscillation)
      atan2(Q, I) ≈ ±π       → cycle trough
    cycle_pos maps phase to [0, 1]: 0 trough, 0.5 peak, 1 trough.
    """
    eps = 1e-12
    price = Series(price).astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    values = price.to_numpy(dtype=float)
    n = len(values)
    L = max(int(window), 48)
    n_fft = next_pow2(L)
    hann = np.hanning(L)
    energy = max(float(np.mean(hann * hann)), eps)
    freqs = np.fft.rfftfreq(n_fft, d=1.0)
    periods = np.full_like(freqs, np.inf)
    periods[1:] = 1.0 / np.maximum(freqs[1:], eps)
    # drop the two lowest-frequency bins to avoid DC lock
    band_idx = np.flatnonzero((periods >= min_p) & (periods <= max_p) & (np.arange(len(periods)) >= 3))
    if len(band_idx) < 3:
        band_idx = np.flatnonzero((periods >= min_p) & (periods <= max_p))

    cycle = np.full(n, default_p)
    quality = np.zeros(n)
    valid = np.zeros(n, dtype=bool)
    phase = np.zeros(n)
    pos = np.full(n, 0.5)
    amp = np.zeros(n)

    prev_p, prev_valid, prev_q = default_p, False, 0.0
    alpha = 2.0 / 7.0

    logv = np.log(np.clip(values, eps, None))
    # causal residual: log price minus EMA so FFT does not sit on drift
    ema = np.empty(n)
    ema_a = 2.0 / (1.0 + L / 3.0)
    prev_e = logv[0]
    for i in range(n):
        prev_e = ema_a * logv[i] + (1.0 - ema_a) * prev_e
        ema[i] = prev_e
    residual = logv - ema

    for i in range(L - 1, n):
        if (i - (L - 1)) % step != 0:
            cycle[i] = prev_p
            valid[i] = prev_valid
            quality[i] = prev_q
            continue
        w = residual[i - L + 1 : i + 1]
        if len(w) != L or not np.all(np.isfinite(w)):
            cycle[i] = prev_p
            continue
        z = robust_z(linear_detrend(w))
        spec_raw = np.fft.rfft(z, n=n_fft)
        y = z * hann
        spec = np.fft.rfft(y, n=n_fft)
        power = (np.abs(spec) ** 2) / (L * energy + eps)
        if n_fft % 2 == 0:
            power[1:-1] *= 2.0
        else:
            power[1:] *= 2.0
        band = power[band_idx]
        total = float(np.sum(band))
        if total <= eps:
            cycle[i] = prev_p
            continue
        local = int(np.argmax(band))
        k_peak = int(band_idx[local])
        peak = float(power[k_peak])
        pr = peak / (float(np.median(band)) + eps)
        dom = peak / (total + eps)
        prob = band / (total + eps)
        ent = -float(np.sum(prob * np.log(prob + eps))) / np.log(max(len(prob), 2))
        # harmonic check: if 2f is stronger-ish, keep fundamental if present
        fund_period = n_fft / max(k_peak, 1)
        half_k = int(round(k_peak / 2.0))
        if 2 <= half_k < len(power) and power[half_k] > 0.45 * peak and periods[half_k] <= max_p:
            if periods[half_k] >= min_p:
                k_peak = half_k
                peak = float(power[k_peak])
                fund_period = n_fft / max(k_peak, 1)

        delta = quinn2_delta(spec_raw, k_peak)
        if not np.isfinite(delta):
            delta = parabolic_delta(power, k_peak)
        k_hat = float(np.clip(k_peak + delta, 1.0, len(spec) - 1.5))
        det = float(np.clip(n_fft / k_hat, min_p, max_p))

        good = pr >= 3.5 and dom >= 0.12 and ent <= 0.88
        # hysteresis
        if prev_valid and pr >= 2.4 and dom >= 0.08:
            good = True
        prev_valid = bool(good)
        if good:
            prev_p = float(np.clip(alpha * det + (1.0 - alpha) * prev_p, min_p, max_p))
        prev_q = float(np.clip(0.5 * (pr / 8.0) + 0.5 * dom, 0.0, 2.0))
        cycle[i] = prev_p
        valid[i] = prev_valid
        quality[i] = prev_q

        # right-edge phase of the dominant bin (causal, window ends at i)
        bin_phase = float(np.angle(spec[k_peak]))
        edge = bin_phase + 2.0 * np.pi * k_hat * (L - 1) / n_fft
        phase[i] = float(np.arctan2(np.sin(edge), np.cos(edge)))
        amp[i] = float(np.abs(spec[k_peak]))

    # fill phase on skipped bars by rotating at instantaneous frequency
    for i in range(L, n):
        if (i - (L - 1)) % step == 0:
            continue
        dphi = 2.0 * np.pi / max(cycle[i], min_p)
        ph = phase[i - 1] + dphi
        phase[i] = float(np.arctan2(np.sin(ph), np.cos(ph)))
        amp[i] = amp[i - 1]

    # Ehlers Hilbert I/Q on the residual, period-adaptive — more stable phase for entries
    hp = residual.copy()
    i_comp, q_comp = ehlers_hilbert_iq(hp, cycle)
    # blend: prefer Hilbert IQ phase when amplitude is alive
    hilbert_phase = np.arctan2(q_comp, i_comp)
    hilbert_amp = np.sqrt(i_comp * i_comp + q_comp * q_comp)
    use_h = hilbert_amp > np.nanmedian(hilbert_amp[hilbert_amp > 0]) * 0.25 if np.any(hilbert_amp > 0) else False
    if isinstance(use_h, np.ndarray):
        mixed = np.where(use_h, hilbert_phase, phase)
    else:
        mixed = hilbert_phase
    # smooth phase via unwrapped EMA then wrap
    unwrapped = np.unwrap(mixed)
    sm = unwrapped.copy()
    a = 2.0 / 5.0
    for i in range(1, n):
        if np.isfinite(unwrapped[i]) and np.isfinite(sm[i - 1]):
            sm[i] = a * unwrapped[i] + (1.0 - a) * sm[i - 1]
    mixed = np.arctan2(np.sin(sm), np.cos(sm))

    # pos: 0 at trough (±π), 0.5 at peak (0)
    pos = (np.pi - np.abs(mixed)) / np.pi  # 0 at ±π (trough), 1 at 0 (peak)
    trough = (np.abs(mixed) >= 2.55) & valid
    peak = (np.abs(mixed) <= 0.40) & valid
    peak_s = pd.Series(peak)
    trough_s = pd.Series(trough)
    peak_cross = (peak_s & ~peak_s.shift(1, fill_value=False)).to_numpy()
    trough_cross = (trough_s & ~trough_s.shift(1, fill_value=False)).to_numpy()

    return DataFrame(
        {
            "cycle_period": cycle,
            "fft_cycle_valid": valid,
            "cycle_quality": quality,
            "cycle_phase": mixed,
            "cycle_pos": pos,
            "cycle_peak": peak,
            "cycle_trough": trough,
            "cycle_peak_cross": peak_cross,
            "cycle_trough_cross": trough_cross,
            "cycle_i": i_comp,
            "cycle_q": q_comp,
        },
        index=price.index,
    )


def adaptive_ema(values: Series, periods: Series, min_period: float = 3.0) -> Series:
    x = Series(values).astype(float).replace([np.inf, -np.inf], np.nan).to_numpy()
    p = (
        Series(periods, index=values.index)
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .fillna(min_period)
        .clip(lower=min_period)
        .to_numpy()
    )
    out = np.full(len(x), np.nan)
    prev = np.nan
    for i, value in enumerate(x):
        if not np.isfinite(value):
            out[i] = prev
            continue
        alpha = float(np.clip(2.0 / (p[i] + 1.0), 0.0, 1.0))
        prev = value if not np.isfinite(prev) else alpha * value + (1.0 - alpha) * prev
        out[i] = prev
    return Series(out, index=values.index)
