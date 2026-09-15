#!/usr/bin/env python3
"""
With Love from DarkReaper

Kalman phase trackers for the cycle residual.

Two models:

1. Rotating-phasor linear KF (2-state I/Q). Frequency comes from Quinn.
   This is a Kalman-smoothed analytic signal — the natural upgrade to Hilbert.

2. EKF PLL (3-state: phase, omega, amplitude). Jointly tracks frequency
   and phase from the residual. No FFT required after init.

Both are strictly causal (filter update uses only y[k]).
"""
from __future__ import annotations

import numpy as np
from numpy.linalg import inv


def wrap_pi(x: float) -> float:
    return float(np.arctan2(np.sin(x), np.cos(x)))


def rotating_phasor_kf(
    y: np.ndarray,
    period: np.ndarray,
    q_rot: float = 3e-3,
    r_meas: float = 0.25,
    min_p: float = 16.0,
    max_p: float = 64.0,
) -> dict[str, np.ndarray]:
    """Linear KF on a phasor rotating at ω = 2π / period.

    State s = [I, Q]. Observation is the residual y ≈ I.
    """
    n = len(y)
    I = np.zeros(n)
    Q = np.zeros(n)
    phase = np.zeros(n)
    amp = np.zeros(n)
    innov = np.zeros(n)
    s = np.zeros(2)
    P = np.eye(2) * 0.25
    H = np.array([[1.0, 0.0]])
    R = np.array([[float(r_meas)]])
    for k in range(n):
        pr = float(period[k]) if np.isfinite(period[k]) else 32.0
        pr = float(np.clip(pr, min_p, max_p))
        w = 2.0 * np.pi / pr
        c, sn = np.cos(w), np.sin(w)
        F = np.array([[c, -sn], [sn, c]])
        s = F @ s
        P = F @ P @ F.T + np.eye(2) * q_rot
        z = float(y[k]) if np.isfinite(y[k]) else 0.0
        yhat = float((H @ s)[0])
        S = H @ P @ H.T + R
        K = (P @ H.T) / float(S[0, 0])
        e = z - yhat
        s = s + (K.flatten() * e)
        P = (np.eye(2) - K @ H) @ P
        I[k], Q[k] = float(s[0]), float(s[1])
        phase[k] = wrap_pi(np.arctan2(s[1], s[0]))
        amp[k] = float(np.hypot(s[0], s[1]))
        innov[k] = e
    return {"I": I, "Q": Q, "phase": phase, "amp": amp, "innov": innov}


def ekf_pll(
    y: np.ndarray,
    period0: float = 32.0,
    q_phase: float = 1e-5,
    q_omega: float = 3e-7,
    q_amp: float = 3e-4,
    r_meas: float = 0.25,
    min_p: float = 16.0,
    max_p: float = 64.0,
) -> dict[str, np.ndarray]:
    """EKF phase-locked loop. State = [theta, omega, amplitude]."""
    n = len(y)
    theta = np.zeros(n)
    omega = np.zeros(n)
    amp = np.zeros(n)
    phase = np.zeros(n)
    innov = np.zeros(n)
    w0 = 2.0 * np.pi / float(np.clip(period0, min_p, max_p))
    x = np.array([0.0, w0, float(np.nanstd(y[: min(200, n)]) or 0.05)])
    P = np.diag([0.5, 1e-4, 0.05])
    Qp = np.diag([q_phase, q_omega, q_amp])
    I3 = np.eye(3)
    w_min = 2.0 * np.pi / max_p
    w_max = 2.0 * np.pi / min_p
    for k in range(n):
        # predict
        F = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        x = np.array([x[0] + x[1], x[1], x[2]])
        P = F @ P @ F.T + Qp
        z = float(y[k]) if np.isfinite(y[k]) else 0.0
        th, w, a = x
        h = a * np.cos(th)
        H = np.array([[-a * np.sin(th), 0.0, np.cos(th)]])
        S = float(np.array(H @ P @ H.T).ravel()[0] + r_meas)
        K = (P @ H.T) / max(S, 1e-12)
        e = z - h
        x = x + K.flatten() * e
        P = (I3 - K @ H) @ P
        x[0] = wrap_pi(x[0])
        x[1] = float(np.clip(x[1], w_min, w_max))
        x[2] = float(np.clip(x[2], 1e-4, 5.0))
        theta[k], omega[k], amp[k] = x
        phase[k] = x[0]
        innov[k] = e
    period = 2.0 * np.pi / np.clip(omega, w_min, w_max)
    return {"phase": phase, "omega": omega, "period": period, "amp": amp, "innov": innov, "theta": theta}


def residual_log_ema(price: np.ndarray, span: float = 64.0) -> np.ndarray:
    x = np.log(np.clip(np.asarray(price, dtype=float), 1e-12, None))
    a = 2.0 / (span + 1.0)
    ema = np.empty_like(x)
    prev = x[0]
    for i, v in enumerate(x):
        prev = a * v + (1.0 - a) * prev
        ema[i] = prev
    return x - ema
