# pragma pylint: disable=missing-docstring, invalid-name, too-many-locals, too-many-instance-attributes
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import talib
import logging

from typing import Dict, Optional, Tuple, List

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter

logger = logging.getLogger("HMMRegimeStrategy")


# =============================================================================
# 1) HMM Gaussian multivariat
# =============================================================================

class HMMConfig:
    def __init__(
        self,
        n_states: int = 6,
        max_iter: int = 40,
        n_restarts: int = 5,
        tol: float = 1e-4,
        seed: int = 7,
        trans_alpha: float = 2.0,
        init_alpha: float = 2.0,
        var_floor: float = 1e-5,
        eps: float = 1e-12,
        train_window: int = 5500,
    ):
        self.n_states = n_states
        self.max_iter = max_iter
        self.n_restarts = n_restarts
        self.tol = tol
        self.seed = seed

        # Priors Dirichlet pentru tranzitii si distributia initiala
        self.trans_alpha = trans_alpha
        self.init_alpha = init_alpha

        # Regularizare emisii
        self.var_floor = var_floor
        self.eps = eps

        # Protocol train
        self.train_window = train_window


class GaussianHMM:

    def __init__(self, cfg: HMMConfig):
        self.cfg = cfg
        self.K = cfg.n_states

        self.pi: Optional[np.ndarray] = None     # (K,)
        self.A: Optional[np.ndarray] = None      # (K,K)
        self.mu: Optional[np.ndarray] = None     # (K,D)
        self.var: Optional[np.ndarray] = None    # (K,D)

    def _logpdf_diag(self, X: np.ndarray) -> np.ndarray:

        if self.mu is None or self.var is None:
            raise RuntimeError("Model emissions not initialized.")
        X = np.asarray(X, dtype=float)
        T, D = X.shape
        K = self.K

        var = np.maximum(self.var, self.cfg.var_floor)
        log_det = np.sum(np.log(2.0 * np.pi * var), axis=1)  # (K,)
        out = np.empty((T, K), dtype=float)

        for k in range(K):
            diff = X - self.mu[k]
            quad = np.sum((diff * diff) / var[k], axis=1)
            out[:, k] = -0.5 * (log_det[k] + quad)

        return out

    def _forward_backward(self, logB: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:

        if self.pi is None or self.A is None:
            raise RuntimeError("Model transitions not initialized.")
        T, K = logB.shape
        eps = self.cfg.eps

        # Stabilizare: B = exp logB - rowmax
        rowmax = logB.max(axis=1)
        B = np.exp(logB - rowmax[:, None])

        alpha = np.zeros((T, K), dtype=float)
        c = np.zeros(T, dtype=float)

        alpha_raw0 = self.pi * B[0]
        c[0] = alpha_raw0.sum() + eps
        alpha[0] = alpha_raw0 / c[0]

        for t in range(1, T):
            alpha_raw = (alpha[t - 1] @ self.A) * B[t]
            c[t] = alpha_raw.sum() + eps
            alpha[t] = alpha_raw / c[t]

        loglik = float(np.sum(np.log(c + eps)) + np.sum(rowmax))

        beta = np.zeros((T, K), dtype=float)
        beta[T - 1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = self.A @ (B[t + 1] * beta[t + 1])
            beta[t] /= (c[t + 1] + eps)


        gamma = alpha * beta
        gamma /= (gamma.sum(axis=1, keepdims=True) + eps)

        return alpha, beta, gamma, loglik

    def _expected_sum_xi(self, B: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:

        if self.A is None:
            raise RuntimeError("A not initialized.")
        T, K = alpha.shape
        eps = self.cfg.eps
        sum_xi = np.zeros((K, K), dtype=float)

        for t in range(T - 1):

            num = (alpha[t][:, None] * self.A) * (B[t + 1] * beta[t + 1])[None, :]
            den = num.sum() + eps
            sum_xi += num / den

        return sum_xi

    def _init_params(self, X: np.ndarray) -> None:
        # Use a local RNG seeded from config so different restarts vary
        rng = np.random.RandomState(self.cfg.seed if self.cfg.seed is not None else 0)
        X = np.asarray(X, dtype=float)
        T, D = X.shape
        K = self.K
        eps = self.cfg.eps

        self.pi = np.ones(K, dtype=float) / K

        A = np.full((K, K), 1.0, dtype=float)
        np.fill_diagonal(A, 8.0)  # more persistence
        A /= A.sum(axis=1, keepdims=True)
        self.A = A


        mu = np.zeros((K, D), dtype=float)
        var = np.zeros((K, D), dtype=float)

        if D >= 2 and T >= K:
            r = X[:, 0]
            v = X[:, 1]

            r_q1, r_q2 = np.nanquantile(r, [1/3, 2/3])
            v_q = np.nanquantile(v, [0.5])[0]

            dir_cls = np.where(r < r_q1, 0, np.where(r < r_q2, 1, 2))  # 0 bear,1 side,2 bull
            vol_cls = (v >= v_q).astype(int)  # 0 low, 1 high

            mapping = {
                (2, 0): 0,  # bull low
                (2, 1): 1,  # bull high
                (0, 0): 2,  # bear low
                (0, 1): 3,  # bear high
                (1, 0): 4,  # side low
                (1, 1): 5,  # side high
            }
            labels = np.array([mapping[(dir_cls[i], vol_cls[i])] for i in range(T)], dtype=int)

            for k in range(K):
                idx = (labels == k)
                if idx.sum() < 10:
                    pick = rng.choice(T, size=min(50, T), replace=False)
                    mu[k] = np.nanmean(X[pick], axis=0)
                    var[k] = np.nanvar(X[pick], axis=0) + self.cfg.var_floor
                else:
                    mu[k] = np.nanmean(X[idx], axis=0)
                    var[k] = np.nanvar(X[idx], axis=0) + self.cfg.var_floor
        else:
            mu = rng.normal(size=(K, D))
            var = np.ones((K, D), dtype=float) * 0.5

        # Add a small, seed-controlled jitter so different seeds yield different inits
        jitter_scale = 1e-4
        mu = mu + rng.normal(scale=jitter_scale, size=mu.shape)

        self.mu = mu
        self.var = np.maximum(var, self.cfg.var_floor + eps)

    def fit(self, X: np.ndarray) -> Dict[str, float]:

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be (T,D).")
        T, D = X.shape
        if T < max(100, self.K * 20):
            raise ValueError("Too few observations for HMM fit.")

        self._init_params(X)
        prev_ll = -np.inf

        for it in range(self.cfg.max_iter):
            logB = self._logpdf_diag(X)
            alpha, beta, gamma, ll = self._forward_backward(logB)

            rowmax = logB.max(axis=1)
            B = np.exp(logB - rowmax[:, None])

            sum_xi = self._expected_sum_xi(B, alpha, beta)

            eps = self.cfg.eps
            K = self.K

            pi = gamma[0] + (self.cfg.init_alpha - 1.0)
            pi = np.maximum(pi, eps)
            pi /= pi.sum()
            self.pi = pi

            A = sum_xi + (self.cfg.trans_alpha - 1.0)
            A = np.maximum(A, eps)
            A /= A.sum(axis=1, keepdims=True)
            self.A = A

            Nk = gamma.sum(axis=0) + eps  # (K,)
            mu = (gamma.T @ X) / Nk[:, None]
            self.mu = mu

            var = np.zeros((K, D), dtype=float)
            for k in range(K):
                diff = X - mu[k]
                var[k] = (gamma[:, k][:, None] * (diff * diff)).sum(axis=0) / Nk[k]
            self.var = np.maximum(var, self.cfg.var_floor)


            if np.isfinite(prev_ll):
                if abs(ll - prev_ll) < self.cfg.tol * (1.0 + abs(prev_ll)):
                    return {"loglik": float(ll), "iters": float(it + 1)}
            prev_ll = ll

        return {"loglik": float(prev_ll), "iters": float(self.cfg.max_iter)}

    def filter(self, X: np.ndarray) -> np.ndarray:

        if self.pi is None or self.A is None or self.mu is None or self.var is None:
            raise RuntimeError("Model is not fitted.")

        X = np.asarray(X, dtype=float)
        T, D = X.shape
        logB = self._logpdf_diag(X)

        rowmax = logB.max(axis=1)
        B = np.exp(logB - rowmax[:, None])

        alpha = np.zeros((T, self.K), dtype=float)
        eps = self.cfg.eps

        a0 = self.pi * B[0]
        alpha[0] = a0 / (a0.sum() + eps)

        for t in range(1, T):
            at = (alpha[t - 1] @ self.A) * B[t]
            alpha[t] = at / (at.sum() + eps)

        return alpha


def build_hmm_features(df: pd.DataFrame, vol_period: int = 20) -> pd.DataFrame:

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df.get("volume", pd.Series(0, index=df.index)).astype(float)

    # core returns
    logp = np.log(close.replace(0, np.nan))
    r = logp.diff()

    # realized vol (train-only will robust-scale)
    rv = r.rolling(vol_period, min_periods=vol_period).std()

    # ADX (trend strength)
    adx_np = talib.ADX(high.values, low.values, close.values, timeperiod=14)
    adx = pd.Series(adx_np, index=df.index).astype(float)

    # range and ATR (vol proxies)
    rng = (high - low) / close.replace(0, np.nan)
    atr_np = talib.ATR(high.values, low.values, close.values, timeperiod=14)
    atr = pd.Series(atr_np, index=df.index).astype(float)

    # MACD histogram (momentum)
    macd, macdsig, macdhist = talib.MACD(close.values, fastperiod=12, slowperiod=26, signalperiod=9)
    macdh = pd.Series(macdhist, index=df.index).astype(float)

    # EMA ratio (trend short vs long)
    ema_short = talib.EMA(close.values, timeperiod=20)
    ema_long = talib.EMA(close.values, timeperiod=50)
    ema_ratio = pd.Series(ema_short / np.maximum(ema_long, 1e-9), index=df.index).astype(float)

    # OBV (volume-flow) and volume momentum
    try:
        obv = pd.Series(talib.OBV(close.values, vol.values), index=df.index).astype(float)
    except Exception:
        obv = vol.fillna(0.0)
    vol_mom = vol.rolling(vol_period, min_periods=1).std()

    # short-horizon returns (lag features)
    r_3 = logp.diff(3)
    r_5 = logp.diff(5)

    # RSI
    rsi_np = talib.RSI(close.values, timeperiod=14)
    rsi = pd.Series(rsi_np, index=df.index).astype(float)

    out = pd.DataFrame(index=df.index)
    # Keep `r` and `rv` as first two features (mapping relies on this)
    out["r"] = r
    out["rv"] = rv

    # additional features
    out["adx"] = adx
    out["rng"] = rng
    out["atr"] = atr
    out["macdh"] = macdh
    out["ema_ratio"] = ema_ratio
    out["obv"] = obv
    out["vol_mom"] = vol_mom
    out["r_3"] = r_3
    out["r_5"] = r_5
    out["rsi"] = rsi

    return out


def robust_scale_train_only(X_train: np.ndarray, X_all: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:

    med = np.nanmedian(X_train, axis=0)
    mad = np.nanmedian(np.abs(X_train - med), axis=0) + eps
    Xs = (X_all - med) / mad
    return Xs.astype(float), {"median": med, "mad": mad}


def _smoke_test_hmm():

    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)

    N = 2000
    t = np.arange(N)
    price = 100 + np.cumsum(np.random.normal(scale=0.5, size=N))
    df = pd.DataFrame({
        "open": price,
        "high": price * (1 + np.random.uniform(0, 0.001, size=N)),
        "low": price * (1 - np.random.uniform(0, 0.001, size=N)),
        "close": price,
        "volume": np.random.randint(1, 1000, size=N),
    })

    cfg = HMMConfig(n_states=6, max_iter=20, n_restarts=3, seed=1, train_window=500)
    out = hmm_regime_indicators(df, cfg)
    print("Smoke test: hmm_trained count=" , out["hmm_trained"].sum())


if __name__ == "__main__":
    try:
        _smoke_test_hmm()
    except Exception:
        logger.exception("Smoke test failed")


def map_states_to_regime_ids(mu: np.ndarray) -> Tuple[Dict[int, int], Dict[int, str]]:

    K = mu.shape[0]
    if K != 6:
        raise ValueError("This mapping assumes exactly 6 states.")

    r_mu = mu[:, 0]
    v_mu = mu[:, 1] if mu.shape[1] > 1 else np.zeros_like(r_mu)

    order = np.argsort(r_mu)  # ascending by return mean
    bear_states = order[:2]
    side_states = order[2:4]
    bull_states = order[4:6]

    def _pair_to_low_high(states: np.ndarray) -> Tuple[int, int]:
        if v_mu[states[0]] <= v_mu[states[1]]:
            return int(states[0]), int(states[1])  # low, high
        return int(states[1]), int(states[0])

    bear_low, bear_high = _pair_to_low_high(bear_states)
    side_low, side_high = _pair_to_low_high(side_states)
    bull_low, bull_high = _pair_to_low_high(bull_states)

    state_to_id = {
        bull_low: 1,
        bull_high: 2,
        bear_low: 3,
        bear_high: 4,
        side_low: 5,
        side_high: 6,
    }
    id_to_name = {
        1: "BULLISH_LOW_VOL",
        2: "BULLISH_HIGH_VOL",
        3: "BEARISH_LOW_VOL",
        4: "BEARISH_HIGH_VOL",
        5: "SIDEWAYS_LOW_VOL",
        6: "SIDEWAYS_HIGH_VOL",
    }
    state_to_name = {s: id_to_name[rid] for s, rid in state_to_id.items()}

    return state_to_id, state_to_name


def hmm_regime_indicators(df: pd.DataFrame, cfg: HMMConfig) -> pd.DataFrame:
    out = df.copy()
    feat = build_hmm_features(out, vol_period=20)
    valid = feat.dropna()

    if not feat.empty:
        out.loc[feat.index, "adx"] = feat["adx"].astype(float)

    for i in range(cfg.n_states):
        out[f"hmm_p{i+1}"] = np.nan
    out["hmm_entropy"] = np.nan
    out["hmm_confidence"] = np.nan
    out["hmm_regime_id"] = np.nan
    out["hmm_state"] = np.nan
    out["hmm_trained"] = False

    if len(valid) < cfg.train_window:
        return out

    X_all = valid[["r", "rv", "adx", "rng"]].values.astype(float)


    X_train_raw = X_all[: cfg.train_window]
    X_scaled, scaler = robust_scale_train_only(X_train_raw, X_all, eps=cfg.eps)

    logger.info(
        "HMM pipeline: total=%d valid=%d train_window=%d n_restarts=%d",
        len(df),
        len(valid),
        cfg.train_window,
        getattr(cfg, "n_restarts", 1),
    )


    best_model: Optional[GaussianHMM] = None
    best_ll = -np.inf
    best_res = None
    n_restarts = getattr(cfg, "n_restarts", 1)

    for r in range(n_restarts):
        seed_r = (cfg.seed or 0) + r

        local_cfg = HMMConfig(
            n_states=cfg.n_states,
            max_iter=cfg.max_iter,
            n_restarts=getattr(cfg, "n_restarts", 1),
            tol=cfg.tol,
            seed=seed_r,
            trans_alpha=cfg.trans_alpha,
            init_alpha=cfg.init_alpha,
            var_floor=cfg.var_floor,
            eps=cfg.eps,
            train_window=cfg.train_window,
        )
        model_r = GaussianHMM(local_cfg)
        try:
            logger.info("HMM restart %d/%d: seed=%d", r + 1, n_restarts, seed_r)
            res = model_r.fit(X_scaled[: cfg.train_window])
            ll = float(res.get("loglik", -np.inf))
            logger.info("HMM restart %d finished: loglik=%.6f iters=%s", r + 1, ll, res.get("iters"))
            if ll > best_ll:
                best_ll = ll
                best_model = model_r
                best_res = res
        except Exception as e:
            logger.exception("HMM restart %d failed: %s", r + 1, e)

    if best_model is None:
        logger.error("All HMM restarts failed; skipping HMM for this run")
        return out

    logger.info("Selected best HMM restart with loglik=%.6f", best_ll)

    try:
        P_state = best_model.filter(X_scaled)  # (T_valid, K)
        logger.info("HMM filtering done: P_state shape=%s", P_state.shape)
    except Exception as e:
        logger.exception("HMM filtering failed: %s", e)
        return out

    state_to_id, _ = map_states_to_regime_ids(best_model.mu)

    T_valid, K = P_state.shape
    P_regime = np.zeros((T_valid, 6), dtype=float)
    for s in range(6):
        rid = state_to_id[s]  # 1..6
        P_regime[:, rid - 1] = P_state[:, s]

    denom = P_regime.sum(axis=1, keepdims=True)
    denom = np.where(denom <= cfg.eps, 1.0, denom)
    P_regime = P_regime / denom

    ent = -(P_regime * np.log(P_regime + cfg.eps)).sum(axis=1) / np.log(6.0)
    conf = P_regime.max(axis=1)
    regime_id = (P_regime.argmax(axis=1) + 1).astype(int)
    state_id = P_state.argmax(axis=1).astype(int)

    idx = valid.index
    for i in range(6):
        out.loc[idx, f"hmm_p{i+1}"] = P_regime[:, i]
    out.loc[idx, "hmm_entropy"] = ent
    out.loc[idx, "hmm_confidence"] = conf
    out.loc[idx, "hmm_regime_id"] = regime_id
    out.loc[idx, "hmm_state"] = state_id

    out.loc[idx, "hmm_trained"] = True
    out["hmm_bull_strength"] = out["hmm_p1"] + out["hmm_p2"]
    out["hmm_bear_strength"] = out["hmm_p3"] + out["hmm_p4"]
    out["hmm_side_strength"] = out["hmm_p5"] + out["hmm_p6"]

    return out

class HMMRegimeStrategy(IStrategy):


    def version(self) -> str:
        version_str = "v1, Alpha: Regime Detector by DarkReaper"
        print(f"\033[93m{version_str}\033[0m")
        time.sleep(6)
        return version_str
    r"""
    $$$$$$$  |  ______    ______  $$ |   __ $$$$$$$  |  ______    ______    ______    ______    ____
    $$ |  $$ | /      \  /      \ $$ |  /  |$$ |__$$ | /      \ /      \  /      \  /      \  /      \ 
    $$ |  $$ | $$$$$$  |/$$$$$$  |$$ |_/$$/ $$    $$< /$$$$$$  | $$$$$$  |/$$$$$$  |/$$$$$$  |/$$$$$$  |
    $$ |  $$ | /    $$ |$$ |  $$/ $$   $$<  $$$$$$$  |$$    $$ | /    $$ |$$ |  $$ |$$    $$ |$$ |  $$/
    $$ |__$$ |/$$$$$$$ |$$ |      $$$$$$  \ $$ |  $$ |$$$$$$$$/ /$$$$$$$ |$$ |__$$ |$$$$$$$$/ $$ |
    $$    $$/ $$    $$ |$$ |      $$ | $$  |$$ |  $$ |$$       |$$    $$ |$$    $$/ $$       |$$ |
    $$$$$$$/   $$$$$$$/ $$/       $$/   $$/ $$/   $$/  $$$$$$$/  $$$$$$$/ $$$$$$$/   $$$$$$$/ $$/
                                                                          $$$
                                                                          $$$
                                                                          $$$
    """

    timeframe = "15m"
    can_short = True

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count = 1800

    minimal_roi = {
        "0": 0.04,
        "180": 0.02,
        "720": 0.01,
        "1440": 0.00
    }
    stoploss = -0.10

    trailing_stop = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    hmm_conf_min = DecimalParameter(0.35, 0.70, default=0.50, decimals=2, space="buy")
    hmm_entropy_max = DecimalParameter(0.30, 0.85, default=0.60, decimals=2, space="buy")

    adx_min = IntParameter(10, 35, default=18, space="buy")
    ema_fast = IntParameter(20, 80, default=50, space="buy")
    ema_slow = IntParameter(100, 300, default=200, space="buy")

    _hmm_cfg = HMMConfig(
        n_states=6,
        max_iter=80,
        n_restarts=6,
        tol=1e-4,
        seed=7,
        trans_alpha=2.0,
        init_alpha=2.0,
        var_floor=1e-5,
        train_window=1700
    )

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df = dataframe.copy()

        df["ema_fast"] = talib.EMA(df["close"].values, timeperiod=int(self.ema_fast.value))
        df["ema_slow"] = talib.EMA(df["close"].values, timeperiod=int(self.ema_slow.value))
        df["rsi"] = talib.RSI(df["close"].values, timeperiod=14)

        ub, mb, lb = talib.BBANDS(df["close"].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df["bb_upper"] = ub
        df["bb_middle"] = mb
        df["bb_lower"] = lb
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"].replace(0, np.nan)

        df = hmm_regime_indicators(df, self._hmm_cfg)

        return df

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df = dataframe


        clear = (
            (df["hmm_confidence"] >= float(self.hmm_conf_min.value)) &
            (df["hmm_entropy"] <= float(self.hmm_entropy_max.value))
        )

        trend_up = (df["ema_fast"] > df["ema_slow"]) & (df["close"] > df["ema_slow"])
        trend_dn = (df["ema_fast"] < df["ema_slow"]) & (df["close"] < df["ema_slow"])


        adx_ok = df["adx"] > int(self.adx_min.value)

        long_cond = (
            clear &
            df["hmm_regime_id"].isin([1, 2]) &
            (df["hmm_bull_strength"] >= 0.50) &
            trend_up &
            adx_ok
        )

        short_cond = (
            clear &
            df["hmm_regime_id"].isin([3, 4]) &
            (df["hmm_bear_strength"] >= 0.50) &
            trend_dn &
            adx_ok
        )

        range_long = (
            clear &
            (df["hmm_regime_id"] == 5) &
            (df["hmm_side_strength"] >= 0.50) &
            (df["rsi"] < 35) &
            (df["close"] <= df["bb_lower"] * 1.01)
        )

        df.loc[long_cond, "enter_long"] = 1
        df.loc[short_cond, "enter_short"] = 1
        df.loc[range_long, "enter_long"] = 1

        return df

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df = dataframe

        clear_prev = (
            (df["hmm_confidence"].shift(1) >= float(self.hmm_conf_min.value)) &
            (df["hmm_entropy"].shift(1) <= float(self.hmm_entropy_max.value))
        )
        regime_changed = df["hmm_regime_id"] != df["hmm_regime_id"].shift(1)

        exit_long = (
            clear_prev &
            (
                df["hmm_regime_id"].isin([3, 4]) |
                (df["hmm_bear_strength"] > 0.50) |
                (regime_changed & (df["hmm_confidence"] > 0.50))
            )
        )

        exit_short = (
            clear_prev &
            (
                df["hmm_regime_id"].isin([1, 2]) |
                (df["hmm_bull_strength"] > 0.60) |
                (regime_changed & (df["hmm_confidence"] > 0.50))
            )
        )

        df.loc[exit_long, "exit_long"] = 1
        df.loc[exit_short, "exit_short"] = 1

        return df
