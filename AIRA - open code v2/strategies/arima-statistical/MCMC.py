import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
import talib.abstract as ta
from freqtrade.strategy import IStrategy


class MCMC_Markov_V1(IStrategy):
    """
    2-state Markov Regime strategy via MCMC:
    - Hidden states: Bear (0), Bull (1)
    - Observations: log-returns, Normal(mu_k, sigma2_k)
    - Priors: Dirichlet for transitions, Normal-Inverse-Gamma for emissions
    - Inference: Gibbs sampling + FFBS (Forward-Filter Backward-Sample)

    Señales:
    - Long cuando P(Bull) cruza > upper (con histéresis)
    - Short cuando P(Bull) cae < lower (con histéresis)
    - Salidas al recuperar zona media
    """

    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = True
    use_exit_signal = True

    # Gestión monetaria base (ajusta a tu gusto)
    minimal_roi = {}
    stoploss = -0.20
    trailing_stop = False

    # --- Hyperparámetros MCMC / Modelo ---
    mcmc_window = 800      # nº de velas usadas (últimas)
    mcmc_iters = 150       # iteraciones totales
    mcmc_burnin = 50       # burn-in
    # Priors Normal-Inverse-Gamma: mu | sigma2 ~ N(mu0, sigma2/kappa0) ; sigma2 ~ InvGamma(a0, b0)
    mu0 = 0.0
    kappa0 = 1.0
    a0 = 2.0
    b0 = 1e-6
    # Prior Dirichlet para filas de A (suaviza permanencia en estado)
    dirichlet_alpha = np.array([2.0, 2.0])

    # Señales
    prob_upper = 0.70
    prob_lower = 0.30
    mid_band = 0.50
    mid_band_hyst = 0.10  # 0.40 / 0.60

    @property
    def plot_config(self):
        return {
            "main_plot": {},
            "subplots": {
                "MCMC Bull Prob": {
                    "bull_prob": {"color": "blue"},
                    "threshold_upper": {"color": "red", "type": "line"},
                    "threshold_lower": {"color": "red", "type": "line"},
                },
                "State Means (posterior)": {
                    "mu_bear": {"color": "red"},
                    "mu_bull": {"color": "green"},
                }
            },
        }

    # ------------------------- Utils Prob/Stats -------------------------

    @staticmethod
    def _logsumexp(a):
        m = np.max(a)
        return m + np.log(np.sum(np.exp(a - m)))

    @staticmethod
    def _normal_logpdf(x, mu, sigma2):
        # log N(x | mu, sigma2)
        return -0.5 * (np.log(2.0 * np.pi * sigma2) + ((x - mu) ** 2) / sigma2)

    @staticmethod
    def _sample_inverse_gamma(shape_a, rate_b):
        # If X ~ InvGamma(a, b) with pdf ∝ b^a x^{-(a+1)} e^{-b/x}, then 1/X ~ Gamma(a, b).
        # numpy gamma uses shape a and scale θ = 1/rate
        y = np.random.gamma(shape=shape_a, scale=1.0 / rate_b)
        return 1.0 / y

    @staticmethod
    def _sample_categorical(probs):
        # probs must sum to 1 (numerically)
        r = np.random.rand()
        c = np.cumsum(probs)
        return int(np.searchsorted(c, r, side="right"))

    # ------------------------- FFBS -------------------------

    def _ffbs_sample_states(self, r, A, mu, sigma2):
        """
        Forward-Filter Backward-Sample of hidden states.
        r: array of returns length T
        A: 2x2 transition matrix
        mu: [mu_bear, mu_bull]
        sigma2: [var_bear, var_bull]
        Returns: sampled path s (length T) with values in {0,1}
        """
        T = len(r)
        logA = np.log(A + 1e-32)
        # initial distribution ~ uniform
        log_pi0 = np.log(np.array([0.5, 0.5]))

        # Emission log-likelihoods L[t,k]
        L = np.vstack([
            self._normal_logpdf(r, mu[0], sigma2[0]),
            self._normal_logpdf(r, mu[1], sigma2[1])
        ]).T  # shape (T,2)

        # Forward messages (log-space)
        log_alpha = np.zeros((T, 2))
        log_alpha[0, :] = log_pi0 + L[0, :]
        for t in range(1, T):
            for k in range(2):
                log_alpha[t, k] = L[t, k] + self._logsumexp(log_alpha[t-1, :] + logA[:, k])

        # Backward sampling
        s = np.zeros(T, dtype=int)
        # sample s_T
        la_T = log_alpha[-1, :] - self._logsumexp(log_alpha[-1, :])
        p_T = np.exp(la_T)
        s[-1] = self._sample_categorical(p_T)
        # sample s_{t} given s_{t+1}
        for t in range(T - 2, -1, -1):
            logp = log_alpha[t, :] + logA[:, s[t+1]]
            logp = logp - self._logsumexp(logp)
            p = np.exp(logp)
            s[t] = self._sample_categorical(p)
        return s

    # ------------------------- MCMC Core -------------------------

    def _mcmc_markov(self, r, iters, burnin):
        """
        Gibbs sampler para:
        - Transiciones A (Dirichlet)
        - Emisiones (mu_k, sigma2_k) con Normal-Inverse-Gamma
        - Estados S via FFBS

        r: ndarray (T,)
        Returns:
          bull_prob (T,)   posterior P(state==Bull) suavizada a partir de muestras (sin lookahead -> se desplazará 1 en señales)
          mu_bear, mu_bull (scalars) últimos samples medios (promedio post-burnin)
        """
        T = len(r)
        # init states via k-means-like split sign/median
        median_r = np.median(r)
        s = (r > median_r).astype(int)

        # initial transition counts
        def estimate_A_from_s(s_):
            counts = np.zeros((2, 2), dtype=float)
            for t in range(T - 1):
                counts[s_[t], s_[t+1]] += 1.0
            # Dirichlet posterior mean
            A = np.zeros((2, 2), dtype=float)
            for i in range(2):
                A[i, :] = (self.dirichlet_alpha + counts[i, :]) / (np.sum(self.dirichlet_alpha + counts[i, :]))
            return A

        A = estimate_A_from_s(s)

        # initial emission params by sample stats
        def sample_emissions(s_, mu0, kappa0, a0, b0):
            mu = np.zeros(2)
            sigma2 = np.zeros(2)
            for k in range(2):
                idx = (s_ == k)
                n = int(idx.sum())
                if n > 0:
                    xk = r[idx]
                    xbar = xk.mean()
                    sk2 = np.sum((xk - xbar) ** 2)
                    kappa_n = kappa0 + n
                    mu_n = (kappa0 * mu0 + n * xbar) / kappa_n
                    a_n = a0 + n / 2.0
                    b_n = b0 + 0.5 * sk2 + 0.5 * (kappa0 * n / kappa_n) * (xbar - mu0) ** 2
                    sigma2[k] = self._sample_inverse_gamma(a_n, b_n)
                    mu[k] = np.random.normal(mu_n, np.sqrt(sigma2[k] / kappa_n))
                else:
                    # sin datos: sample from prior amplio
                    sigma2[k] = self._sample_inverse_gamma(a0, b0 + 1e-6)
                    mu[k] = np.random.normal(mu0, np.sqrt(sigma2[k] / max(kappa0, 1e-9)))
            return mu, sigma2

        mu, sigma2 = sample_emissions(s, self.mu0, self.kappa0, self.a0, self.b0)

        # Contadores para prob posterior del estado Bull por vela
        keep = iters - burnin
        bull_counts = np.zeros(T, dtype=float)
        mu_bear_acc, mu_bull_acc = 0.0, 0.0

        for it in range(iters):
            # 1) Sample path S via FFBS
            s = self._ffbs_sample_states(r, A, mu, sigma2)

            # 2) Sample A | S (Dirichlet por fila)
            counts = np.zeros((2, 2), dtype=float)
            for t in range(T - 1):
                counts[s[t], s[t+1]] += 1.0
            A = np.zeros((2, 2), dtype=float)
            for i in range(2):
                A[i, :] = np.random.dirichlet(self.dirichlet_alpha + counts[i, :])

            # 3) Sample emissions (mu_k, sigma2_k) | S, r
            mu, sigma2 = sample_emissions(s, self.mu0, self.kappa0, self.a0, self.b0)

            # 4) Identificación de etiquetas: forzar que estado 1 sea Bull (mu_1 > mu_0)
            if mu[1] < mu[0]:
                # swap labels
                mu = mu[::-1]
                sigma2 = sigma2[::-1]
                # permutar estados y A
                s = 1 - s
                A = A[[1, 0]][:, [1, 0]]

            # 5) Acumuladores post-burnin
            if it >= burnin:
                bull_counts += (s == 1).astype(float)
                mu_bear_acc += mu[0]
                mu_bull_acc += mu[1]

        bull_prob = bull_counts / max(keep, 1)
        mu_bear_mean = mu_bear_acc / max(keep, 1)
        mu_bull_mean = mu_bull_acc / max(keep, 1)

        return bull_prob, mu_bear_mean, mu_bull_mean

    # ------------------------- Freqtrade hooks -------------------------

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        # log-returns
        df["logret"] = np.log(df["close"]).diff()
        # trabajamos sobre la ventana final (quita NaN inicial)
        r = df["logret"].dropna().values
        if len(r) < 50:
            # muy poco historial: relleno seguro
            df["bull_prob"] = 0.5
            df["mu_bear"] = 0.0
            df["mu_bull"] = 0.0
        else:
            window = min(self.mcmc_window, len(r))
            r_win = r[-window:]
            bull_prob_win, mu_bear_mean, mu_bull_mean = self._mcmc_markov(
                r_win, iters=self.mcmc_iters, burnin=self.mcmc_burnin
            )

            # reubicar al índice del dataframe
            bull_prob_full = np.full(len(df), np.nan)
            # bull_prob_win corresponde a r_win, que empieza en el índice de logret.dropna()[-window:]
            valid_idx = df["logret"].dropna().index[-window:]
            bull_prob_full[valid_idx] = bull_prob_win

            # Para evitar lookahead (la MCMC suaviza con toda la serie),
            # desplazamos 1 vela antes de usar en señales:
            df["bull_prob"] = pd.Series(bull_prob_full, index=df.index).shift(1)

            # Mu posterior promedio (para plot informativo)
            df["mu_bear"] = mu_bear_mean
            df["mu_bull"] = mu_bull_mean

        # Umbrales para plot
        df["threshold_upper"] = float(self.prob_upper)
        df["threshold_lower"] = float(self.prob_lower)

        # Limpieza
        df.fillna(method="ffill", inplace=True)
        df.fillna(0.5, inplace=True)  # prob neutra si aún no hay datos

        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        upper = self.prob_upper
        lower = self.prob_lower

        # Entradas por cruces con histéresis (usamos shift en condición para evitar lookahead)
        # Long: bull_prob cruza por encima de upper
        df.loc[
            (df["bull_prob"] > upper) & (df["bull_prob"].shift(1) <= upper),
            "enter_long"
        ] = 1

        # Short: bull_prob cae por debajo de lower
        df.loc[
            (df["bull_prob"] < lower) & (df["bull_prob"].shift(1) >= lower),
            "enter_short"
        ] = 1

        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        mid_low = self.mid_band - self.mid_band_hyst   # 0.40
        mid_high = self.mid_band + self.mid_band_hyst  # 0.60

        # Cerrar largos si la prob de bull pierde zona media baja
        df.loc[
            (df["bull_prob"] < mid_low) & (df["bull_prob"].shift(1) >= mid_low),
            "exit_long"
        ] = 1

        # Cerrar cortos si la prob de bull recupera zona media alta
        df.loc[
            (df["bull_prob"] > mid_high) & (df["bull_prob"].shift(1) <= mid_high),
            "exit_short"
        ] = 1

        return df

    def leverage(
        self,
        pair: str,
        current_time: "datetime",
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        side: str,
        **kwargs,
    ) -> float:
        return 1.0
