
import logging
from typing import Any, Optional

import numpy as np
from pandas import DataFrame

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)


class _BayesianGibbsLinear:
    """
    Bayesian Linear Regression fitted with Gibbs-sampled MCMC.

    Model (per target):
        y | X, beta, sigma^2 ~ N(X_beta @ beta, sigma^2 I)
        beta | sigma^2       ~ N(m0, sigma^2 V0)
        sigma^2              ~ InvGamma(a0, b0)

    We run a Gibbs sampler over (beta, sigma^2). Gibbs is a (Markov Chain) Monte Carlo
    method; after burn-in and thinning we return posterior means for prediction.

    Supports multi-output targets by fitting one independent model per column of y.
    """

    def __init__(
        self,
        n_iter: int = 2000,
        burn_in: int = 500,
        thin: int = 2,
        a0: float = 2.0,
        b0: float = 2.0,
        tau2: float = 10.0,   # V0 = tau2 * I  (weakly-informative prior)
        random_state: Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        self.n_iter = int(n_iter)
        self.burn_in = int(burn_in)
        self.thin = int(thin)
        self.a0 = float(a0)
        self.b0 = float(b0)
        self.tau2 = float(tau2)
        self.random_state = random_state
        self.verbose = int(verbose)

        self._coef_mean: Optional[np.ndarray] = None  # shape (n_outputs, n_features)
        self._intercept_mean: Optional[np.ndarray] = None  # shape (n_outputs,)
        self._fitted = False

        self._rng = np.random.default_rng(random_state)

    def _design(self, X: np.ndarray) -> np.ndarray:
        # Add intercept
        ones = np.ones((X.shape[0], 1), dtype=float)
        return np.hstack([ones, X])

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "_BayesianGibbsLinear":
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n, p = X.shape
        _, k = y.shape

        Xd = self._design(X)  # (n, p+1)

        # Optional weights -> use sqrt(weights) to weight X and y
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=float).reshape(-1)
            if w.shape[0] != n:
                raise ValueError("sample_weight length must equal n_samples")
            sw = np.sqrt(w).reshape(-1, 1)
            Xd = Xd * sw
            y = y * sw

        p1 = p + 1
        V0_inv = np.eye(p1) / self.tau2
        m0 = np.zeros(p1)

        coef_means = np.zeros((k, p))       # without intercept
        intercept_means = np.zeros(k)

        # Precompute X'X once (after weighting)
        XtX = Xd.T @ Xd

        kept = []
        for target_idx in range(k):
            yt = y[:, target_idx]

            # Initialize via OLS (ridge stabilized)
            ridge = 1e-6
            beta = np.linalg.solve(XtX + ridge * np.eye(p1), Xd.T @ yt)
            resid = yt - Xd @ beta
            sigma2 = float(np.maximum(1e-12, (resid @ resid) / max(n - p1, 1)))

            beta_samples = []
            # Gibbs sampler
            for it in range(self.n_iter):
                # Posterior for beta | sigma2, y
                Vn_inv = XtX + V0_inv
                Vn = np.linalg.inv(Vn_inv)
                mn = Vn @ (Xd.T @ yt + V0_inv @ m0)

                # Draw beta ~ N(mn, sigma2 * Vn)
                try:
                    beta = self._rng.multivariate_normal(mean=mn, cov=sigma2 * Vn)
                except np.linalg.LinAlgError:
                    # Numerical fallback: add small jitter
                    Vn_jitter = Vn + 1e-10 * np.eye(p1)
                    beta = self._rng.multivariate_normal(mean=mn, cov=sigma2 * Vn_jitter)

                # Posterior for sigma2 | beta, y  ~ InvGamma(an, bn)
                resid = yt - Xd @ beta
                an = self.a0 + 0.5 * n
                bn = self.b0 + 0.5 * (resid @ resid + (beta - m0) @ V0_inv @ (beta - m0))

                # Sample sigma^2 from InvGamma(an, bn) via 1/Gamma
                g = self._rng.gamma(shape=an, scale=1.0 / bn)
                sigma2 = 1.0 / g

                # Store post-burn-in thinned draws
                if it >= self.burn_in and ((it - self.burn_in) % self.thin == 0):
                    beta_samples.append(beta)

            if len(beta_samples) == 0:
                raise RuntimeError("No MCMC samples kept. Adjust n_iter/burn_in/thin.")

            beta_samples = np.asarray(beta_samples)  # (n_keep, p1)
            beta_mean = beta_samples.mean(axis=0)

            intercept_means[target_idx] = beta_mean[0]
            coef_means[target_idx, :] = beta_mean[1:]

            if self.verbose:
                logger.info(
                    f"[MCMC target {target_idx}] kept={beta_samples.shape[0]} "
                    f"intercept={intercept_means[target_idx]:.6f} "
                    f"|beta|={np.linalg.norm(coef_means[target_idx]):.6f}"
                )

        self._coef_mean = coef_means         # (k, p)
        self._intercept_mean = intercept_means  # (k,)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        if self._coef_mean is None or self._intercept_mean is None:
            raise RuntimeError("Internal coefficients missing.")

        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        n, p = X.shape
        k, p_check = self._coef_mean.shape
        if p != p_check:
            raise ValueError(f"X has {p} features but model expects {p_check}.")

        # y_hat = X @ beta_mean + intercept
        yhat = (X @ self._coef_mean.T) + self._intercept_mean.reshape(1, -1)
        return yhat


class MCMCLinearRegressor(BaseRegressionModel):
    """
    FreqAI prediction model using Bayesian Linear Regression trained via
    Gibbs-sampled MCMC (Markov Chain Monte Carlo).

    ✔ Multi-output regression supported.
    ✔ sample_weight supported (e.g., recency weighting from FreqAI).
    ✔ Works with the standard FreqAI feature/label pipelines.

    Configure via `model_training_parameters` in your FreqAI config, e.g.:

    "freqai": {
      "identifier": "mcmc_linreg_v1",
      "prediction_model": "freqtrade.freqai.prediction_models.MCMCLinearRegressor",
      "model_training_parameters": {
        "n_iter": 2000,
        "burn_in": 500,
        "thin": 2,
        "a0": 2.0,
        "b0": 2.0,
        "tau2": 10.0,
        "random_state": 42,
        "verbose": 0
      }
    }

    Notes:
    - Gibbs sampling is a special case of Markov Chain Monte Carlo (MCMC). The "Markov
      chain" aspect comes from the fact that each draw depends on the previous draw.
    - Keep `n_iter` reasonably sized for live; increase it for backtesting/accuracy.
    """

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        # Extract arrays from DataKitchen (already transformed by pipelines in BaseRegressionModel)
        X_train = np.asarray(data_dictionary["train_features"], dtype=float)
        y_train = np.asarray(data_dictionary["train_labels"], dtype=float)

        # Some FreqAI setups may include weights
        train_w = data_dictionary.get("train_weights", None)
        if train_w is not None:
            train_w = np.asarray(train_w).reshape(-1)
            if train_w.shape[0] != X_train.shape[0]:
                logger.warning("train_weights length mismatch. Ignoring weights.")
                train_w = None

        # Pull user parameters (with sensible defaults)
        p = self.model_training_parameters or {}
        n_iter = int(p.get("n_iter", 2000))
        burn_in = int(p.get("burn_in", 500))
        thin = int(p.get("thin", 2))
        a0 = float(p.get("a0", 2.0))
        b0 = float(p.get("b0", 2.0))
        tau2 = float(p.get("tau2", 10.0))
        random_state = p.get("random_state", None)
        verbose = int(p.get("verbose", 0))

        # Fit Bayesian MCMC linear model
        mcmc = _BayesianGibbsLinear(
            n_iter=n_iter,
            burn_in=burn_in,
            thin=thin,
            a0=a0,
            b0=b0,
            tau2=tau2,
            random_state=random_state,
            verbose=verbose,
        ).fit(X_train, y_train, sample_weight=train_w)

        # Optional: quick validation log
        if (
            dk.data_dictionary.get("test_features") is not None
            and dk.data_dictionary.get("test_features").shape[0] > 0
        ):
            X_test = np.asarray(dk.data_dictionary["test_features"], dtype=float)
            y_test = np.asarray(dk.data_dictionary["test_labels"], dtype=float)
            y_pred = mcmc.predict(X_test)
            mae = float(np.mean(np.abs(y_test - y_pred)))
            logger.info(f"[MCMCLinearRegressor] Validation MAE: {mae:.6f}")

        return mcmc

    # `predict` uses the BaseRegressionModel implementation which calls our model's .predict().
