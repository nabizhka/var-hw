"""MF731 HW 5 (05_var_est): one-day 95% VaR of a $1M cap-weighted MSFT/AAPL/GOOGL portfolio.

Risk factors are log prices, risk-factor changes are daily log returns x.
  full:  l(x)      = -theta' (e^x - 1)
  lin:   l_lin(x)  = -theta' x
  quad:  l_quad(x) = -theta' (x + x^2 / 2)
theta = V_t w is the dollar position vector.
"""
import numpy as np
from scipy.stats import norm

from common import load_close, log_returns, var_emp

TICKERS = ["MSFT", "AAPL", "GOOGL"]
CAPS = np.array([3.109, 4.536, 4.640])        # $ trillions on 05/22/2026, from the sheet
V0, ALPHA, LAM, THETA_EWMA, N_MC, SEED = 1_000_000.0, 0.95, 0.97, 0.97, 100_000, 731


def l_full(x, theta):
    return -(np.expm1(x) @ theta)


def l_lin(x, theta):
    return -(x @ theta)


def l_quad(x, theta):
    return -((x + 0.5 * x**2) @ theta)


def ewma(X, lam, th):
    """mu_{t+D} = lam mu_t + (1-lam) X_t,  Sigma_{t+D} = th Sigma_t + (1-th)(X_t-mu_t)(X_t-mu_t)'.

    Starts from mu_0 = 0, Sigma_0 = 0 and runs through every return, so the output
    is the estimate as of the last date, to be used for the next trading day.
    """
    d = X.shape[1]
    mu, S = np.zeros(d), np.zeros((d, d))
    for x in X:
        dev = x - mu                       # uses mu_t, before the update
        mu = lam * mu + (1 - lam) * x
        S = th * S + (1 - th) * np.outer(dev, dev)
    return mu, S


def run(price_file="stocks_close.csv"):
    P = load_close(price_file)[TICKERS]
    X = log_returns(P)
    w = CAPS / CAPS.sum()
    theta = V0 * w
    x = X.values

    out = {"prices": P, "returns": X, "w": w, "theta": theta,
           "n_prices": len(P), "n_returns": len(X)}

    # 3. historical (empirical) VaR: the n historical returns are the scenarios
    out["emp"] = {k: var_emp(f(x, theta), ALPHA)
                  for k, f in [("full", l_full), ("lin", l_lin), ("quad", l_quad)]}

    # 4. EWMA mean and covariance for 05/25/2026
    mu, S = ewma(x, LAM, THETA_EWMA)
    out["mu"], out["Sigma"] = mu, S
    # sensitivity to the initial pair: start from the sample moments instead of 0
    out["ewma_init_gap"] = _init_gap(x)

    # 5. normal model. Linear exactly; full and quadratic by MC.
    m, s = -theta @ mu, np.sqrt(theta @ S @ theta)
    out["norm_lin_exact"] = m + s * norm.ppf(ALPHA)
    rng = np.random.default_rng(SEED)
    Z = rng.multivariate_normal(mu, S, size=N_MC)
    out["norm"] = {"full": var_emp(l_full(Z, theta), ALPHA),
                   "lin_mc": var_emp(l_lin(Z, theta), ALPHA),
                   "quad": var_emp(l_quad(Z, theta), ALPHA),
                   "lin": out["norm_lin_exact"]}
    # MC standard error of the full-loss VaR, by 50 independent replications
    reps = [var_emp(l_full(np.random.default_rng(SEED + 1 + i).multivariate_normal(mu, S, size=N_MC), theta), ALPHA)
            for i in range(50)]
    out["norm_full_se"] = np.std(reps, ddof=1)
    out["port_vol_ewma"] = s / V0
    out["port_vol_sample"] = np.sqrt(w @ np.cov(x.T) @ w)
    out["port_vol_recent"] = np.sqrt(w @ np.cov(x[-60:].T) @ w)
    out["port_ret"] = x @ w
    return out


def _init_gap(x):
    mu0, S0 = x.mean(0), np.cov(x.T)
    mu_a, S_a = ewma(x, LAM, THETA_EWMA)
    mu, S = mu0.copy(), S0.copy()
    for xi in x:
        dev = xi - mu
        mu = LAM * mu + (1 - LAM) * xi
        S = THETA_EWMA * S + (1 - THETA_EWMA) * np.outer(dev, dev)
    return np.max(np.abs(S - S_a)) / np.max(np.abs(S_a))
