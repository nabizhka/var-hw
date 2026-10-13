"""MF731 HW 5 (05_var_backtest): backtest one-day 95% VaR for a constant $1 S&P 500 position.

Loss operator l(x) = -(e^x - 1). Two VaR methods on the same evaluation days:
  1. empirical: VaR_t = l_(ceil(M alpha)) of the previous M = 1010 losses
  2. normal log returns with EWMA mean / variance:
       VaR_t = 1 - exp(mu_t + sigma_t N^{-1}(1 - alpha))
"""
import numpy as np
from scipy.stats import norm, binom

from common import load_close, log_returns, var_emp

ALPHA, BETA, LAM, THETA_EWMA, M = 0.95, 0.05, 0.97, 0.97, 1010


def run():
    P = load_close("gspc_close.csv")["GSPC"]
    x = log_returns(P).values
    loss = -np.expm1(x)
    n = len(x)
    J = n - M                               # evaluable days: returns M+1, ..., n

    # Method 1: rolling empirical, window = the M returns before the forecast day
    var_emp_ts = np.array([var_emp(loss[k - M:k], ALPHA) for k in range(M, n)])

    # Method 2: EWMA, initialised with the sample moments of the first M returns
    mu, s2 = x[:M].mean(), x[:M].var(ddof=1)
    var_ewma_ts = np.empty(J)
    for j, k in enumerate(range(M, n)):
        var_ewma_ts[j] = 1 - np.exp(mu + np.sqrt(s2) * norm.ppf(1 - ALPHA))   # forecast for day k
        dev = x[k] - mu                                                        # then observe x_k
        mu = LAM * mu + (1 - LAM) * x[k]
        s2 = THETA_EWMA * s2 + (1 - THETA_EWMA) * dev**2

    realized = loss[M:]
    exc_emp = realized > var_emp_ts
    exc_ewma = realized > var_ewma_ts

    # Task 4: normal-approximation CI from lecture, plus the exact binomial one as a check
    mean = J * (1 - ALPHA)
    half = norm.ppf(1 - BETA / 2) * np.sqrt(J * ALPHA * (1 - ALPHA))
    ci = (mean - half, mean + half)
    ci_exact = (binom.ppf(BETA / 2, J, 1 - ALPHA), binom.ppf(1 - BETA / 2, J, 1 - ALPHA))

    return {
        "prices": P, "dates": P.index[1:][M:], "n_prices": len(P), "n_returns": n, "J": J,
        "first_forecast_date": P.index[1:][M], "window_end_date": P.index[1:][M - 1],
        "var_emp": var_emp_ts, "var_ewma": var_ewma_ts, "realized": realized,
        "exc_emp": exc_emp, "exc_ewma": exc_ewma,
        "N_emp": int(exc_emp.sum()), "N_ewma": int(exc_ewma.sum()),
        "mean": mean, "ci": ci, "ci_exact": ci_exact,
        "clusters_emp": _clusters(exc_emp), "clusters_ewma": _clusters(exc_ewma),
        "init_mu": x[:M].mean(), "init_sd": x[:M].std(ddof=1),
    }


def _clusters(I):
    """Back-to-back exceedances (I_t = I_{t+1} = 1) vs what independence predicts."""
    I = I.astype(int)
    pairs = int((I[1:] * I[:-1]).sum())
    p = I.mean()
    return pairs, (len(I) - 1) * p**2
