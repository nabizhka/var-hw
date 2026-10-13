"""MF731 HW 5 (05_var_10day): 1-day and 10-day 95% VaR of 100 short puts, delta hedged.

Under P:  S_{t+D} = S_t exp((mu - sigma^2/2) D + sigma sqrt(D) Z).
One period [T_{j-1}, T_j]:
  open : h_{j-1} = dP/dS(T_{j-1}, S_{j-1});  Y^new = Y_{j-1} + (h_{j-2} - h_{j-1}) S_{j-1}
  close: V_j = h_{j-1} S_j + Y^new e^{r D} - P(T_j, S_j)
With Y_0 = 0 and h_{-1} = 0, so V_0 = -P(0, S_0). Loss over [0, J D] is -(V_J - V_0).
"""
import numpy as np

from black_scholes import put_price, put_delta, call_price, call_delta
from common import var_emp, es_emp

ALPHA, DT, J = 0.95, 1 / 252, 10
T, S0, K, R, SIGMA, MU = 0.292, 152.51, 170.0, 0.00119, 0.4907, 0.1691
N_OPT, N_SIM, SEED = 100, 40_000, 731


def simulate_paths(n, steps, seed):
    rng = np.random.default_rng(seed)
    X = (MU - 0.5 * SIGMA**2) * DT + SIGMA * np.sqrt(DT) * rng.standard_normal((n, steps))
    S = S0 * np.exp(np.cumsum(X, axis=1))
    return np.hstack([np.full((n, 1), S0), S])          # column j is S_{T_j}


def hedged_loss(S, price, delta, steps):
    """Loss of the self-financing short-option + daily delta hedge over `steps` days."""
    n = S.shape[0]
    h_prev, Y = np.zeros(n), np.zeros(n)
    V0 = -price(0.0, S0, T, K, R, SIGMA)
    for j in range(1, steps + 1):
        t_open = (j - 1) * DT
        h = delta(t_open, S[:, j - 1], T, K, R, SIGMA)
        Y_new = Y + (h_prev - h) * S[:, j - 1]
        Y = Y_new * np.exp(R * DT)
        V = h * S[:, j] + Y - price(j * DT, S[:, j], T, K, R, SIGMA)
        h_prev = h
    return -N_OPT * (V - V0)


def run():
    S1 = simulate_paths(N_SIM, 1, SEED)
    S10 = simulate_paths(N_SIM, J, SEED + 1)

    L1 = hedged_loss(S1, put_price, put_delta, 1)
    L10 = hedged_loss(S10, put_price, put_delta, J)
    var1, var10 = var_emp(L1, ALPHA), var_emp(L10, ALPHA)

    # Check 1: put-call parity. On the same paths the hedged put minus the hedged call
    # is deterministic: L_put - L_call = -N (S0 - K e^{-rT}) (e^{r J D} - 1).
    C1 = hedged_loss(S1, call_price, call_delta, 1)
    C10 = hedged_loss(S10, call_price, call_delta, J)
    gap = lambda steps: -N_OPT * (S0 - K * np.exp(-R * T)) * np.expm1(R * steps * DT)
    parity_err = max(np.max(np.abs(L1 - C1 - gap(1))), np.max(np.abs(L10 - C10 - gap(J))))

    # Check 2: seed stability, 20 replications of each estimator
    reps1 = [var_emp(hedged_loss(simulate_paths(N_SIM, 1, 1000 + i), put_price, put_delta, 1), ALPHA)
             for i in range(20)]
    reps10 = [var_emp(hedged_loss(simulate_paths(N_SIM, J, 2000 + i), put_price, put_delta, J), ALPHA)
              for i in range(20)]

    # Diagnostic: 10-day loss if the 10 daily losses were i.i.d. copies of L1.
    # Isolates the distribution-shape effect from everything else.
    rng = np.random.default_rng(SEED + 2)
    iid_sum = L1[rng.integers(0, N_SIM, size=(N_SIM, J))].sum(axis=1)

    # No rebalancing at all over 10 days (hedge frozen at t = 0), for contrast
    S_end = S10[:, [0, -1]]
    L10_static = hedged_loss_static(S_end)

    return {
        "L1": L1, "L10": L10, "var1": var1, "var10": var10, "var10_sqrt": np.sqrt(J) * var1,
        "es1": es_emp(L1, ALPHA), "es10": es_emp(L10, ALPHA),
        "var_iid_sum": var_emp(iid_sum, ALPHA), "var10_static": var_emp(L10_static, ALPHA),
        "call_var1": var_emp(C1, ALPHA), "call_var10": var_emp(C10, ALPHA),
        "parity_err": parity_err,
        "se1": np.std(reps1, ddof=1), "se10": np.std(reps10, ddof=1),
        "mean1": L1.mean(), "sd1": L1.std(), "mean10": L10.mean(), "sd10": L10.std(),
        "skew1": _skew(L1), "skew10": _skew(L10),
        "P0": put_price(0.0, S0, T, K, R, SIGMA), "h0": put_delta(0.0, S0, T, K, R, SIGMA),
    }


def hedged_loss_static(S_end):
    """Hedge set at t = 0 and never touched; the lecture's 'no rebalancing' comparison."""
    h = put_delta(0.0, S0, T, K, R, SIGMA)
    P0 = put_price(0.0, S0, T, K, R, SIGMA)
    Y = -h * S0 * np.exp(R * J * DT)
    V = h * S_end[:, 1] + Y - put_price(J * DT, S_end[:, 1], T, K, R, SIGMA)
    return -N_OPT * (V + P0)


def _skew(a):
    z = (a - a.mean()) / a.std()
    return float((z**3).mean())
