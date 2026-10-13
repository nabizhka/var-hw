"""Numerical checks for the closed-form VaR / ES answers in HW 6.

Every answer is computed a second way, independent of the derivation:
  - discrete and mixed laws: VaR_u by inverting the cdf on a fine u-grid, ES as the
    average of VaR_u over (alpha, 1)  (the lecture definition, by quadrature)
  - all laws: 2,000,000-draw Monte Carlo with the empirical VaR / ES estimators
Writes generated/checks.tex.
"""
import os
import numpy as np
from scipy.stats import norm, expon

ROOT = os.path.join(os.path.dirname(__file__), "..")
rng = np.random.default_rng(731)
N = 2_000_000


def var_emp(l, a):
    l = np.sort(l)
    return l[int(np.ceil(len(l) * a - 1e-9)) - 1]


def es_emp(l, a):
    l = np.sort(l)
    n = len(l)
    k = int(np.ceil(n * a - 1e-9))
    return (l[k:].sum() + (k - n * a) * l[k - 1]) / (n * (1 - a))


def quantile_discrete(vals, probs):
    """Return u -> VaR_u = inf{x : F(x) >= u} for a finite law."""
    order = np.argsort(vals)
    v, F = np.asarray(vals)[order], np.cumsum(np.asarray(probs)[order])
    return lambda u: v[np.searchsorted(F, u - 1e-15)]


def es_by_quadrature(q, a, m=2_000_000):
    u = a + (1 - a) * (np.arange(m) + 0.5) / m        # midpoint rule on (alpha, 1)
    return q(u).mean()


rows = []

# (1) N(0, 20^2), alpha = .99
a = 0.99
var1, es1 = 20 * norm.ppf(a), 20 * norm.pdf(norm.ppf(a)) / (1 - a)
x = 20 * rng.standard_normal(N)
rows.append(("(1) VaR$_{.99}$, \\$m", var1, var1, var_emp(x, a)))
rows.append(("(1) ES$_{.99}$, \\$m", es1, es_by_quadrature(lambda u: 20 * norm.ppf(u), a), es_emp(x, a)))

# (2) two independent 0/10 losses
p = 0.02
q1 = quantile_discrete([0, 10], [1 - p, p])
qs = quantile_discrete([0, 10, 20], [(1 - p)**2, 2 * p * (1 - p), p**2])
L1 = 10.0 * (rng.random(N) < p)
L2 = 10.0 * (rng.random(N) < p)
for a, lab in [(0.98, ".98"), (0.99, ".99")]:
    rows.append((f"(2) VaR$_{{{lab}}}(L_1)$", {0.98: 0, 0.99: 10}[a], q1(a), var_emp(L1, a)))
    rows.append((f"(2) VaR$_{{{lab}}}(L_1+L_2)$", 10, qs(a), var_emp(L1 + L2, a)))
rows.append(("(2) ES$_{.98}(L_1)$", 10, es_by_quadrature(q1, 0.98), es_emp(L1, 0.98)))
rows.append(("(2) ES$_{.98}(L_1+L_2)$", 10.2, es_by_quadrature(qs, 0.98), es_emp(L1 + L2, 0.98)))

# (3) two loans: default prob 1.25% each, never both; loss U(0,10) on default, -0.2 otherwise
a = 0.99
u = rng.random(N)
d1 = u < 0.0125
d2 = (u >= 0.0125) & (u < 0.025)
U = 10 * rng.random(N)
loan1 = np.where(d1, U, -0.2)
loan2 = np.where(d2, U, -0.2)
port = loan1 + loan2


def q_loan(u):          # F = .9875 at -0.2, then uniform on [0, 10] with mass .0125
    return np.where(u <= 0.9875, -0.2, 10 * (u - 0.9875) / 0.0125)


def q_port(u):          # F = .975 at -0.4, then uniform on [-0.2, 9.8] with mass .025
    return np.where(u <= 0.975, -0.4, -0.2 + 10 * (u - 0.975) / 0.025)


rows.append(("(3) VaR$_{.99}$ loan 1, \\$m", 2.0, float(q_loan(a)), var_emp(loan1, a)))
rows.append(("(3) VaR$_{.99}$ portfolio, \\$m", 5.8, float(q_port(a)), var_emp(port, a)))
rows.append(("(3) ES$_{.99}$ loan 1, \\$m", 6.0, es_by_quadrature(q_loan, a), es_emp(loan1, a)))
rows.append(("(3) ES$_{.99}$ portfolio, \\$m", 7.8, es_by_quadrature(q_port, a), es_emp(port, a)))

# (4) counterexample: L2 = U, L1 = U - |U - alpha| / 2 < L2 a.s., equal VaR_alpha
a = 0.95
Uu = rng.random(N)
rows.append(("(4) VaR$_{.95}(L_2)$, $L_2=U$", a, a, var_emp(Uu, a)))
rows.append(("(4) VaR$_{.95}(L_1)$, $L_1<L_2$", a, a, var_emp(Uu - np.abs(Uu - a) / 2, a)))

# (5a) Exp with mean theta = 1 (scale-equivariant, so theta = 1 is enough), alpha = .99
a, th = 0.99, 1.0
e = expon(scale=th).rvs(size=N, random_state=rng)
rows.append(("(5a) VaR$_{.99}$, $\\theta=1$", -th * np.log(1 - a), expon(scale=th).ppf(a), var_emp(e, a)))
rows.append(("(5a) ES$_{.99}$, $\\theta=1$", th * (1 - np.log(1 - a)),
             es_by_quadrature(lambda u: -th * np.log(1 - u), a), es_emp(e, a)))

# (5b) Pareto, F = 1 - t^{-1/xi}; VaR = (1-a)^{-xi}, ES = VaR / (1 - xi) for xi < 1
for xi in [0.25, 0.5]:
    pa = (1 - rng.random(N)) ** (-xi)
    rows.append((f"(5b) VaR$_{{.99}}$, $\\xi={xi}$", (1 - a) ** -xi, (1 - a) ** -xi, var_emp(pa, a)))
    rows.append((f"(5b) ES$_{{.99}}$, $\\xi={xi}$", (1 - a) ** -xi / (1 - xi),
                 es_by_quadrature(lambda u: (1 - u) ** -xi, a), es_emp(pa, a)))

os.makedirs(os.path.join(ROOT, "generated"), exist_ok=True)
with open(os.path.join(ROOT, "generated", "checks.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrr}\n\\toprule\n")
    f.write("Quantity & Closed form & Quantile integral & Monte Carlo \\\\\n\\midrule\n")
    for lab, cf, qd, mc in rows:
        f.write(f"{lab} & {cf:.4f} & {qd:.4f} & {mc:.4f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
for lab, cf, qd, mc in rows:
    flag = "" if abs(cf - qd) < 1e-3 * max(1, abs(cf)) else "   <-- MISMATCH"
    print(f"{lab:40s} {cf:10.4f} {qd:10.4f} {mc:10.4f}{flag}")
