"""MF731 Homework: numerical checks of the closed-form VaR answers.

For each distribution we compare the closed form against
  (i)  VaR_alpha = inf{l : F(l) >= alpha} found by root-finding / search on the CDF,
  (ii) the empirical quantile L_(ceil(n alpha)) of n Monte Carlo draws.
Run from the repository root:  python src/var_check.py
"""
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(731)
n_mc = 2_000_000


def empirical_var(sample, alpha):
    s = np.sort(sample)
    return s[int(np.ceil(len(s) * alpha)) - 1]


# ---- Problem 1: double-sided exponential with threshold l0 --------------------
def p1_cdf(l, a, b, l0):
    D = a * np.exp(-b * l0) + b * np.exp(a * l0)
    left = b * np.exp(a * np.minimum(l, l0)) / D
    right = a * (np.exp(-b * l0) - np.exp(-b * np.maximum(l, l0))) / D
    return left + right


def p1_var(alpha, a, b, l0):
    D = a * np.exp(-b * l0) + b * np.exp(a * l0)
    return np.log(a / ((1 - alpha) * D)) / b


def p1_sample(n, a, b, l0):
    """Inverse-transform sampling from the piecewise CDF."""
    D = a * np.exp(-b * l0) + b * np.exp(a * l0)
    u = rng.random(n)
    Fl0 = b * np.exp(a * l0) / D
    return np.where(u <= Fl0,
                    np.log(np.maximum(u, 1e-300) * D / b) / a,
                    np.log(a / (np.maximum(1 - u, 1e-300) * D)) / b)


# ---- Problem 2: binomial ------------------------------------------------------
def p2_var(alpha, n, p):
    k = 0
    while binom.cdf(k, n, p) < alpha:
        k += 1
    return k


# ---- Problem 3: ratio of independent exponentials -----------------------------
def p3_cdf(l, lam, theta):
    return lam * l / (theta + lam * l)


def p3_var(alpha, lam, theta):
    return theta * alpha / (lam * (1 - alpha))


def main():
    rows = []

    a, b, l0 = 2.0, 1.0, 0.5
    thresh = b / (b + a * np.exp(-(a + b) * l0))
    for alpha in (0.95, 0.99):
        assert alpha >= thresh
        closed = p1_var(alpha, a, b, l0)
        root = brentq(lambda l: p1_cdf(l, a, b, l0) - alpha, -50, 50)
        mc = empirical_var(p1_sample(n_mc, a, b, l0), alpha)
        rows.append(("1", f"a={a:g}, b={b:g}, l0={l0:g}", alpha, closed, root, mc))

    n, p, alpha = 6, 0.5, 0.9
    closed = 5
    search = p2_var(alpha, n, p)
    mc = empirical_var(rng.binomial(n, p, n_mc), alpha)
    rows.append(("2", f"n={n}, p={p:g}", alpha, closed, search, mc))
    cdf_table = [(k, binom.pmf(k, n, p), binom.cdf(k, n, p)) for k in range(n + 1)]

    lam, theta = 1.0, 2.0
    for alpha in (0.95, 0.99):
        closed = p3_var(alpha, lam, theta)
        root = brentq(lambda l: p3_cdf(l, lam, theta) - alpha, 1e-12, 1e6)
        Y = rng.exponential(1 / lam, n_mc)
        Z = rng.exponential(1 / theta, n_mc)
        mc = empirical_var(Y / Z, alpha)
        rows.append(("3", f"lambda={lam:g}, theta={theta:g}", alpha, closed, root, mc))

    print(f"Problem 1 threshold b/(b+a e^-(a+b)l0) = {thresh:.4f}")
    print(f"{'#':<3}{'params':<24}{'alpha':>6}{'closed':>10}{'CDF inv':>10}{'MC':>10}")
    for r in rows:
        print(f"{r[0]:<3}{r[1]:<24}{r[2]:>6.2f}{r[3]:>10.4f}{r[4]:>10.4f}{r[5]:>10.4f}")
    print("Binomial(6, 1/2): k, pmf, cdf")
    for k, pm, c in cdf_table:
        print(f"  {k}  {pm:.6f}  {c:.6f}")

    # ---- LaTeX table of the checks --------------------------------------------
    lines = [r"\begin{tabular}{clcccc}", r"\toprule",
             r"Problem & Parameters & $\alpha$ & Closed form & CDF inversion & Monte Carlo \\",
             r"\midrule"]
    for r in rows:
        params = "$" + (r[1].replace("lambda", r"\lambda").replace("theta", r"\theta")
                        .replace("l0", "l_0").replace(", ", r",\ ")) + "$"
        lines.append(f"{r[0]} & {params} & {r[2]:.2f} & {r[3]:.4f} & {r[4]:.4f} & {r[5]:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "generated").mkdir(exist_ok=True)
    (ROOT / "generated" / "checks.tex").write_text("\n".join(lines) + "\n")

    # ---- figure: the three CDFs with VaR marked --------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    ls = np.linspace(-3, 5, 600)
    axes[0].plot(ls, p1_cdf(ls, a, b, l0), color="#1f3b73")
    v = p1_var(0.95, a, b, l0)
    axes[0].axhline(0.95, color="grey", lw=0.8, ls=":")
    axes[0].axvline(v, color="#c0392b", ls="--", label=rf"VaR$_{{0.95}}$ = {v:.3f}")
    axes[0].axvline(l0, color="grey", lw=0.8, label=r"$l_0$")
    axes[0].set_title(rf"1. Two-sided exponential ($a$={a:g}, $b$={b:g})")

    ks = np.arange(-1, n + 2)
    axes[1].step(ks, binom.cdf(ks, n, p), where="post", color="#1f3b73")
    axes[1].axhline(0.9, color="grey", lw=0.8, ls=":")
    axes[1].axvline(5, color="#c0392b", ls="--", label=r"VaR$_{0.9}$ = 5")
    axes[1].set_title("2. Binomial(6, 1/2)")

    ls = np.linspace(0, 80, 600)
    axes[2].plot(ls, p3_cdf(ls, lam, theta), color="#1f3b73")
    v = p3_var(0.95, lam, theta)
    axes[2].axhline(0.95, color="grey", lw=0.8, ls=":")
    axes[2].axvline(v, color="#c0392b", ls="--", label=rf"VaR$_{{0.95}}$ = {v:.0f}")
    axes[2].set_title(rf"3. $Y/Z$ ($\lambda$={lam:g}, $\theta$={theta:g})")

    for ax in axes:
        ax.set_xlabel(r"$\ell$")
        ax.legend(frameon=False, loc="lower right", fontsize=8)
    axes[0].set_ylabel(r"$F_L(\ell)$")
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "var_cdfs.png", dpi=200)


if __name__ == "__main__":
    main()
