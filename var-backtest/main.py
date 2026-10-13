"""Run the computation, print a summary, write figures/ and generated/*.tex.

Every number quoted in main.tex comes from generated/values.tex, so the write-up
cannot drift from the code.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

import backtest as bt
from common import ROOT

GEN, FIG = os.path.join(ROOT, "generated"), os.path.join(ROOT, "figures")
os.makedirs(GEN, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

macros = {}


def put(name, value, fmt="{:,.2f}"):
    macros[name] = fmt.format(value) if not isinstance(value, str) else value


def head_tail_table(df, fname, fmt="{:,.2f}"):
    rows = pd.concat([df.head(5), df.tail(5)])
    cols = list(df.columns)
    lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             "Date & " + " & ".join(cols) + r" \\", r"\midrule"]
    for i, (d, r) in enumerate(rows.iterrows()):
        if i == 5:
            lines.append(r"\multicolumn{" + str(len(cols) + 1) + r"}{c}{$\vdots$} \\")
        lines.append(d.strftime("%Y-%m-%d") + " & " + " & ".join(fmt.format(v) for v in r.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(GEN, fname), "w") as f:
        f.write("\n".join(lines) + "\n")

b = bt.run()
head_tail_table(b["prices"].to_frame("S\\&P 500"), "b_prices.tex")
put("BnPrices", b["n_prices"], "{:d}")
put("BnReturns", b["n_returns"], "{:d}")
put("BJ", b["J"], "{:d}")
put("BJsheet", b["n_prices"] - bt.M, "{:d}")
put("BfirstDate", b["first_forecast_date"].strftime("%Y-%m-%d"))
put("BwinEnd", b["window_end_date"].strftime("%Y-%m-%d"))
put("BkEmp", int(np.ceil(bt.M * bt.ALPHA)), "{:d}")
put("BNemp", b["N_emp"], "{:d}")
put("BNewma", b["N_ewma"], "{:d}")
put("Bmean", b["mean"], "{:.2f}")
put("Bsd", np.sqrt(b["J"] * bt.ALPHA * (1 - bt.ALPHA)), "{:.2f}")
put("BciLo", b["ci"][0], "{:.2f}")
put("BciHi", b["ci"][1], "{:.2f}")
put("BciExLo", b["ci_exact"][0], "{:.0f}")
put("BciExHi", b["ci_exact"][1], "{:.0f}")
put("BzEmp", (b["N_emp"] - b["mean"]) / np.sqrt(b["J"] * bt.ALPHA * (1 - bt.ALPHA)), "{:.2f}")
put("BzEwma", (b["N_ewma"] - b["mean"]) / np.sqrt(b["J"] * bt.ALPHA * (1 - bt.ALPHA)), "{:.2f}")
put("BinitMu", 1e4 * b["init_mu"], "{:.2f}")
put("BinitSd", 100 * b["init_sd"], "{:.3f}")
put("BpairsEmp", b["clusters_emp"][0], "{:d}")
put("BpairsEmpExp", b["clusters_emp"][1], "{:.1f}")
put("BpairsEwma", b["clusters_ewma"][0], "{:d}")
put("BpairsEwmaExp", b["clusters_ewma"][1], "{:.1f}")
put("BavgVarEmp", 100 * b["var_emp"].mean(), "{:.2f}")
put("BavgVarEwma", 100 * b["var_ewma"].mean(), "{:.2f}")

# zero-mean EWMA variant: same recursion, mu forced to 0 in the VaR formula only
x = np.log(b["prices"] / b["prices"].shift(1)).dropna().values
mu, s2, cnt0 = x[:bt.M].mean(), x[:bt.M].var(ddof=1), 0
for k in range(bt.M, len(x)):
    cnt0 += (-np.expm1(x[k])) > 1 - np.exp(np.sqrt(s2) * norm.ppf(1 - bt.ALPHA))
    dev = x[k] - mu
    mu = bt.LAM * mu + (1 - bt.LAM) * x[k]
    s2 = bt.THETA_EWMA * s2 + (1 - bt.THETA_EWMA) * dev**2
put("BNewmaZeroMu", int(cnt0), "{:d}")

d = pd.DatetimeIndex(b["dates"])
yr = pd.DataFrame({"emp": b["exc_emp"], "ewma": b["exc_ewma"]}, index=d).groupby(d.year).agg(["sum", "count"])
with open(os.path.join(GEN, "b_years.tex"), "w") as f:
    f.write(r"\begin{tabular}{lrrrr}\toprule" "\n")
    f.write(r"Year & Days & Expected & Empirical & EWMA \\ \midrule" "\n")
    for y, row in yr.iterrows():
        n = row[("emp", "count")]
        f.write(f"{y} & {n} & {0.05*n:.1f} & {row[('emp','sum')]} & {row[('ewma','sum')]} \\\\\n")
    f.write(r"\midrule" "\n")
    f.write(f"Total & {b['J']} & {b['mean']:.1f} & {b['N_emp']} & {b['N_ewma']} \\\\\n")
    f.write(r"\bottomrule\end{tabular}" "\n")

fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.plot(d, 100 * b["realized"], color="#9aa5b1", lw=0.5, label="realized loss")
ax.plot(d, 100 * b["var_emp"], color="#1f3b73", lw=1.1, label=f"empirical VaR ({b['N_emp']} exceedances)")
ax.plot(d, 100 * b["var_ewma"], color="#cc0000", lw=0.9, label=f"EWMA VaR ({b['N_ewma']} exceedances)")
ax.scatter(d[b["exc_emp"]], 100 * b["realized"][b["exc_emp"]] + 0.6, marker="v", s=8, color="#1f3b73")
ax.scatter(d[b["exc_ewma"]], 100 * b["realized"][b["exc_ewma"]] + 1.2, marker="v", s=8, color="#cc0000")
ax.set_ylabel("loss per $1 (%)")
ax.set_ylim(-6, 8)
ax.legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "b_backtest.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- write macros
with open(os.path.join(GEN, "values.tex"), "w") as f:
    for k, v in macros.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


print(f"J = {b['J']}, exceedances empirical {b['N_emp']}, EWMA {b['N_ewma']}, "
      f"CI ({b['ci'][0]:.2f}, {b['ci'][1]:.2f})")
