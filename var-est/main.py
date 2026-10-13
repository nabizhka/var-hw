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

import estimation as est
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

a = est.run()
aa = est.run("stocks_adjclose.csv")
head_tail_table(a["prices"], "a_prices.tex")
put("AnPrices", a["n_prices"], "{:d}")
put("AnReturns", a["n_returns"], "{:d}")
for t, w, th in zip(est.TICKERS, a["w"], a["theta"]):
    put(f"Aw{t}", 100 * w, "{:.2f}")
    put(f"Atheta{t}", th, "{:,.0f}")
for k in ["full", "lin", "quad"]:
    put(f"Aemp{k}", a["emp"][k], "{:,.0f}")
    put(f"Anorm{k}", a["norm"][k], "{:,.0f}")
    put(f"AadjEmp{k}", aa["emp"][k], "{:,.0f}")
    put(f"AadjNorm{k}", aa["norm"][k], "{:,.0f}")
put("AnormlinMC", a["norm"]["lin_mc"], "{:,.0f}")
put("AnormfullSE", a["norm_full_se"], "{:,.0f}")
put("AnormlinExact", a["norm_lin_exact"], "{:,.2f}")
mu_d = a["theta"] @ a["mu"]
put("Adrift", mu_d, "{:,.0f}")
put("AnormlinZeroMu", np.sqrt(a["theta"] @ a["Sigma"] @ a["theta"]) * norm.ppf(est.ALPHA), "{:,.0f}")
put("AvolEWMA", 100 * a["port_vol_ewma"], "{:.2f}")
put("AvolSample", 100 * a["port_vol_sample"], "{:.2f}")
put("AvolRecent", 100 * a["port_vol_recent"], "{:.2f}")
put("AinitGap", a["ewma_init_gap"], "{:.0e}")
put("AlinMinusFull", a["emp"]["lin"] - a["emp"]["full"], "{:,.0f}")
put("AlinMinusFullPct", 100 * (a["emp"]["lin"] / a["emp"]["full"] - 1), "{:.1f}")
put("AempOverNorm", 100 * (a["emp"]["full"] / a["norm"]["full"] - 1), "{:.0f}")
r = a["port_ret"]
put("AworstDay", 100 * r.min(), "{:.2f}")
put("AworstDate", a["returns"].index[np.argmin(r)].strftime("%Y-%m-%d"))
put("AkEmp", int(np.ceil(a["n_returns"] * est.ALPHA - 1e-12)), "{:d}")

with open(os.path.join(GEN, "a_ewma.tex"), "w") as f:
    m, S = a["mu"], a["Sigma"]
    f.write(r"\hat\mu = 10^{-4}\times\begin{pmatrix}" + r"\\".join(f"{1e4*v:.3f}" for v in m) + r"\end{pmatrix},\quad")
    f.write(r"\hat\Sigma = 10^{-4}\times\begin{pmatrix}" +
            r"\\".join(" & ".join(f"{1e4*v:.3f}" for v in row) for row in S) + r"\end{pmatrix}")

# figure A: empirical loss distribution vs the EWMA-normal one
Lh = est.l_full(a["returns"].values, a["theta"])
fig, ax = plt.subplots(figsize=(7.5, 3.4))
ax.hist(Lh, bins=70, density=True, color="#b9c6d8", label="historical full losses (500 days)")
g = np.linspace(-60000, 60000, 600)
sd = np.sqrt(a["theta"] @ a["Sigma"] @ a["theta"])
ax.plot(g, norm.pdf(g, -mu_d, sd), color="#cc0000", lw=1.6, label="EWMA normal (linear loss)")
ax.axvline(a["emp"]["full"], color="#1f3b73", ls="--", lw=1.2, label=f"empirical VaR {a['emp']['full']:,.0f}")
ax.axvline(a["norm"]["full"], color="#cc0000", ls="--", lw=1.2, label=f"normal-EWMA VaR {a['norm']['full']:,.0f}")
ax.set_xlabel("one-day loss ($)")
ax.set_xlim(-60000, 60000)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "a_losses.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- write macros
with open(os.path.join(GEN, "values.tex"), "w") as f:
    for k, v in macros.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


print(f"empirical VaR  full {a['emp']['full']:,.0f}  lin {a['emp']['lin']:,.0f}  quad {a['emp']['quad']:,.0f}")
print(f"normal-EWMA    full {a['norm']['full']:,.0f}  lin {a['norm']['lin']:,.0f}  quad {a['norm']['quad']:,.0f}")
