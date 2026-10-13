"""Run the computation, print a summary, write figures/ and generated/*.tex.

Every number quoted in main.tex comes from generated/values.tex, so the write-up
cannot drift from the code.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

import tenday as td
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

c = td.run()
for k, nm in [("var1", "Cvarone"), ("var10", "Cvarten"), ("var10_sqrt", "Cvarsqrt"),
              ("var_iid_sum", "Cvariid"), ("var10_static", "Cvarstatic"),
              ("call_var1", "Ccallone"), ("call_var10", "Ccallten"), ("es1", "Cesone"),
              ("es10", "Cesten"), ("se1", "Cseone"), ("se10", "Cseten"),
              ("mean1", "Cmeanone"), ("sd1", "Csdone"), ("mean10", "Cmeanten"), ("sd10", "Csdten"),
              ("skew1", "Cskewone"), ("skew10", "Cskewten"), ("P0", "CPzero")]:
    put(nm, c[k])
put("Chzero", c["h0"], "{:.4f}")
put("Cparity", c["parity_err"], "{:.0e}")
put("Cgap", c["var10_sqrt"] - c["var10"])
put("CgapPct", 100 * (c["var10_sqrt"] / c["var10"] - 1), "{:.1f}")
put("CgapSE", np.sqrt(c["se10"]**2 + 10 * c["se1"]**2))
put("Cratio", c["var10"] / c["var1"], "{:.3f}")
put("Csqrtten", np.sqrt(10), "{:.3f}")

fig, ax = plt.subplots(figsize=(7.5, 3.4))
bins = np.linspace(-150, 400, 160)
ax.hist(c["L10"], bins=bins, density=True, histtype="stepfilled", color="#b9c6d8",
        label="10-day loss, daily rebalancing")
ax.hist(np.sqrt(10) * c["L1"], bins=bins, density=True, histtype="step", color="#cc0000",
        lw=1.2, label=r"$\sqrt{10}\times$ one-day loss")
ax.axvline(c["var10"], color="#1f3b73", ls="--", lw=1.2, label=f"10-day VaR {c['var10']:.2f}")
ax.axvline(c["var10_sqrt"], color="#cc0000", ls="--", lw=1.2, label=rf"$\sqrt{{10}}$ rule {c['var10_sqrt']:.2f}")
ax.set_xlabel("loss on 100 short puts ($)")
ax.set_xlim(-150, 400)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "c_tenday.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- write macros
with open(os.path.join(GEN, "values.tex"), "w") as f:
    for k, v in macros.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


print(f"VaR 1d {c['var1']:.2f}, sqrt(10) x 1d {c['var10_sqrt']:.2f}, 10d {c['var10']:.2f}")
