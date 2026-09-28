# -*- coding: utf-8 -*-
"""
plot_welfare.py

(1) Newborns: plots the expenditure-equivalent welfare loss of sea level rise
    (sheet 'tax_equiv_newborns' in welfare_results.xlsx), one figure per belief type.
(2) Households alive in 1998: LaTeX table of the expenditure-equivalent tax
    (sheets 'tax_equiv_C', 'tax_equiv_NC', 'tax_equiv_renter') by income level,
    plus an average over income levels weighted by the t = 0 income distribution.

Output (in OUT_DIR): welfare_newborns_realists/sceptics (.png and .eps)
                     and welfare_SLR_table.tex (tabular only; \\input it in the paper)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import misc_functions as misc
import par_epsilons as parfile
import tauchen as tauch

FILE = "welfare_results.xlsx"
SHEET = "tax_equiv_newborns"
OUT_DIR = r"C:\Users\TPRINS\OneDrive - UvA\Documenten\Python files\New coding round July 2026\Plaatjes"
START_YEAR, STEP = 1998, 2
SHARE_Y = True   # same y-axis range in both panels, so they compare directly side by side

# one blue hue, light -> dark = income level 1 -> 5
COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]

plt.rcParams.update({
    "font.size": 18, "axes.labelsize": 20, "xtick.labelsize": 17,
    "ytick.labelsize": 17, "legend.fontsize": 15, "legend.title_fontsize": 15,
    "axes.spines.top": False, "axes.spines.right": False,
})
os.makedirs(OUT_DIR, exist_ok=True)

# ============ (1) newborns: figures ============
# --- load: t is only written on the first row of each (t, k) block, so fill it down
df = pd.read_excel(FILE, sheet_name=SHEET)
df["t"] = df["t"].ffill().astype(int)
df = df[df["t"] < df["t"].max()]          # last period is never computed (all zeros)
df["year"] = START_YEAR + STEP * df["t"]
e_cols = [c for c in df.columns if c.startswith("e")]
df[e_cols] = 100 * df[e_cols]             # to percent

ymax = df[e_cols].max().max() * 1.05

for k, name in [(0, "realists"), (1, "sceptics")]:
    d = df[df["k"] == k]
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, col in enumerate(e_cols):
        ax.plot(d["year"], d[col], color=COLORS[i], lw=2.2, label=str(i + 1))
    ax.set_xlabel("Year")
    ax.set_ylabel("Equivalent tax (%)")
    ax.set_xlim(START_YEAR, df["year"].max())
    if SHARE_Y:
        ax.set_ylim(0, ymax)
    ax.grid(axis="y", color="0.88", lw=0.8)
    ax.legend(title="Income level", frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "eps"):
        fig.savefig(os.path.join(OUT_DIR, f"welfare_newborns_{name}.{ext}"), dpi=300)
    plt.close(fig)

# ============ (2) households alive in 1998: table ============
TABLE_SHEETS = [("tax_equiv_C", "Coastal owners"),
                ("tax_equiv_NC", "Non-coastal owners"),
                ("tax_equiv_renter", "Renters")]

# income weights: population distribution over income levels at t = 0 (first row of mPi_E)
par = misc.construct_jitclass(parfile.par_dict)
mMarkov, vE = tauch.tauchen(par.dRho, par.dSigmaeps, par.iNumStates, par.iM, par.time_increment)
vPi_E = tauch.initial_dist(par, vE)
mPi_E = tauch.weight_matrix(par, vE, vPi_E, mMarkov)
w = np.asarray(mPi_E[0], dtype=float)
w = w / w.sum()

# rows: income levels 1..5; columns: (household group, belief type); values in percent
cols = {}
for sheet, group in TABLE_SHEETS:
    d = pd.read_excel(FILE, sheet_name=sheet, index_col="k")
    for k, kname in [(0, "Realists"), (1, "Sceptics")]:
        cols[(group, kname)] = 100 * d.loc[k, e_cols].to_numpy(dtype=float)
tab = pd.DataFrame(cols, index=[str(i + 1) for i in range(len(e_cols))])
tab.loc["Average"] = w @ tab.to_numpy()
print("Income weights:", np.round(w, 4))
print(tab.round(3))

# booktabs tabular; numbers in math mode so negatives get a proper minus sign
fmt = lambda v: f"${v:.2f}$"
lines = [r"\begin{tabular}{l" + "r" * tab.shape[1] + "}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{g}}}" for _, g in TABLE_SHEETS) + r" \\",
         "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(TABLE_SHEETS))),
         "Income level & " + " & ".join(kname for _, kname in tab.columns) + r" \\",
         r"\midrule"]
for label, row in tab.iterrows():
    if label == "Average":
        lines.append(r"\midrule")
    lines.append(f"{label} & " + " & ".join(fmt(v) for v in row) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]

with open(os.path.join(OUT_DIR, "welfare_SLR_table.tex"), "w") as fh:
    fh.write("\n".join(lines) + "\n")