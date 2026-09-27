# -*- coding: utf-8 -*-
"""
plot_welfare_newborns.py

Plots the expenditure-equivalent welfare loss of sea level rise for newborns
(sheet 'tax_equiv_newborns' in welfare_results.xlsx), one figure per belief type.
Output: welfare_newborns_realists.pdf and welfare_newborns_sceptics.pdf
"""

import pandas as pd
import matplotlib.pyplot as plt

FILE = "welfare_results.xlsx"
SHEET = "tax_equiv_newborns"
START_YEAR, STEP = 1998, 2
SHARE_Y = True   # same y-axis range in both panels, so they compare directly side by side

# one blue hue, light -> dark = income level 1 -> 5
COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]

plt.rcParams.update({
    "font.size": 18, "axes.labelsize": 20, "xtick.labelsize": 17,
    "ytick.labelsize": 17, "legend.fontsize": 15, "legend.title_fontsize": 15,
    "axes.spines.top": False, "axes.spines.right": False,
})

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
    fig.savefig(f"welfare_newborns_{name}.pdf")
    fig.savefig(f"welfare_newborns_{name}.png", dpi=150)
    plt.close(fig)
