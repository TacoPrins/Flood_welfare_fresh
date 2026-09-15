# -*- coding: utf-8 -*-
"""
plot_dist.py — generate all distribution diagnostics from the reduced inputs
written by create_plotting_inputs.py.

This file never touches the full 6D distributions.  It reads the small,
already-reduced arrays from the Excel file and plots them, so you can re-style
or re-select plots (e.g. pick different t) without rerunning the model.

Reduced sheets (see create_plotting_inputs.py):
  masses   : scenario, k, t, mc, mnc, mr
  defaults : scenario, k, t, def_c, def_nc
  cond     : scenario, loc, var, k, e, t, i, value   (owner marginals over l/m/h)
  renter   : scenario, k, e, t, i, value             (renter marginal over x)
  ltv      : scenario, loc, k, t, i, value           (LTV density, e aggregated)
  grids    : vTime, vM_sim, vH, vL_sim, vX_sim
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

K_LABEL = {0: "Realist", 1: "Sceptic"}
_VAR_GRID = {"l": "vL_sim", "m": "vM_sim", "h": "vH"}
_LOC_NAME = {"C": "Coastal", "NC": "Non-coastal"}


# ---------- small numeric helpers (unchanged in spirit) ----------
def _norm(w):
    s = w.sum()
    return w / s if s > 0 else w


def _wmean(grid, w):
    s = w.sum()
    return np.dot(grid, w) / s if s > 0 else np.nan


def _wquant(grid, w, q):
    s = w.sum()
    if s <= 0:
        return np.nan
    c = np.cumsum(w) / s
    return np.interp(q, c, grid)


def _safe_div(a, b):
    """Elementwise a/b, returning NaN where b == 0 (zero-mass points)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    return np.divide(a, b, out=np.full_like(a, np.nan), where=b != 0)


# ---------- load reduced inputs from Excel ----------
def load_inputs(path="plotting_inputs.xlsx"):
    """Read all sheets into a dict of DataFrames + a grids dict."""
    sheets = pd.read_excel(path, sheet_name=["masses", "defaults", "cond",
                                             "renter", "ltv", "grids"])
    g = sheets.pop("grids")
    grids = {c: g[c].dropna().to_numpy() for c in g.columns}
    return sheets, grids


def _pivot_TN(df):
    """Long (t, i, value) rows -> dense (T, n) array, t and i sorted ascending."""
    m = df.pivot_table(index="t", columns="i", values="value", aggfunc="sum")
    m = m.sort_index().sort_index(axis=1)
    return m.to_numpy()


# ---------- 1) tenure shares by belief type ----------
def plot_tenure_shares(masses, tag, vTime=None):
    d = masses[masses["scenario"] == tag]
    ks = sorted(d["k"].unique()); K = len(ks)
    piv = d.pivot_table(index="t", columns="k", values=["mc", "mnc", "mr"])
    t = piv.index.to_numpy() if vTime is None else vTime[:len(piv)]
    mc = piv["mc"].to_numpy(); mnc = piv["mnc"].to_numpy(); mr = piv["mr"].to_numpy()
    tot = mc + mnc + mr

    fig, ax = plt.subplots(1, K, figsize=(5.5 * K, 4), squeeze=False); ax = ax[0]
    for j, k in enumerate(ks):
        ax[j].stackplot(t, _safe_div(mc[:, j], tot[:, j]),
                        _safe_div(mnc[:, j], tot[:, j]),
                        _safe_div(mr[:, j], tot[:, j]),
                        labels=["Coastal own", "Non-coastal own", "Renter"],
                        alpha=.85)
        ax[j].set_title(K_LABEL.get(k, f"k={k}"))
        ax[j].set_xlabel("t"); ax[j].set_ylim(0, 1)
    ax[0].set_ylabel("Population share"); ax[0].legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    return ax


def plot_coastal_share_lines(masses, tag, vTime=None):
    d = masses[masses["scenario"] == tag]
    ks = sorted(d["k"].unique())
    piv = d.pivot_table(index="t", columns="k", values=["mc", "mnc", "mr"])
    t = piv.index.to_numpy() if vTime is None else vTime[:len(piv)]
    mc = piv["mc"].to_numpy(); mnc = piv["mnc"].to_numpy(); mr = piv["mr"].to_numpy()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for j, k in enumerate(ks):
        ax[0].plot(t, _safe_div(mc[:, j], mc[:, j] + mnc[:, j]), label=K_LABEL.get(k, k))
        ax[1].plot(t, _safe_div(mc[:, j] + mnc[:, j], mc[:, j] + mnc[:, j] + mr[:, j]),
                   label=K_LABEL.get(k, k))
    ax[0].set_title("Coastal share of owners"); ax[1].set_title("Homeownership rate")
    for a in ax: a.set_xlabel("t"); a.legend()
    plt.tight_layout()
    return ax


# ---------- 2) conditional distributions over l, m, h (owners) and x (renters) ----------
def _cond_marg(cond, tag, loc, var, k, e):
    """Reduced owner marginal -> (T, n) for one (scenario, loc, var, k, e)."""
    sel = cond[(cond["scenario"] == tag) & (cond["location"] == loc) &
               (cond["var"] == var) & (cond["k"] == k) & (cond["e"] == e)]
    return _pivot_TN(sel)


def plot_conditional_dists(cond, grids, tag, loc, k, e_list=None,
                           t_list=(0, -1)):
    d = cond[(cond["scenario"] == tag) & (cond["location"] == loc)]
    E = int(d["e"].max()) + 1
    e_list = list(range(E)) if e_list is None else e_list

    specs = [("l", grids["vL_sim"], "LTV $l$"),
             ("m", grids["vM_sim"], "Savings $m$"),
             ("h", grids["vH"], "House size $h$")]
    fig, ax = plt.subplots(3, len(e_list), figsize=(3.4 * len(e_list), 8),
                           squeeze=False)
    for r, (var, grid, name) in enumerate(specs):
        for c, e in enumerate(e_list):
            marg = _cond_marg(cond, tag, loc, var, k, e)     # (T, n)
            T = marg.shape[0]; tl = [t % T for t in t_list]
            for t in tl:
                ax[r, c].plot(grid[:marg.shape[1]], _norm(marg[t]), label=f"t={t}")
            if r == 0: ax[r, c].set_title(f"e={e}")
            if c == 0: ax[r, c].set_ylabel(name)
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(f"{_LOC_NAME[loc]} owners — {K_LABEL.get(k, k)} ({tag})")
    plt.tight_layout()
    return ax


def plot_renter_savings(renter, grids, tag, k_list=None, e_list=None,
                        t_list=(0, -1)):
    d = renter[renter["scenario"] == tag]
    vX = grids["vX_sim"]
    ks = sorted(d["k"].unique()); es = sorted(d["e"].unique())
    k_list = ks if k_list is None else k_list
    e_list = es if e_list is None else e_list

    fig, ax = plt.subplots(len(k_list), len(e_list),
                           figsize=(3.4 * len(e_list), 3.2 * len(k_list)),
                           squeeze=False)
    for r, k in enumerate(k_list):
        for c, e in enumerate(e_list):
            marg = _pivot_TN(d[(d["k"] == k) & (d["e"] == e)])      # (T, nx)
            T = marg.shape[0]; tl = [t % T for t in t_list]
            for t in tl:
                ax[r, c].plot(vX[:marg.shape[1]], _norm(marg[t]), label=f"t={t}")
            if r == 0: ax[r, c].set_title(f"e={e}")
            if c == 0: ax[r, c].set_ylabel(K_LABEL.get(k, k))
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(f"Renter savings $x$ ({tag})")
    plt.tight_layout()
    return ax


# ---------- 3) moment paths ----------
def plot_moment_paths(cond, grids, tag, loc, var="l", vTime=None):
    """Plot mean paths by income group, with a publication-style layout."""
    grid = grids[_VAR_GRID[var]]
    d = cond[(cond["scenario"] == tag) &
             (cond["location"] == loc) &
             (cond["var"] == var)]

    ks = sorted(d["k"].unique())
    es = sorted(d["e"].unique())
    E = len(es)
    T = int(d["t"].max()) + 1

    # Calendar years: model period 0 corresponds to 1998,
    # with each subsequent period representing two years.
    t = 1998 + 2 * np.arange(T)

    var_name = {
        "l": "LTV ratio",
        "m": "savings",
        "h": "house size",
    }.get(var, var)

    # Fixed y-axis ranges for comparability across all plots.
    y_limits = {
        "l": (0, 0.8),  # LTV ratio
        "m": (0, 3.0),  # Savings
        "h": (0, 5.0),  # House size
    }

    # Muted, colour-blind-friendly palette.
    # The pooled mean is deliberately neutral and heavier.
    palette = ["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677"]

    if E <= len(palette):
        colours = palette[:E]
    else:
        colours = plt.cm.viridis(np.linspace(0.12, 0.88, E))

    axes = []

    for k in ks:
        belief = K_LABEL.get(k, f"k={k}")

        fig, ax = plt.subplots(
            figsize=(7.0, 4.6),
            constrained_layout=True
        )

        # Collect each income level's mass-carrying marginal (unnormalized).
        margs = []

        for idx, e in enumerate(es):
            marg = _pivot_TN(
                d[(d["k"] == k) & (d["e"] == e)]
            )
            margs.append(marg)

            g = grid[:marg.shape[1]]

            mu = np.array([
                _wmean(g, marg[i])
                for i in range(marg.shape[0])
            ])

            sd = idx - (E - 1) / 2

            sd_label = (
                f"{sd:+.0f} SD"
                if sd != 0
                else "Median income"
            )

            ax.plot(
                t[:len(mu)],
                mu,
                label=sd_label,
                color=colours[idx],
                lw=1.7,
                alpha=0.95,
            )

        # Income-averaged mean: pool all income groups (mass-weighted).
        pooled = np.sum(margs, axis=0)

        g = grid[:pooled.shape[1]]

        mu_all = np.array([
            _wmean(g, pooled[i])
            for i in range(pooled.shape[0])
        ])

        ax.plot(
            t[:len(mu_all)],
            mu_all,
            label="All incomes (mass-weighted)",
            color="#222222",
            lw=2.6,
            zorder=5,
        )

        # Title and axis labels.
        ax.set_title(
            f"{_LOC_NAME.get(loc, loc)} owners · {belief} · {tag}",
            loc="left",
            fontsize=11,
            fontweight="semibold",
            pad=10,
        )

        ax.set_xlabel("Year")
        ax.set_ylabel(f"Mean {var_name}")

        # Fixed y-axis range depending on variable.
        if var in y_limits:
            ax.set_ylim(*y_limits[var])

        # Calendar-year x-axis.
        # The data begin in 1998 and remain at two-year intervals,
        # but only selected years are labelled.
        ax.set_xlim(t[0], t[-1])
        ax.set_xticks([
            2000,
            2020,
            2040,
            2060,
            2080,
            2100,
        ])

        # Quiet editorial styling:
        # horizontal guides only, no box around axes.
        ax.grid(
            axis="y",
            which="major",
            color="#D9D9D9",
            lw=0.7,
            alpha=0.8,
        )

        ax.set_axisbelow(True)

        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#888888")

        ax.tick_params(
            axis="both",
            labelsize=9,
            colors="#444444",
        )

        ax.margins(x=0.01)

        # Put the legend outside the data region in two compact rows.
        ax.legend(
            title="Income group",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=3,
            fontsize=8.5,
            title_fontsize=8.5,
            frameon=False,
            handlelength=2.6,
            columnspacing=1.4,
        )

        axes.append(ax)

    return axes


# ---------- 4) heatmap: LTV distribution over time ----------
def plot_ltv_heatmap(ltv, grids, tag, loc, k, vTime=None):
    vL = grids["vL_sim"]
    d = ltv[(ltv["scenario"] == tag) & (ltv["location"] == loc) & (ltv["k"] == k)]
    w = _pivot_TN(d)                                          # (T, l)
    w = np.array([_norm(row) for row in w])
    T = w.shape[0]; t = np.arange(T) if vTime is None else vTime[:T]

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.pcolormesh(t, vL[:w.shape[1]], w.T, shading="auto", cmap="magma")
    fig.colorbar(im, ax=ax, label="density")
    ax.set_xlabel("t"); ax.set_ylabel("LTV $l$")
    ax.set_title(f"{_LOC_NAME[loc]} — {K_LABEL.get(k, k)} ({tag})")
    plt.tight_layout()
    return ax


# ---------- 5) default rates by belief type ----------
def plot_default_rates(masses, defaults, tag, vTime=None):
    """
    Plot default-rate paths separately for coastal and non-coastal owners,
    using a publication-style layout.
    """
    dm = masses[masses["scenario"] == tag]
    dd = defaults[defaults["scenario"] == tag]

    ks = sorted(dd["k"].unique())

    # Pivot masses and defaults into (T, K) arrays.
    m = dm.pivot_table(
        index="t",
        columns="k",
        values=["mc", "mnc"]
    )

    de = dd.pivot_table(
        index="t",
        columns="k",
        values=["def_c", "def_nc"]
    )

    T = len(de)

    # Calendar years: model period 0 = 1998,
    # with two-year spacing between periods.
    t = 1998 + 2 * np.arange(T)

    mc = m["mc"].to_numpy()[:T]
    mnc = m["mnc"].to_numpy()[:T]

    dc = de["def_c"].to_numpy()
    dnc = de["def_nc"].to_numpy()

    # Restrained, colour-blind-friendly colours.
    palette = ["#4477AA", "#EE6677", "#228833", "#CCBB44"]

    # Location-specific inputs.
    specs = [
        {
            "name": "Coastal",
            "defaults": dc,
            "mass": mc,
        },
        {
            "name": "Non-coastal",
            "defaults": dnc,
            "mass": mnc,
        },
    ]

    axes = []

    for spec in specs:

        fig, ax = plt.subplots(
            figsize=(7.0, 4.6),
            constrained_layout=True
        )

        for j, k in enumerate(ks):

            rate = _safe_div(
                spec["defaults"][:, j],
                spec["mass"][:, j]
            )

            ax.plot(
                t,
                rate,
                label=K_LABEL.get(k, f"k={k}"),
                color=palette[j % len(palette)],
                lw=2.0,
                alpha=0.95,
            )

        # Title and axis labels.
        ax.set_title(
            f"{spec['name']} owners · {tag}",
            loc="left",
            fontsize=11,
            fontweight="semibold",
            pad=10,
        )

        ax.set_xlabel("Year")
        ax.set_ylabel("Default rate")

        # Calendar-year x-axis:
        # observations remain at two-year intervals, but only selected
        # years are labelled.
        ax.set_xlim(t[0], t[-1])
        ax.set_xticks([
            2000,
            2020,
            2040,
            2060,
            2080,
            2100,
        ])

        # Default rates should naturally start at zero.
        ax.set_ylim(bottom=0)

        # Quiet editorial styling.
        ax.grid(
            axis="y",
            which="major",
            color="#D9D9D9",
            lw=0.7,
            alpha=0.8,
        )

        ax.set_axisbelow(True)

        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#888888")

        ax.tick_params(
            axis="both",
            labelsize=9,
            colors="#444444",
        )

        ax.margins(x=0.01)

        # Legend outside plotting region.
        ax.legend(
            title="Belief type",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=len(ks),
            fontsize=8.5,
            title_fontsize=8.5,
            frameon=False,
            handlelength=2.6,
            columnspacing=1.8,
        )

        axes.append(ax)

    return axes


# ---------- driver ----------
def plot_all_distributions(path="plotting_inputs.xlsx"):
    """Load reduced inputs from Excel and plot every diagnostic for HE and RE."""
    sheets, grids = load_inputs(path)
    masses, defaults = sheets["masses"], sheets["defaults"]
    cond, renter, ltv = sheets["cond"], sheets["renter"], sheets["ltv"]
    vTime = grids.get("vTime")

    for tag in ("HE", "RE"):
        # 1) tenure shares
        plot_tenure_shares(masses, tag, vTime)
        plot_coastal_share_lines(masses, tag, vTime)

        # 2) conditional dists + LTV heatmaps, per loc and k
        ks = sorted(masses[masses["scenario"] == tag]["k"].unique())
        for loc in ("C", "NC"):
            for k in ks:
                plot_conditional_dists(cond, grids, tag, loc, k)
                plot_ltv_heatmap(ltv, grids, tag, loc, k, vTime)

        # renters
        plot_renter_savings(renter, grids, tag)

        # 3) moment paths
        for loc in ("C", "NC"):
            for var in ("l", "m", "h"):
                plot_moment_paths(cond, grids, tag, loc, var, vTime)

        # 5) default rates
        plot_default_rates(masses, defaults, tag, vTime)

    plt.show()


if __name__ == "__main__":
    plot_all_distributions()