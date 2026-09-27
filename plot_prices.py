# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:04:17 2026

@author: tprins
"""
import matplotlib.pyplot as plt
import numpy as np
import LoM_epsilons as lom
import par_epsilons as parfile
import grid_creation as grid_creation
import misc_functions as misc
import coeff_io
"""
Inputs: coefficienten voor: (1) initial steady state, (2) baseline transition with sceptics, (3) baseline transition without sceptics (RE), and three sets of policy coefficients for (4) full information shock, (5) building restrictions, (6) mortgage premium

outputs: four distributions over individual state variables over time for the above 5 cases: owners C, owners NC, renters C, renters NC. Perhaps collapsed on the G, J dimensions (if we're not interested in this source of heterogeneity)
"""
def plot_pricepaths(
    par,
    grids,
    vCoeff_C_initial,
    vCoeff_NC_initial,
    vCoeff_C,
    vCoeff_NC,
    vCoeff_C_RE,
    vCoeff_NC_RE,
    vCoeff_C_terminal_RE,
    vCoeff_NC_terminal_RE,
    vCoeff_C_terminal_HE,
    vCoeff_NC_terminal_HE,
):
    normalisation = vCoeff_NC_initial[0]

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(7.0, 4.6))

    T = len(grids.vTime)
    t_indices = np.arange(T)
    years = par.starting_year + par.time_increment * t_indices

    yC_RE = np.array([
        lom.LoM(par, grids, t_index, vCoeff_C_RE)
        for t_index in t_indices
    ]) / normalisation

    yNC_RE = np.array([
        lom.LoM(par, grids, t_index, vCoeff_NC_RE)
        for t_index in t_indices
    ]) / normalisation

    yC_HE = np.array([
        lom.LoM(par, grids, t_index, vCoeff_C)
        for t_index in t_indices
    ]) / normalisation

    yNC_HE = np.array([
        lom.LoM(par, grids, t_index, vCoeff_NC)
        for t_index in t_indices
    ]) / normalisation

    lineC, = plt.plot(years, yC_RE, linestyle=":", linewidth=2)
    lineNC, = plt.plot(years, yNC_RE, linestyle=":", linewidth=2)

    plt.plot(
        years,
        yC_HE,
        linewidth=2,
        color=lineC.get_color(),
        label="Flood-exposed price trajectory",
    )

    plt.plot(
        years,
        yNC_HE,
        linewidth=2,
        color=lineNC.get_color(),
        label="Inland price trajectory",
    )

    # Initial prices
    x0_year = years[0]
    y_coastal = lom.LoM(par, grids, 0, vCoeff_C_initial) / normalisation
    y_inland = lom.LoM(par, grids, 0, vCoeff_NC_initial) / normalisation

    plt.scatter([x0_year], [y_coastal], zorder=5)
    plt.scatter([x0_year], [y_inland], zorder=5)

    #plt.annotate(
    #    "Initial flood-exposed price",
    #    (x0_year, y_coastal),
    #    xytext=(10, 10),
    #    textcoords="offset points",
    #    ha="center",
    #    fontsize=9,
    #)

    #plt.annotate(
    #    "Initial inland price",
    #    (x0_year, y_inland),
    #    xytext=(10, 10),
    #    textcoords="offset points",
    #    ha="center",
    #    fontsize=9,
    #)

    # Terminal prices
    xT_year = years[-1]

    yC_terminal_HE = (
        lom.LoM(par, grids, T - 1, vCoeff_C_terminal_HE)
        / normalisation
    )

    yNC_terminal_HE = (
        lom.LoM(par, grids, T - 1, vCoeff_NC_terminal_HE)
        / normalisation
    )

    plt.scatter(
        [xT_year],
        [yC_terminal_HE],
        color=lineC.get_color(),
        zorder=5,
    )

    plt.scatter(
        [xT_year],
        [yNC_terminal_HE],
        color=lineNC.get_color(),
        zorder=5,
    )

    #plt.annotate(
    #    "Terminal flood-exposed price",
    #    (xT_year, yC_terminal_HE),
    #    xytext=(-10, 10),
    #    textcoords="offset points",
    #    ha="right",
    #    fontsize=9,
    #)

    #plt.annotate(
    #    "Terminal inland price",
    #    (xT_year, yNC_terminal_HE),
    #    xytext=(-10, 10),
    #    textcoords="offset points",
    #    ha="right",
    #    fontsize=9,
    #)

    plt.vlines(x0_year, y_coastal, yC_HE[0],
               linestyles="dotted", linewidth=1)
    plt.vlines(x0_year, y_inland, yNC_HE[0],
               linestyles="dotted", linewidth=1)

    xticks = np.arange(int(years[0]), int(years[-1]) + 1, 20)
    plt.xticks(xticks)

    plt.xlabel("Year", fontsize=18)
    plt.ylabel("Price", fontsize=18)
    plt.title("House price trajectories", fontsize=18)
    plt.tick_params(labelsize=16)
    plt.legend(frameon=False, fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()
    
def plot_price_transition_exp(
        vCoeff_C_initial,
        vCoeff_NC_initial,
        vCoeff_C_baseline,
        vCoeff_NC_baseline,
        vCoeff_C_experiment,
        vCoeff_NC_experiment,
        par,
        grids,
        title,
        switch_index=14,
    ):
        T = len(grids.vTime)
        t_indices = np.arange(T)
        years = par.starting_year + par.time_increment * t_indices

        P_C_base = np.array([
            lom.LoM(par, grids, t_index, vCoeff_C_baseline)
            for t_index in t_indices
        ])

        P_NC_base = np.array([
            lom.LoM(par, grids, t_index, vCoeff_NC_baseline)
            for t_index in t_indices
        ])

        P_C_exp = np.array([
            lom.LoM(par, grids, t_index, vCoeff_C_experiment)
            for t_index in t_indices
        ])

        P_NC_exp = np.array([
            lom.LoM(par, grids, t_index, vCoeff_NC_experiment)
            for t_index in t_indices
        ])

        P_C_initial = lom.LoM(par, grids, 0, vCoeff_C_initial)
        P_NC_initial = lom.LoM(par, grids, 0, vCoeff_NC_initial)

        year0 = years[0]
        switch_year = years[switch_index]
        final_year = years[-1]

        fig, ax = plt.subplots(figsize=(10, 6))

        color_C = "tab:blue"
        color_NC = "tab:orange"

        # Flood-exposed C
        ax.scatter(year0, P_C_initial, marker="o", color=color_C)
        ax.scatter(year0, P_C_base[0], marker="o", color=color_C)
        ax.plot([year0, year0], [P_C_initial, P_C_base[0]],
                linestyle="--", color=color_C)

        ax.plot(years[:switch_index + 1],
                P_C_base[:switch_index + 1],
                color=color_C,
                label="Flood-exposed")

        ax.scatter(switch_year, P_C_base[switch_index],
                   marker="o", color=color_C)
        ax.scatter(switch_year, P_C_exp[switch_index],
                   marker="o", color=color_C)

        ax.plot([switch_year, switch_year],
                [P_C_base[switch_index], P_C_exp[switch_index]],
                linestyle="--", color=color_C)

        ax.plot(years[switch_index:],
                P_C_exp[switch_index:],
                color=color_C)

        ax.scatter(final_year, P_C_exp[-1], marker="o", color=color_C)

        # Non-flood-exposed NC
        ax.scatter(year0, P_NC_initial, marker="s", color=color_NC)
        ax.scatter(year0, P_NC_base[0], marker="s", color=color_NC)
        ax.plot([year0, year0], [P_NC_initial, P_NC_base[0]],
                linestyle="--", color=color_NC)

        ax.plot(years[:switch_index + 1],
                P_NC_base[:switch_index + 1],
                color=color_NC,
                label="Non-flood-exposed")

        ax.scatter(switch_year, P_NC_base[switch_index],
                   marker="s", color=color_NC)
        ax.scatter(switch_year, P_NC_exp[switch_index],
                   marker="s", color=color_NC)

        ax.plot([switch_year, switch_year],
                [P_NC_base[switch_index], P_NC_exp[switch_index]],
                linestyle="--", color=color_NC)

        ax.plot(years[switch_index:],
                P_NC_exp[switch_index:],
                color=color_NC)

        ax.scatter(final_year, P_NC_exp[-1], marker="s", color=color_NC)

        ax.axvline(switch_year, linestyle=":", linewidth=1, color="black")

        ax.set_xlabel("Year")
        ax.set_ylabel("Price")
        ax.set_title(title)
        ax.legend()
        ax.set_ylim(0.4, 0.85)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        
def rental_price(par, grids, t_index, P, P_prime, coastal):
    """User-cost rent given this period's price P and next period's price P_prime."""
    dmg = grids.vPi_S_median[t_index] * np.dot(grids.vPDF_z[1:], 1 - grids.vZ[1:]) if coastal else 0.0
    return par.dPsi + max(P - (1 - par.dDelta - dmg) / (1 + par.r) * P_prime, 0)


def rent_path(par, grids, vCoeff, coastal):
    """Rents along the transition; price is held constant in the last period (P' = P)."""
    T = len(grids.vTime)
    P = [lom.LoM(par, grids, t, vCoeff) for t in range(T)]
    P.append(P[-1])
    return np.array([rental_price(par, grids, t, P[t], P[t + 1], coastal) for t in range(T)])

def plot_rentpaths(par, grids, vCoeff_C_initial, vCoeff_NC_initial, vCoeff_C, vCoeff_NC,
                   vCoeff_C_RE, vCoeff_NC_RE, vCoeff_C_terminal_RE, vCoeff_NC_terminal_RE,
                   vCoeff_C_terminal_HE, vCoeff_NC_terminal_HE, end_year=2100):
    
    P0_NC = lom.LoM(par, grids, 0, vCoeff_NC_initial)
    normalisation = rental_price(par, grids, 0, P0_NC, P0_NC, False)   # initial inland rent                 # same as price graph
    T = len(grids.vTime)
    years = par.starting_year + par.time_increment * np.arange(T)
    n = int((end_year - par.starting_year) / par.time_increment) + 1   # periods up to end_year
    years = years[:n]
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(7.0, 4.6))

    specs = [  # (coastal, HE, HE terminal, RE, RE terminal, initial, name)
        (True,  vCoeff_C,  vCoeff_C_terminal_HE,  vCoeff_C_RE,  vCoeff_C_terminal_RE,  vCoeff_C_initial,  "flood-exposed"),
        (False, vCoeff_NC, vCoeff_NC_terminal_HE, vCoeff_NC_RE, vCoeff_NC_terminal_RE, vCoeff_NC_initial, "inland"),
    ]
    for coastal, cHE, cHE_T, cRE, cRE_T, c0, name in specs:
        rHE = rent_path(par, grids, cHE, coastal)[:n] / normalisation
        rRE = rent_path(par, grids, cRE, coastal)[:n] / normalisation
        P0 = lom.LoM(par, grids, 0, c0)                       # initial steady state: P' = P
        PT = lom.LoM(par, grids, T - 1, cHE_T)                # terminal steady state: P' = P
        r0 = rental_price(par, grids, 0, P0, P0, coastal) / normalisation
        rT = rental_price(par, grids, T - 1, PT, PT, coastal) / normalisation

        line, = plt.plot(years, rRE, linestyle=":", linewidth=2)
        col = line.get_color()
        plt.plot(years, rHE, linewidth=2, color=col, label=f"{name.capitalize()} rent trajectory")

        plt.scatter([years[0]], [r0], color=col, zorder=5)
        #plt.annotate(f"Initial {name} rent", (years[0], r0), xytext=(10, 10),
        #             textcoords="offset points", ha="center", fontsize=9)
        plt.vlines(years[0], r0, rHE[0], linestyles="dotted", linewidth=1)

        plt.scatter([years[-1]], [rT], color=col, zorder=5)
        #plt.annotate(f"Terminal {name} rent", (years[-1], rT), xytext=(-10, 10),
        #             textcoords="offset points", ha="right", fontsize=9)

    plt.xticks(np.arange(int(years[0]), int(years[-1]) + 1, 20))
    plt.xlabel("Year", fontsize=18)
    plt.ylabel("Rent", fontsize=18)
    plt.title("Rental price trajectories", fontsize=18)
    plt.tick_params(labelsize=16)
    plt.legend(frameon=False, fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    """############################################################################
    ### Plot price paths
    ############################################################################"""
    par = misc.construct_jitclass(parfile.par_dict)
    grids = grid_creation.create(par)

    # load solved coefficients by label (falls back to {} if file absent)
    C = coeff_io.load_coefficients()

    ### without experiments
    plot_pricepaths(
        par,
        grids,
        C["vCoeff_C_initial_HE"],
        C["vCoeff_NC_initial_HE"],
        C["vCoeff_C_HE"],
        C["vCoeff_NC_HE"],
        C["vCoeff_C_RE"],
        C["vCoeff_NC_RE"],
        C["vCoeff_C_terminal_RE"],
        C["vCoeff_NC_terminal_RE"],
        C["vCoeff_C_terminal_HE"],
        C["vCoeff_NC_terminal_HE"],
    )

    # Mortgage premium experiment
    plot_price_transition_exp(
        C["vCoeff_C_initial_HE"],
        C["vCoeff_NC_initial_HE"],
        C["vCoeff_C_HE"],
        C["vCoeff_NC_HE"],
        C["vCoeff_C_MP"],
        C["vCoeff_NC_MP"],
        par,
        grids,
        title="House price transition: mortgage premium experiment",
        switch_index=14,
    )

    # Building restriction experiment
    plot_price_transition_exp(
        C["vCoeff_C_initial_HE"],
        C["vCoeff_NC_initial_HE"],
        C["vCoeff_C_HE"],
        C["vCoeff_NC_HE"],
        C["vCoeff_C_BR"],
        C["vCoeff_NC_BR"],
        par,
        grids,
        title="House price transition: building restriction experiment",
        switch_index=14,
    )
    
    plot_rentpaths(par, grids,
       C["vCoeff_C_initial_HE"], C["vCoeff_NC_initial_HE"],
       C["vCoeff_C_HE"], C["vCoeff_NC_HE"],
       C["vCoeff_C_RE"], C["vCoeff_NC_RE"],
       C["vCoeff_C_terminal_RE"], C["vCoeff_NC_terminal_RE"],
       C["vCoeff_C_terminal_HE"], C["vCoeff_NC_terminal_HE"])