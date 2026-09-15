"""
collect_results.py
"""

###########################################################
### Imports
import numpy as np
from numba import njit
import equilibrium as equil
import proper_welfare_debug as welfare_stats
import plot_creation as plot_creat
import household_problem_epsilons_nolearning as household_problem
import simulation as sim
import equilibrium as equil
import LoM_epsilons as lom
import proper_welfare_debug as welfare_stats
import pandas as pd
import simulate_initial_joint as initial_joint_sim


def collect_results(par, grids, vCoeff_C_initial_HE, vCoeff_NC_initial_HE, vCoeff_C, vCoeff_NC, vCoeff_C_RE, vCoeff_NC_RE, vCoeff_C_MortPrem, vCoeff_NC_MortPrem, vCoeff_C_BuildRest, vCoeff_NC_BuildRest, solve_initial_ss_HE, solve_initial_ss_RE, path_until_experiment,transition_path, transition_path_RE, experiment_building_rest, experiment_mortgage_prem,vCoeff_C_terminal_RE,vCoeff_NC_terminal_RE, vCoeff_C_terminal_HE, vCoeff_NC_terminal_HE, calculate_welfare):
    
    """
    Inputs: coefficienten voor: (1) initial steady state, (2) baseline transition with sceptics, (3) baseline transition without sceptics (RE), and three sets of policy coefficients for (4) full information shock, (5) building restrictions, (6) mortgage premium
    
    outputs: four distributions over individual state variables over time for the above 5 cases: owners C, owners NC, renters C, renters NC. Perhaps collapsed on the G, J dimensions (if we're not interested in this source of heterogeneity)
    """
    """############################################################################
    ### Plot price paths
    ############################################################################"""
    ### without experiments
    plot_creat.plot_pricepaths(
    par,
    grids,
    vCoeff_C_initial_HE,
    vCoeff_NC_initial_HE,
    vCoeff_C,
    vCoeff_NC,
    vCoeff_C_RE,
    vCoeff_NC_RE,
    vCoeff_C_terminal_RE,
    vCoeff_NC_terminal_RE,
    vCoeff_C_terminal_HE,
    vCoeff_NC_terminal_HE,
)
    # Mortgage premium experiment
    plot_creat.plot_price_transition_exp(
        vCoeff_C_initial_HE,
        vCoeff_NC_initial_HE,
        vCoeff_C,
        vCoeff_NC,
        vCoeff_C_MortPrem,
        vCoeff_NC_MortPrem,
        par,
        grids,
        title="House price transition: mortgage premium experiment",
        switch_index=14,
    )
    
    # Building restriction experiment
    plot_creat.plot_price_transition_exp(
        vCoeff_C_initial_HE,
        vCoeff_NC_initial_HE,
        vCoeff_C,
        vCoeff_NC,
        vCoeff_C_BuildRest,
        vCoeff_NC_BuildRest,
        par,
        grids,
        title="House price transition: building restriction experiment",
        switch_index=14,
    )
    
    """############################################################################
    ### BASELINE ECONOMY 
    ############################################################################"""
    

    "(1) initial steady state"
    
    dP_C_initial = lom.LoM(par,grids,0,vCoeff_C_initial_HE)
    dP_NC_initial = lom.LoM(par,grids,0,vCoeff_NC_initial_HE)

    vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter,_,_,_ = household_problem.solve_ss(grids, par, vCoeff_C_initial_HE[0], vCoeff_NC_initial_HE[0], solve_initial_ss_HE)
    mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0, no_beq=sim.stat_dist_finder(par, grids, vt_stay_c[0,], vt_stay_nc[0,], vt_renter[0,], b_stay_c[0,], b_stay_nc[0,], b_renter[0,], vCoeff_C_initial_HE,vCoeff_NC_initial_HE, solve_initial_ss_HE)

    "(2) baseline transition with sceptics"
    # run generate price path without experiments, with sceptics. save the (collapsed) distributions. Also save the 2026 distributions and welfare value functions
    _, _, _, _, _, _, vcoastal_beq, vnoncoastal_beq, vsavings_beq, _, _, _, v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf, full_dist_C_HE, full_dist_NC_HE, full_dist_renter_HE =equil.generate_pricepath(grids, par, vCoeff_C,vCoeff_NC, dP_C_initial, dP_NC_initial, mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0, transition_path)    
    print("diagnose tranistion path")
    diagnose_newborns(par, grids, vCoeff_C, vCoeff_NC, mDist1_renter_SS, vcoastal_beq, vnoncoastal_beq, vsavings_beq, transition_path)
    

    
    "(3) baseline transition without sceptics (RE)"
    # run generate price path without experiments, without sceptics. save the (collapsed) distributions and welfare value functions
    price_history_RE, _, _, _, _, _, vcoastal_beq_RE, vnoncoastal_beq_RE, vsavings_beq_RE, _, _, _, v_owner_c_wf_RE, v_owner_nc_wf_RE, v_nonowner_wf_RE, full_dist_C_RE, full_dist_NC_RE, full_dist_renter_RE = equil.generate_pricepath(grids, par, vCoeff_C_RE, vCoeff_NC_RE, dP_C_initial, dP_NC_initial, mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0, transition_path_RE)
    print("diagnose RE path")
    diagnose_newborns(par, grids, vCoeff_C_RE, vCoeff_NC_RE, mDist1_renter_SS, vcoastal_beq_RE, vnoncoastal_beq_RE, vsavings_beq_RE, transition_path_RE)
    # "Plots of distributions"
    # ### PLOT - Baseline model dynamics (leverage, savings, sorting, etc.) Use: full_dist_C_HE, full_dist_NC_HE, full_dist_renter_HE, full_dist_C_RE, full_dist_NC_RE, full_dist_renter_RE 
    
    del full_dist_C_HE, full_dist_NC_HE, full_dist_renter_HE, full_dist_C_RE, full_dist_NC_RE, full_dist_renter_RE
    
    if calculate_welfare:
        "WELFARE COSTS OF MISBELIEFS"
        tax_equiv_C_RE, tax_equiv_NC_RE, tax_equiv_renter_RE, tax_equiv_newborns_RE =  welfare_stats.find_expenditure_equiv_EK_SLR(par, grids, vCoeff_C_initial_HE, vCoeff_NC_initial_HE, vCoeff_C_RE, vCoeff_NC_RE, mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, vcoastal_beq_RE, vnoncoastal_beq_RE, vsavings_beq_RE, v_owner_c_wf_RE, v_owner_nc_wf_RE, v_nonowner_wf_RE, solve_initial_ss_RE, transition_path_RE)
        tax_equiv_C, tax_equiv_NC, tax_equiv_renter, tax_equiv_newborns             =  welfare_stats.find_expenditure_equiv_EK_SLR(par, grids, vCoeff_C_initial_HE, vCoeff_NC_initial_HE, vCoeff_C, vCoeff_NC, mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS,  vcoastal_beq, vnoncoastal_beq, vsavings_beq, v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf, solve_initial_ss_HE, transition_path)
    
    
    """############################################################################
    ### Plot 2026 dist
    ############################################################################"""
    # price_history, mDist1_c_2026, mDist1_nc_2026, mDist1_renter_2026, rental_stock_C_2026, rental_stock_NC_2026, _, _, _, _, _, _, _, _, _, _, _, _=equil.generate_pricepath(grids, par, vCoeff_C, vCoeff_NC, dP_C_initial, dP_NC_initial, mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0, path_until_experiment)
    #plot_creat.plot_distribution_2026(price_history, mDist1_c_2026, mDist1_nc_2026, mDist1_renter_2026, rental_stock_C_2026, rental_stock_NC_2026, vcoastal_beq, vnoncoastal_beq, vsavings_beq, vCoeff_C, vCoeff_NC)
    
    """############################################################################
    ### START EXPERIMENTS
    ############################################################################"""
    # if calculate_welfare:
        # "(4) + (5) welfare effects of policy: building restrictions and mortgage premium"
        # print("start with welfare of policy")
        # tax_equiv_C_MP, tax_equiv_NC_MP, tax_equiv_renter_MP, tax_equiv_newborns_MP,tax_equiv_C_BR, tax_equiv_NC_BR, tax_equiv_renter_BR, tax_equiv_newborns_BR = welfare_stats.find_expenditure_equiv_EK_policy(par, grids, vCoeff_C, vCoeff_NC,vCoeff_C_MortPrem, vCoeff_NC_MortPrem, vCoeff_C_BuildRest, vCoeff_NC_BuildRest, mDist1_c_2026, mDist1_nc_2026, mDist1_renter_2026, vcoastal_beq, vnoncoastal_beq, vsavings_beq, transition_path, experiment_mortgage_prem, experiment_building_rest)
     
    ### PLOT - WELFARE EFFECTS OF POLICIES (4), (5), COMPARED TO BASELINE IN (2)
    
    ### PLOT SORTING AFTER POLICY CHANGES: EXTENSIVE VERSUS INTENSIVE MARGIN PER TYPE, INCOME
    return tax_equiv_C_RE, tax_equiv_NC_RE, tax_equiv_renter_RE, tax_equiv_newborns_RE, tax_equiv_C, tax_equiv_NC, tax_equiv_renter, tax_equiv_newborns#, tax_equiv_C_BR, tax_equiv_NC_BR, tax_equiv_renter_BR, tax_equiv_newborns_BR , tax_equiv_C_MP, tax_equiv_NC_MP, tax_equiv_renter_MP, tax_equiv_newborns_MP


def diagnose_newborns(par, grids, vCoeff_C_in, vCoeff_NC_in, mDist1_renter_SS,
                      vcoastal_beq, vnoncoastal_beq, vsavings_beq, config):
 
    dP_C_lom = lom.LoM_path(par, grids, vCoeff_C_in, config)
    dP_NC_lom = lom.LoM_path(par, grids, vCoeff_NC_in, config)
    k_dim = 1 if config.sceptics == False else grids.vK.size
    k_dim_SS = mDist1_renter_SS.shape[1]
 
    print('=' * 78)
    print('config.sceptics =', config.sceptics, '| k_dim used =', k_dim,
          '| k_dim of mDist1_renter_SS =', k_dim_SS)
    if k_dim != k_dim_SS:
        print('  >>> MISMATCH: the welfare fn only reads mDist1_renter_SS[0,0,...],')
        print('  >>> i.e. the initially-REALIST slice, but gen_initial_dist returns')
        print('  >>> k_weight = 1.0. Masses differ by 1/mTypes[0,0].')
 
    print('\n--- mTypes (belief shares) by t ---')
    for t in range(0, grids.vTime.size, 2):
        row = '  t=%3d  ' % t
        for k in range(grids.mTypes.shape[1]):
            row += 'mTypes[%d]=%.6f  ' % (k, grids.mTypes[t, k])
        print(row)
 
    print('\n--- SS newborn mass vs transition newborn mass, by (t,k) ---')
    print('  if the ratio is not ~1.0 for all t, the welfare comparison is')
    print('  contaminated by a population-share change, not by welfare.')
    mass_SS = np.zeros(k_dim)
    for k in range(k_dim):
        if k_dim == k_dim_SS:
            mass_SS[k] = np.sum(mDist1_renter_SS[0, k])
        else:
            mass_SS[k] = np.sum(mDist1_renter_SS[0])   # collapse over beliefs
 
    beq_hist = np.zeros(grids.vTime.size - 1)
    for t in range(grids.vTime.size - 1):
        nd = sim.gen_initial_dist(par, grids, t, dP_C_lom[t], dP_NC_lom[t],
                                  vcoastal_beq[t], vnoncoastal_beq[t],
                                  vsavings_beq[t], config.sceptics)
        # reproduce the bequest that drives the newborn x-distribution
        coastal_damage_frac = grids.vPi_S_median[t] * np.dot(grids.vPDF_z[1:], (1 - grids.vZ[1:]))
        housing_bequest = (vcoastal_beq[t] * (1 - coastal_damage_frac - par.dDelta) * dP_C_lom[t]
                           + vnoncoastal_beq[t] * (1 - par.dDelta) * dP_NC_lom[t])
        beq_hist[t] = (housing_bequest + vsavings_beq[t] * (1 + par.r)) * par.iNj
 
        line = '  t=%3d  total_beq=%.6f  vPi_S_median=%.6f  ' % (
            t, beq_hist[t], grids.vPi_S_median[t])
        for k in range(k_dim):
            mt = np.sum(nd[k])
            ratio = mt / mass_SS[k] if mass_SS[k] > 1e-14 else np.nan
            line += '| k=%d mass=%.3e ratio=%.4f ' % (k, mt, ratio)
        print(line)
 
    print('\n--- jumps in total_bequest (candidate cause of the kink) ---')
    d = np.abs(np.diff(beq_hist))
    med = np.median(d[d > 0]) if np.any(d > 0) else 0.0
    for t in range(d.size):
        if med > 0 and d[t] > 5 * med:
            print('  JUMP between t=%d and t=%d: %.6f -> %.6f (%.1fx median step)'
                  % (t, t + 1, beq_hist[t], beq_hist[t + 1], d[t] / med))
 
    print('\n--- newborn x-distribution: is initial_joint interpolating or snapping? ---')
    print('  if the support jumps by a whole grid point as total_beq moves')
    print('  smoothly, initial_joint is doing nearest-node placement.')
    prev_support = None
    for t in range(grids.vTime.size - 1):
        mPi = initial_joint_sim.initial_joint(par, grids, beq_hist[t])
        support = tuple(np.where(np.sum(mPi, axis=1) > 1e-12)[0])
        if support != prev_support:
            print('  t=%3d  x-support changes to %s  (total_beq=%.6f)'
                  % (t, support, beq_hist[t]))
            prev_support = support
    print('=' * 78)

def tax_equiv_to_long(
    array,
    scenario,
    household,
    grids,
    par,
):
    """
    Convert either a (K, E) or (T, K, E) tax-equivalent array
    to a tidy pandas DataFrame.
    """
    array = np.asarray(array)

    k_values = np.asarray(grids.vK)
    e_values = np.asarray(grids.vE)

    if array.ndim == 2:
        # Array dimensions: K x E
        k_dim, e_dim = array.shape

        if e_dim != e_values.size:
            raise ValueError(
                f"{scenario}/{household}: E dimension is {e_dim}, "
                f"but grids.vE has length {e_values.size}."
            )

        k_index, e_index = np.indices(array.shape)

        df = pd.DataFrame(
            {
                "scenario": scenario,
                "household": household,
                "t_index": pd.NA,
                "time": np.nan,
                "year": np.nan,
                "k_index": k_index.ravel(),
                "e_index": e_index.ravel(),
                "tax_equiv": array.ravel(),
            }
        )

        df["k_value"] = k_values[df["k_index"].to_numpy()]
        df["e_value"] = e_values[df["e_index"].to_numpy()]

    elif array.ndim == 3:
        # Array dimensions: T x K x E
        t_dim, k_dim, e_dim = array.shape

        if t_dim != grids.vTime.size:
            raise ValueError(
                f"{scenario}/{household}: T dimension is {t_dim}, "
                f"but grids.vTime has length {grids.vTime.size}."
            )

        if e_dim != e_values.size:
            raise ValueError(
                f"{scenario}/{household}: E dimension is {e_dim}, "
                f"but grids.vE has length {e_values.size}."
            )

        t_index, k_index, e_index = np.indices(array.shape)

        df = pd.DataFrame(
            {
                "scenario": scenario,
                "household": household,
                "t_index": t_index.ravel(),
                "k_index": k_index.ravel(),
                "e_index": e_index.ravel(),
                "tax_equiv": array.ravel(),
            }
        )

        df["time"] = np.asarray(grids.vTime)[
            df["t_index"].to_numpy()
        ]
        
        df["year"] = (
            par.starting_year
            + df["time"].to_numpy() * par.time_increment
        )

        df["k_value"] = k_values[df["k_index"].to_numpy()]
        df["e_value"] = e_values[df["e_index"].to_numpy()]

    else:
        raise ValueError(
            f"Expected a 2D or 3D array, received shape {array.shape}."
        )

    return df[
        [
            "scenario",
            "household",
            "t_index",
            "time",
            "year",
            "k_index",
            "k_value",
            "e_index",
            "e_value",
            "tax_equiv",
        ]
    ]


