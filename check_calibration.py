
import numpy as np
import misc_functions as misc
import grid_creation as grid_creation
import household_problem_epsilons_nolearning as household_problem
import simulation as sim
import moments as find_moments
import par_epsilons as parfile
import experiment_config
import unpack_configs
import coeff_io
import LoM_epsilons as lom




def calibration_check():
    par = misc.construct_jitclass(parfile.par_dict)
    grids=grid_creation.create(par)
    cfg = unpack_configs.unpack(experiment_config, misc)
    C = coeff_io.load_with_defaults()
    vCoeff_C=C["vCoeff_C_initial_HE"]
    vCoeff_NC=C["vCoeff_NC_initial_HE"]
    dP_C_lom = lom.LoM(par,grids,0,C["vCoeff_C_initial_HE"])
    dP_NC_lom = lom.LoM(par,grids,0,C["vCoeff_NC_initial_HE"])
    vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter,_,_,_ = household_problem.solve_ss(grids, par, C["vCoeff_C_initial_HE"][0], C["vCoeff_NC_initial_HE"][0], cfg.solve_initial_ss_HE)
    mDist1_c, mDist1_nc, mDist1_renter, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0, no_beq=sim.stat_dist_finder(par, grids, vt_stay_c[0,], vt_stay_nc[0,], vt_renter[0,], b_stay_c[0,], b_stay_nc[0,], b_renter[0,], C["vCoeff_C_initial_HE"],C["vCoeff_NC_initial_HE"], cfg.solve_initial_ss_HE)

    
    # MODEL MOMENTS
    HO_C_share, HO_NC_share, R_C_share, R_NC_share, HO_C_share_before35, HO_NC_share_before35, HO_C_share_death, HO_NC_share_death, total_NW_HO_C, total_NW_HO_NC, total_NW_R, total_NW_HO, total_NW_age_15, total_NW_age_27, total_NW_all_ages, median_NW_age_15, median_NW_age_27, median_NW_all_ages, thirtythree_percentile_NW_age_27, sixtyseven_percentile_NW_age_27, thirtythree_percentile_NW_age_30, sixtyseven_percentile_NW_age_30, tenth_percentile_housing, median_housing, ninetieth_percentile_housing, cumdens_housing_all_ages, NW_housing_share_sorted=find_moments.calc_moments(par, grids, 0, mDist1_c, mDist1_nc,mDist1_renter,  vCoeff_C, vCoeff_NC)
    total_saving_model = median_NW_all_ages
    NW_decay_model = total_NW_age_27/total_NW_age_15
    bequest_ineq_model = sixtyseven_percentile_NW_age_30/thirtythree_percentile_NW_age_30
    homeownership_model = HO_C_share+HO_NC_share
    price_diff_model = (dP_C_lom-dP_NC_lom)/dP_NC_lom
    homeownership_young_model = HO_C_share_before35 + HO_NC_share_before35
    med_housing_model = median_housing
    
  
    
    # DATA MOMENTS
    total_saving_data = 1.2
    NW_decay_data = 1.51
    bequest_ineq_data = 3.24
    homeownership = 0.66
    price_diff = -0.114
    homeownership_young = 0.39
    med_housing = 0.5
    
    print('total_saving_model', total_saving_model, 'data:' , total_saving_data)
    print('NW_decay_data', NW_decay_model, 'data:', NW_decay_data)
    print('bequest_ineq_data', bequest_ineq_model, 'data:', bequest_ineq_data)
    print('homeownership', homeownership_model, 'data:', homeownership)
    print('total_saving_model', price_diff_model, 'data:', price_diff)
    print('homeownership_young_model', homeownership_young_model, 'data:', homeownership_young)
    print('med_housing_model', med_housing_model, 'data:', med_housing)
    

    
    sq_saving       = ((total_saving_data-total_saving_model)/total_saving_data)**2
    sq_nw           = ((NW_decay_data-NW_decay_model)/NW_decay_data)**2
    sq_ineqnw       = ((bequest_ineq_data-bequest_ineq_model)/bequest_ineq_data)**2
    sq_homeownership = ((homeownership - homeownership_model)/homeownership)**2
    sq_price_diff     = ((price_diff - price_diff_model)/price_diff)**2
    sq_homeownership_young = ((homeownership_young - homeownership_young_model)/homeownership_young)**2
    sq_med_housing    = ((med_housing - med_housing_model)/med_housing)**2
    weights = np.array([1,1,1,1.5,1,1,1])
    
    squaredsum =  weights[0]*sq_saving + weights[1]*sq_nw + weights[2]*sq_ineqnw + weights[3]*sq_homeownership + weights[4]*sq_price_diff +  weights[5]*sq_homeownership_young + weights[6]*sq_med_housing 
    
    return squaredsum

def main():
    squared_sum=calibration_check()

    
    
###########################################################
### start main
if __name__ == "__main__":
    main()