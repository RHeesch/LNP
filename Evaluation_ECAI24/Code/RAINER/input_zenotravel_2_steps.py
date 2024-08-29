import pandas as pd

integrate_ml = True        # enables the integration of ML-models
stepnumber_known = False
stepnumber = 6
approx = True               # enables the approximation of the results
#this is necessary, if small ML models are integrated, due to their inaccuracy 
cc = False

# Table of fix parameters
Parameter_fix = ["dis_0_1", "dis_0_2", "dis_1_2", "slow_burn_a1", "fast_burn_a1", "capacity_a1", "zoom_limit_a1"]
Values_fix = pd.Series([6.0, 7.0, 8.0, 4.0, 1.50, 100.0, 8.0], index = Parameter_fix) 
para_fix = pd.concat([Values_fix], axis = 1)
limits_fix = ['Values_fix']
para_fix.columns = limits_fix

# Table of variable parameters
Parameter_var = ["fuel"]
Top_var = pd.Series([1000], index = Parameter_var)
Down_var = pd.Series([-1000], index = Parameter_var)
para_var = pd.concat([Top_var, Down_var], axis = 1)
limits_var = ["Top_var", "Down_var"]
para_var.columns = limits_var

Variables = ["location_p1","location_p2", "location_p3", "in_p1", "in_p2", "in_p3", "location_a1", "fuel_a1", "onboard_a1", "total_fuel_used"]
sym_actions = ["board_a1_p1_c0", "board_a1_p1_c1", "board_a1_p1_c2", "board_a1_p2c0", "board_a1_p2_c1", "board_a1_p2_c2", "board_a1_p3_c0", "board_a1_p3_c1", "board_a1_p3_c2",
               "debark_a1_p1_c0", "debark_a1_p1_c1", "debark_a1_p1_c2", "debark_a1_p2_c0", "debark_a1_p2_c1", "debark_a1_p2_c2", "debark_a1_p3_c0", "debark_a1_p3_c1", "debark_a1_p3_c2",
               "fs_c0_c1", "fs_c0_c2", "fs_c1_c2", "fs_c1_c0", "fs_c2_c0", "fs_c2_c1", 
               "ff_c0_c1", "ff_c0_c2", "ff_c1_c2", "ff_c1_c0", "ff_c2_c0", "ff_c2_c1",
               "refuel"]
learn_actions = ["L_Refuel"]

# Table of preconditions of symbolic actions
# Variables = ["location_p1","location_p2", "location_p3", "in_p1", "in_p2", "in_p3", "location_a1", "fuel_a1", "onboard_a1", "total_fuel_used"]
board_a1_p1_c0_pre = pd.Series(["0.0", ">=0.0", ">=0.0", "0.0", ">=0.0", ">=0.0", "0.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
board_a1_p1_c1_pre = pd.Series(["10.0", ">=0.0", ">=0.0", "0.0", ">=0.0", ">=0.0", "10.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
board_a1_p1_c2_pre = pd.Series(["20.0", ">=0.0", ">=0.0", "0.0", ">=0.0", ">=0.0", "20.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
board_a1_p2_c0_pre = pd.Series([">=0.0", "00.0", ">=0.0", ">=0.0", "0.0", ">=0.0", "0.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
board_a1_p2_c1_pre = pd.Series([">=0.0", "10.0", ">=0.0", ">=0.0", "0.0", ">=0.0", "10.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
board_a1_p2_c2_pre = pd.Series([">=0.0", "20.0", ">=0.0", ">=0.0", "0.0", ">=0.0", "20.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
board_a1_p3_c0_pre = pd.Series([">=0.0", ">=0.0", "0.0", ">=0.0", ">=0.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
board_a1_p3_c1_pre = pd.Series([">=0.0", ">=0.0", "10.0", ">=0.0", ">=0.0", "0.0", "10.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
board_a1_p3_c2_pre = pd.Series([">=0.0", ">=0.0", "20.0", ">=0.0", ">=0.0", "0.0", "20.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
debark_a1_p1_c0_pre = pd.Series([">=0.0", ">=0.0", ">=0.0", ">=10.0", ">=0.0", ">=0.0", "0.0", ">=0.0", ">=10.0", ">=0.0"], index=Variables)
debark_a1_p1_c1_pre = pd.Series([">=0.0", ">=0.0", ">=0.0", ">=10.0", ">=0.0", ">=0.0", "10.0", ">=0.0", ">=10.0", ">=0.0"], index=Variables)
debark_a1_p1_c2_pre = pd.Series([">=0.0", ">=0.0", ">=0.0", ">=10.0", ">=0.0", ">=0.0", "20.0", ">=0.0", ">=10.0", ">=0.0"], index=Variables)
debark_a1_p2_c0_pre = pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=10.0", ">=0.0", "0.0", ">=0.0", ">=10.0", ">=0.0"], index=Variables)
debark_a1_p2_c1_pre = pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=10.0", ">=0.0", "10.0", ">=0.0", ">=10.0", ">=0.0"], index=Variables)
debark_a1_p2_c2_pre = pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=10.0", ">=0.0", "20.0", ">=0.0", ">=10.0", ">=0.0"], index=Variables)
debark_a1_p3_c0_pre = pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=10.0", "0.0", ">=0.0", ">=10.0", ">=0.0"], index=Variables)
debark_a1_p3_c1_pre = pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=10.0", "10.0", ">=0.0", ">=10.0", ">=0.0"], index=Variables)
debark_a1_p3_c2_pre = pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=10.0", "20.0", ">=0.0", ">=10.0", ">=0.0"], index=Variables)
fs_c0_c1_pre = pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "0.0", ">=(dis_0_1*slow_burn_a1)", ">=0.0", ">=0.0"], index=Variables)
fs_c0_c2_pre = pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "0.0", ">=(dis_0_2*slow_burn_a1)", ">=0.0", ">=0.0"], index=Variables)
fs_c1_c2_pre = pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "10.0", ">=(dis_1_2*slow_burn_a1)", ">=0.0", ">=0.0"], index=Variables)
fs_c1_c0_pre = pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "10.0", ">=(dis_0_1*slow_burn_a1)", ">=0.0", ">=0.0"], index=Variables)
fs_c2_c0_pre = pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "20.0", ">=(dis_0_2*slow_burn_a1)", ">=0.0", ">=0.0"], index=Variables)
fs_c2_c1_pre = pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "20.0", ">=(dis_1_2*slow_burn_a1)", ">=0.0", ">=0.0"], index=Variables)
ff_c0_c1_pre = pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "0.0", ">=(dis_0_1*fast_burn_a1)", "<=zoom_limit_a1", ">=0.0"], index=Variables)
ff_c0_c2_pre = pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "0.0", ">=(dis_0_2*fast_burn_a1)", "<=zoom_limit_a1", ">=0.0"], index=Variables)
ff_c1_c2_pre = pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "10.0", ">=(dis_1_2*fast_burn_a1)", "<=zoom_limit_a1", ">=0.0"], index=Variables)
ff_c1_c0_pre = pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "10.0", ">=(dis_0_1*fast_burn_a1)", "<=zoom_limit_a1", ">=0.0"], index=Variables)
ff_c2_c0_pre = pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "20.0", ">=(dis_0_2*fast_burn_a1)", "<=zoom_limit_a1", ">=0.0"], index=Variables)
ff_c2_c1_pre = pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "20.0", ">=(dis_1_2*fast_burn_a1)", "<=zoom_limit_a1", ">=0.0"], index=Variables)
refuel_pre = pd.Series([">=3000.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "<capacity_a1", ">=0.0", ">=0.0"], index=Variables)
precons = pd.concat([board_a1_p1_c0_pre, board_a1_p1_c1_pre, board_a1_p1_c2_pre, board_a1_p2_c0_pre, board_a1_p2_c1_pre, board_a1_p2_c2_pre, board_a1_p3_c0_pre, board_a1_p3_c1_pre, board_a1_p3_c2_pre,
                     debark_a1_p1_c0_pre, debark_a1_p1_c1_pre, debark_a1_p1_c2_pre, debark_a1_p2_c0_pre, debark_a1_p2_c1_pre, debark_a1_p2_c2_pre, debark_a1_p3_c0_pre, debark_a1_p3_c1_pre, debark_a1_p3_c2_pre,
                     fs_c0_c1_pre, fs_c0_c2_pre, fs_c1_c2_pre, fs_c1_c0_pre, fs_c2_c0_pre, fs_c2_c1_pre,
                     ff_c0_c1_pre, ff_c0_c2_pre, ff_c1_c2_pre, ff_c1_c0_pre, ff_c2_c0_pre, ff_c2_c1_pre,
                     refuel_pre], axis = 1)
precons.columns = sym_actions

# Table of symbolic effects
# Variables = ["location_p1","location_p2", "location_p3", "in_p1", "in_p2", "in_p3", "location_a1", "fuel_a1", "onboard_a1", "total_fuel_used"]
board_a1_p1_c0_eff = pd.Series(["-0.0", "0.0", "0.0", "10.0", "0.0", "0.0", "0.0", "0.0", "10.0", "0.0"], index=Variables)
board_a1_p1_c1_eff = pd.Series(["-10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "0.0", "0.0", "10.0", "0.0"], index=Variables)
board_a1_p1_c2_eff = pd.Series(["-20.0", "0.0", "0.0", "10.0", "0.0", "0.0", "0.0", "0.0", "10.0", "0.0"], index=Variables)
board_a1_p2_c0_eff = pd.Series(["0.0", "-0.0", "0.0", "0.0", "10.0", "0.0", "0.0", "0.0", "10.0", "0.0"], index=Variables)
board_a1_p2_c1_eff = pd.Series(["0.0", "-10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "0.0", "10.0", "0.0"], index=Variables)
board_a1_p2_c2_eff = pd.Series(["0.0", "-20.0", "0.0", "0.0", "10.0", "0.0", "0.0", "0.0", "10.0", "0.0"], index=Variables)
board_a1_p3_c0_eff = pd.Series(["0.0", "0.0", "-0.0", "0.0", "0.0", "10.0", "0.0", "0.0", "10.0", "0.0"], index=Variables)
board_a1_p3_c1_eff = pd.Series(["0.0", "0.0", "-10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "10.0", "0.0"], index=Variables)
board_a1_p3_c2_eff = pd.Series(["0.0", "0.0", "-20.0", "0.0", "0.0", "10.0", "0.0", "0.0", "10.0", "0.0"], index=Variables)
debark_a1_p1_c0_eff = pd.Series(["0.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "0.0", "-10.0", "0.0"], index=Variables)
debark_a1_p1_c1_eff = pd.Series(["10.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "0.0", "-10.0", "0.0"], index=Variables)
debark_a1_p1_c2_eff = pd.Series(["20.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "0.0", "-10.0", "0.0"], index=Variables)
debark_a1_p2_c0_eff = pd.Series(["0.0", "0.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "-10.0", "0.0"], index=Variables)
debark_a1_p2_c1_eff = pd.Series(["0.0", "10.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "-10.0", "0.0"], index=Variables)
debark_a1_p2_c2_eff = pd.Series(["0.0", "20.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "-10.0", "0.0"], index=Variables)
debark_a1_p3_c0_eff = pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "-10.0", "0.0"], index=Variables)
debark_a1_p3_c1_eff = pd.Series(["0.0", "0.0", "10.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "-10.0", "0.0"], index=Variables)
debark_a1_p3_c2_eff = pd.Series(["0.0", "0.0", "20.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "-10.0", "0.0"], index=Variables)
fs_c0_c1_eff = pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "10.0", "-(dis_0_1*slow_burn_a1)", "0.0", "(dis_0_1*slow_burn_a1)"], index=Variables)
fs_c0_c2_eff = pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "20.0", "-(dis_0_2*slow_burn_a1)", "0.0", "(dis_0_2*slow_burn_a1)"], index=Variables)
fs_c1_c2_eff = pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "10.0", "-(dis_1_2*slow_burn_a1)", "0.0", "(dis_1_2*slow_burn_a1)"], index=Variables)
fs_c1_c0_eff = pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "-10.0", "-(dis_0_1*slow_burn_a1)", "0.0", "(dis_0_1*slow_burn_a1)"], index=Variables)
fs_c2_c0_eff = pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "-20.0", "-(dis_0_2*slow_burn_a1)", "0.0", "(dis_0_2*slow_burn_a1)"], index=Variables)
fs_c2_c1_eff = pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "-10.0", "-(dis_1_2*slow_burn_a1)", "0.0", "(dis_1_2*slow_burn_a1)"], index=Variables)
ff_c0_c1_eff = pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "10.0", "-(dis_0_1*fast_burn_a1)", "0.0", "(dis_0_1*fast_burn_a1)"], index=Variables)
ff_c0_c2_eff = pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "20.0", "-(dis_0_2*fast_burn_a1)", "0.0", "(dis_0_2*fast_burn_a1)"], index=Variables)
ff_c1_c2_eff = pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "10.0", "-(dis_1_2*fast_burn_a1)", "0.0", "(dis_1_2*fast_burn_a1)"], index=Variables)
ff_c1_c0_eff = pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "-10.0", "-(dis_0_1*fast_burn_a1)", "0.0", "(dis_0_1*fast_burn_a1)"], index=Variables)
ff_c2_c0_eff = pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "-20.0", "-(dis_0_2*fast_burn_a1)", "0.0", "(dis_0_2*fast_burn_a1)"], index=Variables)
ff_c2_c1_eff = pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "-10.0", "-(dis_1_2*fast_burn_a1)", "0.0", "(dis_1_2*fast_burn_a1)"], index=Variables)
refuel_eff = pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "fuel", "0.0", "0.0"], index=Variables)
effects = pd.concat([board_a1_p1_c0_eff, board_a1_p1_c1_eff, board_a1_p1_c2_eff, board_a1_p2_c0_eff, board_a1_p2_c1_eff, board_a1_p2_c2_eff, board_a1_p3_c0_eff, board_a1_p3_c1_eff, board_a1_p3_c2_eff,
                     debark_a1_p1_c0_eff, debark_a1_p1_c1_eff, debark_a1_p1_c2_eff, debark_a1_p2_c0_eff, debark_a1_p2_c1_eff, debark_a1_p2_c2_eff, debark_a1_p3_c0_eff, debark_a1_p3_c1_eff, debark_a1_p3_c2_eff,
                     fs_c0_c1_eff, fs_c0_c2_eff, fs_c1_c2_eff, fs_c1_c0_eff, fs_c2_c0_eff, fs_c2_c1_eff,
                     ff_c0_c1_eff, ff_c0_c2_eff, ff_c1_c2_eff, ff_c1_c0_eff, ff_c2_c0_eff, ff_c2_c1_eff,
                     refuel_eff], axis = 1)
effects.columns = sym_actions

if integrate_ml == True:
    #Table of models
    learned_models = ["L_Refuel"]
    trained_model = pd.Series(["model(state_in)"], index = learned_models)
    ld_models = pd.concat([trained_model], axis = 1)
    ld_models.columns = learned_models

    # Table of precondition of learned actions
    L_Heat_pre = pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "<capacity_a1", ">=0.0", ">=0.0"], index=Variables)
    L_precons = pd.concat([L_Heat_pre], axis = 1)
    L_precons.columns = learn_actions

# Table of initial- & goalstate
# Variables = ["location_p1","location_p2", "location_p3", "in_p1", "in_p2", "in_p3", "location_a1", "fuel_a1", "onboard_a1", "total_fuel_used"]
Init = pd.Series(["0.0", "0.0", "0.0", "0.0", "10.0", "0.0", "10.0", "100.0", "10.0", "0.0"], index=Variables)
Goal = pd.Series(["0.0", "0.0", "0.0", "0.0", "10.0", "0.0", "20.0", ">100.0", "10.0", ">=0.0"], index=Variables)
start_end = pd.concat([Init, Goal], axis = 1)

def print_all():
    print(para_fix)
    print(para_var)
    print(precons)
    print(effects)
    print(start_end)

# print_all()