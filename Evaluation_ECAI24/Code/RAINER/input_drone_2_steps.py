import pandas as pd

integrate_ml = True        # enables the integration of ML-models
stepnumber_known = False
stepnumber = 6
approx = True               # enables the approximation of the results
#this is necessary, if small ML models are integrated, due to their inaccuracy 
cc = False

# Table of fix parameters
Parameter_fix = ["battery_level_full", "min_x", "min_y", "min_z", "max_x", "max_y", "max_z"]
Values_fix = pd.Series([200.0, 0.0, 0.0, 0.0, 40.0, 40.0, 40.0], index = Parameter_fix) 
para_fix = pd.concat([Values_fix], axis = 1)
limits_fix = ['Values_fix']
para_fix.columns = limits_fix

# Table of variable parameters
Parameter_var = ["energy"]
Top_var = pd.Series([10000000], index = Parameter_var)
Down_var = pd.Series([-10000], index = Parameter_var)
para_var = pd.concat([Top_var, Down_var], axis = 1)
limits_var = ["Top_var", "Down_var"]
para_var.columns = limits_var

Variables = ["visited_loc1","visited_loc2", "x-axis", "y-axis", "z-axis", "battery_level"]
sym_actions = ["in_x", "in_y", "in_z", "de_x", "de_y", "de_z", "visit_l1", "visit_l2", "recharge"]
learn_actions = ["L_charge"]

# Table of symbolic effects
in_x_eff = pd.Series(["0.0", "0.0", "10.0", "0.0", "0.0", "-10.0"], index=Variables)
in_y_eff = pd.Series(["0.0", "0.0", "0.0", "10.0", "0.0", "-10.0"], index=Variables)
in_z_eff = pd.Series(["0.0", "0.0", "0.0", "0.0", "10.0", "-10.0"], index=Variables)
de_x_eff = pd.Series(["0.0", "0.0", "-10.0", "0.0", "0.0", "-10.0"], index=Variables)
de_y_eff = pd.Series(["0.0", "0.0", "0.0", "-10.0", "0.0", "-10.0"], index=Variables)
de_z_eff = pd.Series(["0.0", "0.0", "0.0", "0.0", "-10.0", "-10.0"], index=Variables)
visit_l1_eff = pd.Series(["10.0", "0.0", "0.0", "0.0", "0.0", "-10.0"], index=Variables)
visit_l2_eff = pd.Series(["0.0", "10.0", "0.0", "0.0", "0.0", "-10.0"], index=Variables)
recharge_eff = pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "energy"], index=Variables)
# recharge_eff = pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "(energy)"], index=Variables)
effects = pd.concat([in_x_eff, in_y_eff, in_z_eff, de_x_eff, de_y_eff, de_z_eff, visit_l1_eff, visit_l2_eff, recharge_eff], axis = 1)
effects.columns = sym_actions

if integrate_ml == True:
    #Table of models
    learned_models = ["L_charge"]
    trained_model = pd.Series(["model(state_in)"], index = learned_models)
    ld_models = pd.concat([trained_model], axis = 1)
    ld_models.columns = learned_models

    # Table of precondition of learned actions
    L_Heat_pre = pd.Series([">=0.0", ">=0.0", "0.0", "0.0", "0.0", "<battery_level_full"], index=Variables)
    L_precons = pd.concat([L_Heat_pre], axis = 1)
    L_precons.columns = learn_actions

# Table of preconditions of symbolic actions
in_x_pre = pd.Series([">=0.0", ">=0.0", "<=max_x-10.0", ">=0.0", ">=0.0", ">=10.0"], index=Variables)
in_y_pre = pd.Series([">=0.0", ">=0.0", ">=0.0", "<=max_y-10.0", ">=0.0", ">=10.0"], index=Variables)
in_z_pre = pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", "<=max_z-10.0", ">=10.0"], index=Variables)
de_x_pre = pd.Series([">=0.0", ">=0.0", ">=min_x+10.0", ">=0.0", ">=0.0", ">=10.0"], index=Variables)
de_y_pre = pd.Series([">=0.0", ">=0.0", ">=0.0", ">=min_y+10.0", ">=0.0", ">=10.0"], index=Variables)
de_z_pre = pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=min_z+10.0", ">=10.0"], index=Variables)
visit_l1_pre = pd.Series([">=0.0", ">=0.0", "0.0", "0.0", "0.0", ">=10.0"], index=Variables)
visit_l2_pre = pd.Series([">=0.0", ">=0.0", "20.0", "20.0", "20.0", ">=10.0"], index=Variables)
recharge_pre = pd.Series([">=0.0", ">=0.0", "0.0", "0.0", "0.0", ">battery_level_full"], index=Variables)
precons = pd.concat([in_x_pre, in_y_pre, in_z_pre, de_x_pre, de_y_pre, de_z_pre, visit_l1_pre, visit_l2_pre, recharge_pre], axis = 1)
precons.columns = sym_actions

# Table of initial- & goalstate
Init = pd.Series(["0.0", "0.0", "10.0", "0.0", "0.0", "90"], index=Variables)
Goal = pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", ">90"], index=Variables)
start_end = pd.concat([Init, Goal], axis = 1)

def print_all():
    print(para_fix)
    print(para_var)
    print(precons)
    print(effects)
    print(start_end)

# print_all()