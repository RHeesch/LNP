import pandas as pd

integrate_ml = True        # enables the integration of ML-models
stepnumber_known = False
stepnumber = 6
approx = True               # enables the approximation of the results
#this is necessary, if small ML models are integrated, due to their inaccuracy 
cc = False

# Table of fix parameters
Parameter_fix = []
Values_fix = pd.Series([], index = Parameter_fix) 
para_fix = pd.concat([Values_fix], axis = 1)
limits_fix = ['Values_fix']
para_fix.columns = limits_fix

# Table of variable parameters
Parameter_var = ["cash"]
Top_var = pd.Series([1000], index = Parameter_var)
Down_var = pd.Series([-1000], index = Parameter_var)
para_var = pd.concat([Top_var, Down_var], axis = 1)
limits_var = ["Top_var", "Down_var"]
para_var.columns = limits_var

Variables = ["at_pub", "at_supermarket", "at_home", "at_bank_1", "at_bank_2", "can_withdraw_money_bank1", "can_withdraw_money_bank2", 
                             "canbuy_pub", "canbuy_supermarket", "gotsnackes", "have_enough_dollar", "in_pocket", "max_withdraw_bank_1", "max_withdraw_bank_2"]
sym_actions = ["pub_to_supermarkte", "pub_to_home", "pub_to_bank_1", "pub_to_bank_2", 
                              "supermarket_to_pub", "supermarket_to_home", "supermarket_to_bank_1", "supermarket_to_bank_2", 
                              "home_to_pub", "home_to_supermarket", "home_to_bank_1", "home_to_bank_2",
                              "bank_1_to_pub", "bank_1_to_supermarket", "bank_1_to_home", "bank_1_to_bank_2",
                              "bank_2_to_pub", "bank_2_to_supermarket", "bank_2_to_home", "bank_2_to_bank_1",
                              "buy_snacks_pub", "buy_snacks_supermarket", "withdraw_bank_1", "withdraw_bank_2", "check_pocket"]
learn_actions = ["L_withdraw_bank_1"]

# Table of preconditions of symbolic actions
pub_to_supermarket_pre = pd.Series(["10.0", "0.0", "0.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
pub_to_home_pre = pd.Series(["10.0", "0.0", "0.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
pub_to_bank_1_pre = pd.Series(["10.0", "0.0", "0.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
pub_to_bank_2_pre = pd.Series(["10.0", "0.0", "0.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
supermarket_to_pub_pre = pd.Series(["0.0", "10.0", "0.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
supermarket_to_home_pre = pd.Series(["0.0", "10.0", "0.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
supermarket_to_bank_1_pre = pd.Series(["0.0", "10.0", "0.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
supermarket_to_bank_2_pre = pd.Series(["0.0", "10.0", "0.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
home_to_pub_pre = pd.Series(["0.0", "0.0", "10.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
home_to_supermarket_pre = pd.Series(["0.0", "0.0", "10.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
home_to_bank_1_pre = pd.Series(["0.0", "0.0", "10.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
home_to_bank_2_pre = pd.Series(["0.0", "0.0", "10.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
bank_1_to_pub_pre = pd.Series(["0.0", "0.0", "0.0", "10.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
bank_1_to_supermarket_pre = pd.Series(["0.0", "0.0", "0.0", "10.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
bank_1_to_home_pre = pd.Series(["0.0", "0.0", "0.0", "10.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
bank_1_to_bank_2_pre = pd.Series(["0.0", "0.0", "0.0", "10.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
bank_2_to_pub_pre = pd.Series(["0.0", "0.0", "0.0", "0.0", "10.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
bank_2_to_supermarket_pre = pd.Series(["0.0", "0.0", "0.0", "0.0", "10.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
bank_2_to_home_pre = pd.Series(["0.0", "0.0", "0.0", "0.0", "10.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
bank_2_to_bank_1_pre= pd.Series(["0.0", "0.0", "0.0", "0.0", "10.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
buy_snacks_pub_pre = pd.Series(["10.0", "0.0", "0.0", "0.0", "0.0", "0.0", ">=0.0", "10.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
buy_snacks_supermarket_pre = pd.Series(["0.0", "10.0", "0.0", "0.0", "0.0", "10.0", ">=0.0", ">=0.0", "10.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=Variables)
withdraw_bank_1_pre = pd.Series(["0.0", "0.0", "0.0", "10.0", "0.0", "10.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "<=-200.0", ">=0.0", ">=0.0"], index=Variables)
withdraw_bank_2_pre = pd.Series(["0.0", "0.0", "0.0", "0.0", "10.0", ">=0.0", "10.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=5.0", ">=0.0", ">=0.0"], index=Variables)
check_pocket_pre = pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=100.0", ">=0.0", ">=0.0"], index=Variables)
precons = pd.concat([pub_to_supermarket_pre, pub_to_home_pre, pub_to_bank_1_pre, pub_to_bank_2_pre, supermarket_to_pub_pre, supermarket_to_home_pre, supermarket_to_bank_1_pre, supermarket_to_bank_2_pre, home_to_pub_pre,
                     home_to_supermarket_pre, home_to_bank_1_pre, home_to_bank_2_pre, bank_1_to_pub_pre, bank_1_to_supermarket_pre, bank_1_to_home_pre, bank_1_to_bank_2_pre, bank_2_to_pub_pre, bank_2_to_supermarket_pre,
                     bank_2_to_home_pre, bank_2_to_bank_1_pre, buy_snacks_pub_pre, buy_snacks_supermarket_pre, withdraw_bank_1_pre, withdraw_bank_2_pre,
                     check_pocket_pre], axis = 1)
precons.columns = sym_actions

# Table of symbolic effects
pub_to_supermarket_eff = pd.Series(["-10.0", "10.0", "", "", "", "", "", "", "", "", "", "", "", ""], index=Variables)
pub_to_home_eff = pd.Series(["-10.0", "", "10.0", "", "", "", "", "", "", "", "", "", "", ""], index=Variables)
pub_to_bank_1_eff = pd.Series(["-10.0", "", "", "10.0", "", "", "", "", "", "", "", "", "", ""], index=Variables)
pub_to_bank_2_ef = pd.Series(["-10.0", "", "", "", "10.0", "", "", "", "", "", "", "", "", ""], index=Variables)
supermarket_to_pub_eff = pd.Series(["10.0", "-10.0", "", "", "", "", "", "", "", "", "", "", "", ""], index=Variables)
supermarket_to_home_eff = pd.Series(["", "-10.0", "10.0", "", "", "", "", "", "", "", "", "", "", ""], index=Variables)
supermarket_to_bank_1_eff = pd.Series(["", "-10.0", "", "10.0", "", "", "", "", "", "", "", "", "", ""], index=Variables)
supermarket_to_bank_2_eff = pd.Series(["", "-10.0", "", "", "10.0", "", "", "", "", "", "", "", "", ""], index=Variables)
home_to_pub_eff = pd.Series(["10.0", "", "-10.0", "", "", "", "", "", "", "", "", "", "", ""], index=Variables)
home_to_supermarket_eff = pd.Series(["", "10.0", "-10.0", "", "", "", "", "", "", "", "", "", "", ""], index=Variables)
home_to_bank_1_eff = pd.Series(["", "", "-10.0", "10.0", "", "", "", "", "", "", "", "", "", ""], index=Variables)
home_to_bank_2_eff = pd.Series(["", "", "-10.0", "", "10.0", "", "", "", "", "", "", "", "", ""], index=Variables)
bank_1_to_pub_eff = pd.Series(["10.0", "", "", "-10.0", "", "", "", "", "", "", "", "", "", ""], index=Variables)
bank_1_to_supermarket_eff = pd.Series(["", "10.0", "", "-10.0", "", "", "", "", "", "", "", "", "", ""], index=Variables)
bank_1_to_home_eff = pd.Series(["", "", "10.0", "-10.0", "", "", "", "", "", "", "", "", "", ""], index=Variables)
bank_1_to_bank_2_eff = pd.Series(["", "", "", "-10.0", "10.0", "", "", "", "", "", "", "", "", ""], index=Variables)
bank_2_to_pub_eff = pd.Series(["10.0", "", "", "", "-10.0", "", "", "", "", "", "", "", "", ""], index=Variables)
bank_2_to_supermarket_eff = pd.Series(["", "10.0", "", "", "-10.0", "", "", "", "", "", "", "", "", ""], index=Variables)
bank_2_to_home_eff = pd.Series(["", "", "10.0", "", "-10.0", "", "", "", "", "", "", "", "", ""], index=Variables)
bank_2_to_bank_1_eff = pd.Series(["", "", "", "10.0", "-10.0", "", "", "", "", "", "", "", "", ""], index=Variables)
buy_snacks_pub_eff = pd.Series(["", "", "", "", "", "", "", "", "", "10.0", "", "-5.0", "", ""], index=Variables)
buy_snacks_supermarket_eff = pd.Series(["", "", "", "", "", "", "", "", "", "10.0", "", "-5.0", "", ""], index=Variables)
withdraw_bank_1_eff = pd.Series(["", "", "", "", "", "", "", "", "", "", "", "cash", "-cash", ""], index=Variables)
withdraw_bank_2_eff = pd.Series(["", "", "", "", "", "", "", "", "", "", "", "cash", "", "-cash"], index=Variables)
check_pocket_eff = pd.Series(["", "", "", "", "", "", "", "", "", "", "10.0", "-100.0", "", ""], index=Variables)
effects = pd.concat([pub_to_supermarket_eff, pub_to_home_eff, pub_to_bank_1_eff, pub_to_bank_2_ef, supermarket_to_pub_eff, supermarket_to_home_eff, supermarket_to_bank_1_eff, supermarket_to_bank_2_eff, home_to_pub_eff,
                     home_to_supermarket_eff, home_to_bank_1_eff, home_to_bank_2_eff, bank_1_to_pub_eff, bank_1_to_supermarket_eff, bank_1_to_home_eff, bank_1_to_bank_2_eff, bank_2_to_pub_eff, bank_2_to_supermarket_eff,
                     bank_2_to_home_eff, bank_2_to_bank_1_eff, buy_snacks_pub_eff, buy_snacks_supermarket_eff, withdraw_bank_1_eff, withdraw_bank_2_eff,
                     check_pocket_eff], axis = 1)
effects.columns = sym_actions

if integrate_ml == True:
    #Table of models
    learned_models = ["L_withdraw_bank_1"]
    trained_model = pd.Series(["model(state_in)"], index = learned_models)
    ld_models = pd.concat([trained_model], axis = 1)
    ld_models.columns = learned_models

    # Table of precondition of learned actions
    L_Heat_pre = pd.Series(["0.0", "0.0", "0.0", "10.0", "0.0", "10.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=5.0", ">=0.0", ">=0.0"], index=Variables)
    L_precons = pd.concat([L_Heat_pre], axis = 1)
    L_precons.columns = learn_actions

# Table of initial- & goalstate
Init = pd.Series(["0.0", "0.0", "10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "5.0", "200", "200"], index=Variables)
Goal = pd.Series(["0.0", "0.0", "0.0", "10.0", "0.0", "10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "20.0", ">=185", ">=200"], index=Variables)
start_end = pd.concat([Init, Goal], axis = 1)

def print_all():
    print(para_fix)
    print(para_var)
    print(precons)
    print(effects)
    print(start_end)

# print_all()