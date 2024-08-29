import pandas as pd
import random
import numpy as np

class input: 
    def __init__(self, domain, problem):
        self.domain = domain
        self.problem = problem

class domain: 
    def __init__(self, variables, symbolic_actions, learned_actions, fix, var, sym_precon, sym_effect, ml_precon, ml_effects):
        self.variables = variables
        self.symbolic_actions = symbolic_actions
        self.learned_actions = learned_actions
        self.fix_parameters = fix
        self.variable_parameters = var
        self.symbolic_preconditions = sym_precon
        self.symbolic_effects = sym_effect
        self.subsymbolic_preconditions = ml_precon
        self.subsymbolic_effetcs = ml_effects

class df_fix_parameters:
    def __init__(self, parameters, values):
        self.columns = ['limits']
        self.parameters = parameters
        self.values = pd.Series(values, index=parameters)
        self.df = pd.concat([self.values], axis=1)
        self.df.columns = self.columns

class df_variable_parameters:
    def __init__(self, parameters, upper_bounds, under_bounds):
        self.columns = ["Top_var", "Down_var"]
        self.parameters = parameters
        self.upper_bounds = pd.Series(upper_bounds, index=parameters)
        self.under_bounds = pd.Series(under_bounds, index=parameters)
        self.df = pd.concat([self.upper_bounds, self.under_bounds], axis=1)
        self.df.columns = self.columns

class df_symbolic_preconditions:
    def __init__(self, actions, variables, preconditions):
        self.actions = actions
        self.variables = variables
        self.df = pd.concat(preconditions, axis=1)
        self.df.columns = self.actions

class df_subsymbolic_preconditions:
    def __init__(self, learned_actions, variables, preconditions):
        self.learned_actions = learned_actions
        self.variables = variables
        self.df = pd.concat(preconditions, axis=1)
        self.df.columns = self.learned_actions

class df_subsymbolic_models:
    def __init__(self, learned_actions, models):
        self.learned_actions = learned_actions
        self.models = models
        self.df = pd.concat(self.models, axis=1)
        self.df.columns = self.learned_actions

class df_symbolic_effects:
    def __init__(self, variables, actions, effects):
        self.variables = variables
        self.actions = actions
        self.df = pd.concat(effects, axis=1)
        self.df.columns = self.actions

class df_subsymbolic_label:
    def __init__(self, variables, ml_actions, label):
        self.variables = variables
        self.columns = ml_actions
        self.df = pd.concat(label, axis=1)
        self.df.columns = self.columns

class df_subsymbolic_under_bound:
    def __init__(self, variables, ml_actions, under_bound):
        self.variables = variables
        self.columns = ml_actions
        self.df = pd.concat(under_bound, axis=1)
        self.df.columns = self.columns

class variable_parameters:
    def __init__(self, ml_actions, variable_parameters, assignment):
        self.variable_parameters = variable_parameters
        self.columns = ml_actions
        self.df = pd.concat(assignment, axis=1)
        self.df.columns = self.columns

class df_subsymbolic_upper_bound:
    def __init__(self, variables, ml_actions, upper_bound):
        self.variables = variables
        self.columns = ml_actions
        self.df = pd.concat(upper_bound, axis=1)
        self.df.columns = self.columns

class subsymbolic_effetcs: 
    def __init__(self, labels, under_bounds, upper_bounds, variable_parameters):
        self.labels = labels
        self.under_bounds = under_bounds
        self.upper_bounds = upper_bounds
        self.variable_parameters = variable_parameters

class df_problem:
    def __init__(self, variables, init_values, goal_values):
        self.columns = ['Init', 'Goal']
        self.variables = variables
        self.init_values = pd.Series(init_values, index=variables)
        self.goal_values = pd.Series(goal_values, index=variables)
        self.df = pd.concat([self.init_values, self.goal_values], axis=1)
        self.df.columns = self.columns

class data_set:
    def __init__(self, setlength, variable_names, fix_parameter_names, variable_parameter_names):
        self.setlength = setlength
        self.features = len(variable_names)
        self.number_fix_Parameters = len(fix_parameter_names)
        self.number_variable_Parameters = len(variable_parameter_names)

    def generate(self):
        dimensions_States = (self.setlength, self.features)
        dimensions_fix_Parameters = (self.setlength, self.number_fix_Parameters)
        dimensions_var_Parameters = (self.setlength, self.number_variable_Parameters)

        State_in = np.zeros(dimensions_States)
        State_out = np.zeros(dimensions_States)
        Parameters_fix = np.zeros(dimensions_fix_Parameters)
        Parameters_var = np.zeros(dimensions_var_Parameters)

        ### start individual data generator
        # Parameter_var = ["cash"]
        # Parameter_fix = []
        # Features = ["at_pub", "at_supermarket", "at_home", "at_bank_1", "at_bank_2", "can_withdraw_money_bank1", "can_withdraw_money_bank2", 
                    # "canbuy_supermarket", "canbuy_pub", "gotsnackes", "have_enough_dollar", "in_pocket", "max_withdraw_bank_1", "max_withdraw_bank_2"]

        bool_ls = [0, 10]

        cash_top = 100
        cash_down = 0
        can_withdraw_1_top = 200
        can_withdraw_1_down = 0
        can_withdraw_2_top = 200
        can_withdraw_2_down = 0

        for i in range(len(State_out)):

            State_in[i][0] = 0
            State_out[i][0] = State_in[i][0]
            State_in[i][1] = 0
            State_out[i][1] = State_in[i][1]
            State_in[i][2] = 0
            State_out[i][2] = State_in[i][2]
            State_in[i][3] = 10
            State_out[i][3] = State_in[i][3]
            State_in[i][4] = 0
            State_out[i][4] = State_in[i][4]
            State_in[i][5] = 10
            State_out[i][5] = State_in[i][5]
            State_in[i][6] = random.choice(bool_ls)
            State_out[i][6] = State_in[i][6]
            State_in[i][7] = random.choice(bool_ls)
            State_out[i][7] = State_in[i][7]
            State_in[i][8] = random.choice(bool_ls)
            State_out[i][8] = State_in[i][8]
            State_in[i][9] = random.choice(bool_ls)
            State_out[i][9] = State_in[i][9]
            State_in[i][10] = random.choice(bool_ls)
            State_out[i][10] = State_in[i][10]
            State_in[i][11] = random.choice(bool_ls)

            State_in[i][13] = random.randrange(can_withdraw_2_down, can_withdraw_2_top, 10)
            State_out[i][13] = State_in[i][13]
        
            Parameters_fix[i][0] = 5

            State_in[i][12] = random.randrange(can_withdraw_1_down, can_withdraw_1_top, 10)
            Parameters_var[i][0] = random.randrange(cash_down, cash_top, 1)
            State_out[i][12] = State_in[i][12] - Parameters_var[i][0]
            State_out[i][11] = State_in[i][11] + Parameters_var[i][0]

        ### end individual data generator

        Input = np.concatenate((State_in, Parameters_fix, Parameters_var), axis =1 )
        Output = State_out

        return Input, Output
    
cashpoint_variables_names = ["at_pub", "at_supermarket", "at_home", "at_bank_1", "at_bank_2", "can_withdraw_money_bank1", "can_withdraw_money_bank2", 
                             "canbuy_pub", "canbuy_supermarket", "gotsnackes", "have_enough_dollar", "in_pocket", "max_withdraw_bank_1", "max_withdraw_bank_2"]
cashpoint_symbolic_actions = ["pub_to_supermarkte", "pub_to_home", "pub_to_bank_1", "pub_to_bank_2", 
                              "supermarket_to_pub", "supermarket_to_home", "supermarket_to_bank_1", "supermarket_to_bank_2", 
                              "home_to_pub", "home_to_supermarket", "home_to_bank_1", "home_to_bank_2",
                              "bank_1_to_pub", "bank_1_to_supermarket", "bank_1_to_home", "bank_1_to_bank_2",
                              "bank_2_to_pub", "bank_2_to_supermarket", "bank_2_to_home", "bank_2_to_bank_1",
                              "buy_snacks_pub", "buy_snacks_supermarket", "withdraw_bank_1", "withdraw_bank_2", "check_pocket"]
cashpoint_learned_actions = ["L_withdraw_bank_1"]

    # Table of fix parameters
cashpoint_fix_parameters_names = ["max_snacks"]
cashpoint_fix_parameters_values = [5]
cashpoint_fix_paraneters = df_fix_parameters(cashpoint_fix_parameters_names, cashpoint_fix_parameters_values)

    # Table of variable parameters
cashpoint_var_parameters_names = ["cash"]
cashpoint_var_parameters_top_compare = [110.0]
cashpoint_var_parameters_down_compare = [0.0]
cashpoint_var_parameters_top_listener = [20.0]
cashpoint_var_parameters_down_listener = [5.0]
cashpoint_var_parameters = df_variable_parameters(cashpoint_var_parameters_names, cashpoint_var_parameters_top_compare, cashpoint_var_parameters_down_compare)

    # Table of symbolic preconditions
cashpoint_symbolic_preconditions_data = [
        pd.Series(["10.0", "0.0", "0.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
        pd.Series(["10.0", "0.0", "0.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
        pd.Series(["10.0", "0.0", "0.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
        pd.Series(["10.0", "0.0", "0.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),        
        
        pd.Series(["0.0", "10.0", "0.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
        pd.Series(["0.0", "10.0", "0.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
        pd.Series(["0.0", "10.0", "0.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
        pd.Series(["0.0", "10.0", "0.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
        
        pd.Series(["0.0", "0.0", "10.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
        pd.Series(["0.0", "0.0", "10.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
        pd.Series(["0.0", "0.0", "10.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
        pd.Series(["0.0", "0.0", "10.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
        
        pd.Series(["0.0", "0.0", "0.0", "10.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "10.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "10.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "10.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),

        pd.Series(["0.0", "0.0", "0.0", "0.0", "10.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "10.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "10.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "10.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),

        pd.Series(["10.0", "0.0", "0.0", "0.0", "0.0", "0.0", ">=0.0", "10.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
        pd.Series(["0.0", "10.0", "0.0", "0.0", "0.0", "10.0", ">=0.0", ">=0.0", "10.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
        pd.Series(["200.0", "0.0", "0.0", "10.0", "0.0", "10.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "10.0", ">=0.0", "10.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=5.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=100.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),

]       
cashpoint_symbolic_precons = df_subsymbolic_preconditions(cashpoint_symbolic_actions, cashpoint_variables_names, cashpoint_symbolic_preconditions_data)

    # Table of symbolic effects
cashpoint_symbolic_effects_data = [
        pd.Series(["-10.0", "10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
        pd.Series(["-10.0", "0.0", "10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
        pd.Series(["-10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
        pd.Series(["-10.0", "0.0", "0.0", "0.0", "10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),        
        
        pd.Series(["10.0", "-10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
        pd.Series(["0.0", "-10.0", "10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
        pd.Series(["0.0", "-10.0", "0.0", "10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
        pd.Series(["0.0", "-10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names), 

        pd.Series(["10.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
        pd.Series(["0.0", "10.0", "-10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
        pd.Series(["0.0", "0.0", "-10.0", "10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
        pd.Series(["0.0", "0.0", "-10.0", "0.0", "10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names), 

        pd.Series(["10.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
        pd.Series(["0.0", "10.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
        pd.Series(["0.0", "0.0", "10.0", "-10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "-10.0", "10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names), 

        pd.Series(["10.0", "0.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
        pd.Series(["0.0", "10.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
        pd.Series(["0.0", "0.0", "10.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "10.0", "-10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names), 

        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "10.0", "0.0", "-5.0", "0.0", "0.0"], index=cashpoint_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "10.0", "0.0", "-5.0", "0.0", "0.0"], index=cashpoint_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "cash", "-cash", "0.0"], index=cashpoint_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "cash", "0.0", "-cash"], index=cashpoint_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "10.0", "-100.0", "0.0", "0.0"], index=cashpoint_variables_names),
        ]
cashpoint_symbolic_effects = df_symbolic_effects(cashpoint_variables_names, cashpoint_symbolic_actions, cashpoint_symbolic_effects_data)

    # Table of subsymbolic preconditions    
cashpoint_subsymbolic_preconditions_data = [
    pd.Series(["0.0", "0.0", "0.0", "10.0", "0.0", "10.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
    ]
cashpoint_subsymbolic_precons = df_subsymbolic_preconditions(cashpoint_learned_actions, cashpoint_variables_names, cashpoint_subsymbolic_preconditions_data)

    # Table of subsymbolic effects
cashpoint_subsymbolic_effects_data = [
        pd.Series(["model(state_in)"], index=cashpoint_learned_actions),
    ]
    #cashpoint_subsymbolic_effects = df_subsymbolic_models(cashpoint_learned_actions, cashpoint_subsymbolic_effects_data)

cashpoint_variable_parameters_data = [
        pd.Series(["10.0"], index=cashpoint_var_parameters_names)
    ]

    # Table of initial- & goalstate
# compare two step
cashpoint_init_1 = ["0.0", "0.0", "10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "5.0", "200", "200"]
cashpoint_goal_1 = ["0.0", "0.0", "0.0", "10.0", "0.0", "10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "20.0", ">=185", ">=200"]
cashpoint_1 = df_problem(cashpoint_variables_names, cashpoint_init_1, cashpoint_goal_1)
                        # Action_0       home_to_bank_1    0
                        # Action_1  L_withdraw_bank_1&1    0, ['L_withdraw_bank_1'])

# compare three step
cashpoint_init_2 = ["0.0", "0.0", "10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "5.0", "200", "200"]
cashpoint_goal_2 = ["10.0", "0.0", "0.0", "0.0", "0.0", "10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "20.0", ">=185", ">=200"]
cashpoint_2 = df_problem(cashpoint_variables_names, cashpoint_init_2, cashpoint_goal_2)
                        # Action_0       home_to_bank_1    0
                        # Action_1  L_withdraw_bank_1&1    0
                        # Action_2        bank_1_to_pub    0, ['L_withdraw_bank_1'])

# compare input
cashpoint_init_3 = ["0.0", "0.0", "10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "5.0", "200", "200"]
cashpoint_goal_3 = ["10.0", "0.0", "0.0", "0.0", "0.0", "10.0", "0.0", "0.0", "10.0", "10.0", "10.0", ">=0.0", "<=200", "<=200"]
cashpoint_3 = df_problem(cashpoint_variables_names, cashpoint_init_3, cashpoint_goal_3)
                        # Action_0          home_to_bank_1    0
                        # Action_1     L_withdraw_bank_1&1    0
                        # Action_2            check_pocket    0
                        # Action_3   bank_1_to_supermarket    0
                        # Action_4  buy_snacks_supermarket    0
                        # Action_5      supermarket_to_pub    0, ['L_withdraw_bank_1'])

recharge_listener_labels_data = [
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "10.0", "10.0", "0.0"], index=cashpoint_variables_names)]
        # pd.Series(["10.0", "10.0", "10.0", "10.0", "10.0", "10.0", "10.0", "10.0", "10.0", "10.0", "10.0", "10.0", "10.0", "10.0"], index=cashpoint_variables_names)]             # include this to run the planer with no listener information
recharge_listener_under_bounds_data = [
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "-95.0", "0.0"], index=cashpoint_variables_names)]
recharge_listener_upper_bounds_data = [
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "107.0", "0.0", "00.0"], index=cashpoint_variables_names)]

cashpoint_subsymbolic_effects_parameters = variable_parameters(cashpoint_learned_actions, cashpoint_var_parameters_names, cashpoint_variable_parameters_data)
cashpoint_subsymbolic_labels = df_subsymbolic_label(cashpoint_variables_names, cashpoint_learned_actions, recharge_listener_labels_data)
cashpoint_subsymbolic_under_bounds = df_subsymbolic_under_bound(cashpoint_variables_names, cashpoint_learned_actions, recharge_listener_under_bounds_data)
cashpoint_subsymbolic_upper_bounds = df_subsymbolic_upper_bound(cashpoint_variables_names, cashpoint_learned_actions, recharge_listener_upper_bounds_data)

cashpoint_subsymbolic_effects = subsymbolic_effetcs(cashpoint_subsymbolic_labels, cashpoint_subsymbolic_under_bounds, cashpoint_subsymbolic_upper_bounds, cashpoint_subsymbolic_effects_parameters)

cashpoint_domain = domain(cashpoint_variables_names, cashpoint_symbolic_actions, cashpoint_learned_actions, cashpoint_fix_paraneters, cashpoint_var_parameters, cashpoint_symbolic_precons, cashpoint_symbolic_effects, cashpoint_subsymbolic_precons,cashpoint_subsymbolic_effects)
cashpoint_domains = input(cashpoint_domain, cashpoint_3)

def create_cashpoint_dataset(n_samples):
    cashpoint_variables_names = ["at_pub", "at_supermarket", "at_home", "at_bank_1", "at_bank_2", "can_withdraw_money_bank1", "can_withdraw_money_bank2", 
                                "canbuy_supermarket", "canbuy_pub", "gotsnackes", "have_enough_dollar", "in_pocket", "max_withdraw_bank_1", "max_withdraw_bank_2"]
    cashpoint_symbolic_actions = ["pub_to_supermarkte", "pub_to_home", "pub_to_bank_1", "pub_to_bank_2", 
                                "supermarket_to_pub", "supermarket_to_home", "supermarket_to_bank_1", "supermarket_to_bank_2", 
                                "home_to_pub", "home_to_supermarket", "home_to_bank_1", "home_to_bank_2",
                                "bank_1_to_pub", "bank_1_to_supermarket", "bank_1_to_home", "bank_1_to_bank_2",
                                "bank_2_to_pub", "bank_2_to_supermarket", "bank_2_to_home", "bank_2_to_bank_1",
                                "buy_snacks_pub", "buy_snacks_supermarket", "withdraw_bank_1", "withdraw_bank_2", "check_pocket"]
    cashpoint_learned_actions = ["L_withdraw_bank_1"]

        # Table of fix parameters
    cashpoint_fix_parameters_names = ["max_snacks"]
    cashpoint_fix_parameters_values = [5]
    cashpoint_fix_paraneters = df_fix_parameters(cashpoint_fix_parameters_names, cashpoint_fix_parameters_values)

        # Table of variable parameters
    cashpoint_var_parameters_names = ["cash"]
    cashpoint_var_parameters_top_compare = [1000.0]
    cashpoint_var_parameters_down_compare = [-1000.0]
    cashpoint_var_parameters_top_listener = [20.0]
    cashpoint_var_parameters_down_listener = [5.0]
    cashpoint_var_parameters = df_variable_parameters(cashpoint_var_parameters_names, cashpoint_var_parameters_top_compare, cashpoint_var_parameters_down_compare)

        # Table of symbolic preconditions
    cashpoint_symbolic_preconditions_data = [
            pd.Series(["10.0", "0.0", "0.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
            pd.Series(["10.0", "0.0", "0.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
            pd.Series(["10.0", "0.0", "0.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
            pd.Series(["10.0", "0.0", "0.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),        
            
            pd.Series(["0.0", "10.0", "0.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
            pd.Series(["0.0", "10.0", "0.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
            pd.Series(["0.0", "10.0", "0.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
            pd.Series(["0.0", "10.0", "0.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
            
            pd.Series(["0.0", "0.0", "10.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
            pd.Series(["0.0", "0.0", "10.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
            pd.Series(["0.0", "0.0", "10.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
            pd.Series(["0.0", "0.0", "10.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
            
            pd.Series(["0.0", "0.0", "0.0", "10.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
            pd.Series(["0.0", "0.0", "0.0", "10.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
            pd.Series(["0.0", "0.0", "0.0", "10.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
            pd.Series(["0.0", "0.0", "0.0", "10.0", "0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),

            pd.Series(["0.0", "0.0", "0.0", "0.0", "10.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
            pd.Series(["0.0", "0.0", "0.0", "0.0", "10.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
            pd.Series(["0.0", "0.0", "0.0", "0.0", "10.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
            pd.Series(["0.0", "0.0", "0.0", "0.0", "10.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),

            pd.Series(["10.0", "0.0", "0.0", "0.0", "0.0", "0.0", ">=0.0", "10.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
            pd.Series(["0.0", "10.0", "0.0", "0.0", "0.0", "10.0", ">=0.0", ">=0.0", "10.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
            pd.Series(["200.0", "0.0", "0.0", "10.0", "0.0", "10.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
            pd.Series(["0.0", "0.0", "0.0", "0.0", "10.0", ">=0.0", "10.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=5.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
            pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=100.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),

    ]       
    cashpoint_symbolic_precons = df_subsymbolic_preconditions(cashpoint_symbolic_actions, cashpoint_variables_names, cashpoint_symbolic_preconditions_data)

        # Table of symbolic effects
    cashpoint_symbolic_effects_data = [
            pd.Series(["-10.0", "10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
            pd.Series(["-10.0", "0.0", "10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
            pd.Series(["-10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
            pd.Series(["-10.0", "0.0", "0.0", "0.0", "10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),        
            
            pd.Series(["10.0", "-10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
            pd.Series(["0.0", "-10.0", "10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
            pd.Series(["0.0", "-10.0", "0.0", "10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
            pd.Series(["0.0", "-10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names), 

            pd.Series(["10.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
            pd.Series(["0.0", "10.0", "-10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
            pd.Series(["0.0", "0.0", "-10.0", "10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
            pd.Series(["0.0", "0.0", "-10.0", "0.0", "10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names), 

            pd.Series(["10.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
            pd.Series(["0.0", "10.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
            pd.Series(["0.0", "0.0", "10.0", "-10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
            pd.Series(["0.0", "0.0", "0.0", "-10.0", "10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names), 

            pd.Series(["10.0", "0.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
            pd.Series(["0.0", "10.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
            pd.Series(["0.0", "0.0", "10.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names),
            pd.Series(["0.0", "0.0", "0.0", "10.0", "-10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=cashpoint_variables_names), 

            pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "10.0", "0.0", "-5.0", "0.0", "0.0"], index=cashpoint_variables_names),
            pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "10.0", "0.0", "-5.0", "0.0", "0.0"], index=cashpoint_variables_names),
            pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "cash", "-cash", "0.0"], index=cashpoint_variables_names),
            pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "cash", "0.0", "-cash"], index=cashpoint_variables_names),
            pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "10.0", "-100.0", "0.0", "0.0"], index=cashpoint_variables_names),
            ]
    cashpoint_symbolic_effects = df_symbolic_effects(cashpoint_variables_names, cashpoint_symbolic_actions, cashpoint_symbolic_effects_data)

        # Table of subsymbolic preconditions    
    cashpoint_subsymbolic_preconditions_data = [
        pd.Series(["0.0", "0.0", "0.0", "10.0", "0.0", "10.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0"], index=cashpoint_variables_names),
        ]
    cashpoint_subsymbolic_precons = df_subsymbolic_preconditions(cashpoint_learned_actions, cashpoint_variables_names, cashpoint_subsymbolic_preconditions_data)

        # Table of subsymbolic effects
    cashpoint_subsymbolic_effects_data = [
            pd.Series(["model(state_in)"], index=cashpoint_learned_actions),
        ]
        #cashpoint_subsymbolic_effects = df_subsymbolic_models(cashpoint_learned_actions, cashpoint_subsymbolic_effects_data)

    cashpoint_variable_parameters_data = [
            pd.Series(["10.0"], index=cashpoint_var_parameters_names)
        ]

        # Table of initial- & goalstate
    # compare two step
    cashpoint_init_1 = ["0.0", "0.0", "10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "5.0", "200", "200"]
    cashpoint_goal_1 = ["0.0", "0.0", "0.0", "10.0", "0.0", "10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "20.0", ">=185", ">=200"]
    cashpoint_1 = df_problem(cashpoint_variables_names, cashpoint_init_1, cashpoint_goal_1)

    # compare three step
    cashpoint_init_2 = ["0.0", "0.0", "10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "5.0", "200", "200"]
    cashpoint_goal_2 = ["10.0", "0.0", "0.0", "0.0", "0.0", "10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "20.0", ">=185", ">=200"]
    cashpoint_2 = df_problem(cashpoint_variables_names, cashpoint_init_2, cashpoint_goal_2)

    # compare input
    cashpoint_init_3 = ["0.0", "0.0", "10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "5.0", "200", "200"]
    cashpoint_goal_3 = ["10.0", "0.0", "0.0", "0.0", "0.0", "10.0", "0.0", "0.0", "10.0", "10.0", "10.0", ">=0.0", "<=200", "<=200"]
    cashpoint_3 = df_problem(cashpoint_variables_names, cashpoint_init_3, cashpoint_goal_3)

    recharge_listener_labels_data = [
            pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "10.0", "10.0", "0.0"], index=cashpoint_variables_names)]
    recharge_listener_under_bounds_data = [
            pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "-95.0", "0.0"], index=cashpoint_variables_names)]
    recharge_listener_upper_bounds_data = [
            pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "107.0", "0.0", "00.0"], index=cashpoint_variables_names)]

    cashpoint_subsymbolic_effects_parameters = variable_parameters(cashpoint_learned_actions, cashpoint_var_parameters_names, cashpoint_variable_parameters_data)
    cashpoint_subsymbolic_labels = df_subsymbolic_label(cashpoint_variables_names, cashpoint_learned_actions, recharge_listener_labels_data)
    cashpoint_subsymbolic_under_bounds = df_subsymbolic_under_bound(cashpoint_variables_names, cashpoint_learned_actions, recharge_listener_under_bounds_data)
    cashpoint_subsymbolic_upper_bounds = df_subsymbolic_upper_bound(cashpoint_variables_names, cashpoint_learned_actions, recharge_listener_upper_bounds_data)

    cashpoint_subsymbolic_effects = subsymbolic_effetcs(cashpoint_subsymbolic_labels, cashpoint_subsymbolic_under_bounds, cashpoint_subsymbolic_upper_bounds, cashpoint_subsymbolic_effects_parameters)

    cashpoint_domain = domain(cashpoint_variables_names, cashpoint_symbolic_actions, cashpoint_learned_actions, cashpoint_fix_paraneters, cashpoint_var_parameters, cashpoint_symbolic_precons, cashpoint_symbolic_effects, cashpoint_subsymbolic_precons,cashpoint_subsymbolic_effects)
    cashpoint_domains = input(cashpoint_domain, cashpoint_1)

    set_1 = data_set(n_samples, cashpoint_variables_names, cashpoint_fix_parameters_names, cashpoint_var_parameters_names)
    set_1_input, set2_output = set_1.generate()

    return set_1_input, set2_output

# Quicktest
if __name__ == "__main__":
    set_1_input, set2_output = create_cashpoint_dataset(500)
    print(set_1_input[0])
    print(set2_output[0])
    print("yippee")