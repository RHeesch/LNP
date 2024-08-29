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
        # Parameter_var = ["refuel"]
        # Parameter_fix = ["dis_0_1", "dis_0_2", "dis_1_2", "slow_burn_a1", "fast_burn_a1", "capacity_a1", "zoom_limit_a1"]
        # Features = ["location_p1","location_p2", "location_p3", "in_p1", "in_p2", "in_p3", "location_a1", "fuel_a1", "onboard_a1", "total_fuel_used"]

        bool_ls = [0, 10]

        dis_0_1_top = 10
        dis_0_1_down = 0
        dis_0_2_top = 10
        dis_0_2_down = 0
        dis_1_2_top = 10
        dis_1_2_down = 0
        slow_burn_a1_top = 10
        slow_burn_a1_down = 0
        fast_burn_a1_top = 10
        fast_burn_a1_down = 0
        capacity_a1_top = 100
        capacity_a1_down = 0
        zoom_limit_a1_top = 10
        zoom_limit_a1_down = 0

        total_fuel_used_top = 200
        total_fuel_used_down = 0

        for i in range(len(State_out)):
            Parameters_fix[i][1] = random.randrange(dis_0_1_down, dis_0_1_top, 1)
            Parameters_fix[i][2] = random.randrange(dis_0_2_down, dis_0_2_top, 1)
            Parameters_fix[i][3] = random.randrange(dis_1_2_down, dis_1_2_top, 1)
            Parameters_fix[i][4] = random.randrange(slow_burn_a1_down, slow_burn_a1_top, 1)
            Parameters_fix[i][5] = random.randrange(fast_burn_a1_down, fast_burn_a1_top, 1)
            Parameters_fix[i][6] = random.randrange(capacity_a1_down, capacity_a1_top, 1)
            Parameters_fix[i][6] = random.randrange(zoom_limit_a1_down, zoom_limit_a1_top, 1)


            State_in[i][0] = random.choice(bool_ls)
            State_out[i][0] = State_in[i][0]
            State_in[i][1] = random.choice(bool_ls)
            State_out[i][1] = State_in[i][1]
            State_in[i][2] = 0
            State_out[i][2] = State_in[i][2]
            State_in[i][3] = 0
            State_out[i][3] = State_in[i][3]
            State_in[i][4] = 0
            State_out[i][4] = State_in[i][4]
            State_in[i][5] = 0
            State_out[i][5] = State_in[i][5]
            State_in[i][6] = 0
            State_out[i][6] = State_in[i][6]
            State_in[i][9] = 0
            State_out[i][9] = State_in[i][9]
            State_in[i][8] = 0
            State_out[i][8] = State_in[i][8]
            State_in[i][7] = random.randrange(total_fuel_used_down, total_fuel_used_top, 1)
            State_out[i][7] = Parameters_fix[i][0]

            Parameters_var[i][0] = State_out[i][7] - State_in[i][7]
        ### end individual data generator

        Input = np.concatenate((State_in, Parameters_fix, Parameters_var), axis =1 )
        Output = State_out

        return Input, Output

Zeno_variables_names = ["location_p1","location_p2", "location_p3", "in_p1", "in_p2", "in_p3", "location_a1", "fuel_a1", "onboard_a1", "total_fuel_used"]
Zeno_symbolic_actions = ["board_a1_p1_c0", "board_a1_p1_c1", "board_a1_p1_c2", "board_a1_p2c0", "board_a1_p2_c1", "board_a1_p2_c2", "board_a1_p3_c0", "board_a1_p3_c1", "board_a1_p3_c2",
               "debark_a1_p1_c0", "debark_a1_p1_c1", "debark_a1_p1_c2", "debark_a1_p2_c0", "debark_a1_p2_c1", "debark_a1_p2_c2", "debark_a1_p3_c0", "debark_a1_p3_c1", "debark_a1_p3_c2",
               "fs_c0_c1", "fs_c0_c2", "fs_c1_c2", "fs_c1_c0", "fs_c2_c0", "fs_c2_c1", 
               "ff_c0_c1", "ff_c0_c2", "ff_c1_c2", "ff_c1_c0", "ff_c2_c0", "ff_c2_c1",
               "refuel"]
Zeno_learned_actions = ["ML_refuel"]

    # Table of fix parameters
Zeno_fix_parameters_names = ["dis_0_1", "dis_0_2", "dis_1_2", "slow_burn_a1", "fast_burn_a1", "capacity_a1", "zoom_limit_a1"]
Zeno_fix_parameters_values = ["6.0", "7.0", "8.0", "4.0", "1.50", "100.0", "8.0"]
Zeno_fix_paraneters = df_fix_parameters(Zeno_fix_parameters_names, Zeno_fix_parameters_values)

    # Table of variable parameters
Zeno_var_parameters_names = ["fuel"]
Zeno_var_parameters_top_compare = [200.0]
Zeno_var_parameters_down_compare = [0.0]
Zeno_var_parameters_top_listener = [20.0]
Zeno_var_parameters_down_listener = [0.0]
Zeno_var_parameters = df_variable_parameters(Zeno_var_parameters_names, Zeno_var_parameters_top_compare, Zeno_var_parameters_down_compare)

# Table of symbolic effects
Zeno_symbolic_preconditions_data = [
    # Variables = ["location_p1","location_p2", "location_p3", "in_p1", "in_p2", "in_p3", "location_a1", "fuel_a1", "onboard_a1", "total_fuel_used"]
        pd.Series(["0.0", ">=0.0", ">=0.0", "0.0", ">=0.0", ">=0.0", "0.0", ">=0.0", ">=0.0", ">=0.0"], index=Zeno_variables_names),
        pd.Series(["10.0", ">=0.0", ">=0.0", "0.0", ">=0.0", ">=0.0", "10.0", ">=0.0", ">=0.0", ">=0.0"], index=Zeno_variables_names),
        pd.Series(["20.0", ">=0.0", ">=0.0", "0.0", ">=0.0", ">=0.0", "20.0", ">=0.0", ">=0.0", ">=0.0"], index=Zeno_variables_names),
        pd.Series([">=0.0", "00.0", ">=0.0", ">=0.0", "0.0", ">=0.0", "0.0", ">=0.0", ">=0.0", ">=0.0"], index=Zeno_variables_names),
        pd.Series([">=0.0", "10.0", ">=0.0", ">=0.0", "0.0", ">=0.0", "10.0", ">=0.0", ">=0.0", ">=0.0"], index=Zeno_variables_names),
        pd.Series([">=0.0", "20.0", ">=0.0", ">=0.0", "0.0", ">=0.0", "20.0", ">=0.0", ">=0.0", ">=0.0"], index=Zeno_variables_names),
        pd.Series([">=0.0", ">=0.0", "0.0", ">=0.0", ">=0.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0"], index=Zeno_variables_names),
        pd.Series([">=0.0", ">=0.0", "10.0", ">=0.0", ">=0.0", "0.0", "10.0", ">=0.0", ">=0.0", ">=0.0"], index=Zeno_variables_names),
        pd.Series([">=0.0", ">=0.0", "20.0", ">=0.0", ">=0.0", "0.0", "20.0", ">=0.0", ">=0.0", ">=0.0"], index=Zeno_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", ">=10.0", ">=0.0", ">=0.0", "0.0", ">=0.0", ">=10.0", ">=0.0"], index=Zeno_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", ">=10.0", ">=0.0", ">=0.0", "10.0", ">=0.0", ">=10.0", ">=0.0"], index=Zeno_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", ">=10.0", ">=0.0", ">=0.0", "20.0", ">=0.0", ">=10.0", ">=0.0"], index=Zeno_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=10.0", ">=0.0", "0.0", ">=0.0", ">=10.0", ">=0.0"], index=Zeno_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=10.0", ">=0.0", "10.0", ">=0.0", ">=10.0", ">=0.0"], index=Zeno_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=10.0", ">=0.0", "20.0", ">=0.0", ">=10.0", ">=0.0"], index=Zeno_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=10.0", "0.0", ">=0.0", ">=10.0", ">=0.0"], index=Zeno_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=10.0", "10.0", ">=0.0", ">=10.0", ">=0.0"], index=Zeno_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=10.0", "20.0", ">=0.0", ">=10.0", ">=0.0"], index=Zeno_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "0.0", ">=(dis_0_1*slow_burn_a1)", ">=0.0", ">=0.0"], index=Zeno_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "0.0", ">=(dis_0_2*slow_burn_a1)", ">=0.0", ">=0.0"], index=Zeno_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "10.0", ">=(dis_1_2*slow_burn_a1)", ">=0.0", ">=0.0"], index=Zeno_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "10.0", ">=(dis_0_1*slow_burn_a1)", ">=0.0", ">=0.0"], index=Zeno_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "20.0", ">=(dis_0_2*slow_burn_a1)", ">=0.0", ">=0.0"], index=Zeno_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "20.0", ">=(dis_1_2*slow_burn_a1)", ">=0.0", ">=0.0"], index=Zeno_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "0.0", ">=(dis_0_1*fast_burn_a1)", "<=zoom_limit_a1", ">=0.0"], index=Zeno_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "0.0", ">=(dis_0_2*fast_burn_a1)", "<=zoom_limit_a1", ">=0.0"], index=Zeno_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "10.0", ">=(dis_1_2*fast_burn_a1)", "<=zoom_limit_a1", ">=0.0"], index=Zeno_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "10.0", ">=(dis_0_1*fast_burn_a1)", "<=zoom_limit_a1", ">=0.0"], index=Zeno_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "20.0", ">=(dis_0_2*fast_burn_a1)", "<=zoom_limit_a1", ">=0.0"], index=Zeno_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "20.0", ">=(dis_1_2*fast_burn_a1)", "<=zoom_limit_a1", ">=0.0"], index=Zeno_variables_names),
        pd.Series([">=3000.0", ">=0.0", "<=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "<capacity_a1", ">=0.0", ">=0.0"], index=Zeno_variables_names)
    ]
Zeno_symbolic_precons = df_subsymbolic_preconditions(Zeno_symbolic_actions, Zeno_variables_names, Zeno_symbolic_preconditions_data)

    # Table of symbolic effects
Zeno_symbolic_effects_data = [
        pd.Series(["-0.0", "0.0", "0.0", "10.0", "0.0", "0.0", "0.0", "0.0", "10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["-10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "0.0", "0.0", "10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["-20.0", "0.0", "0.0", "10.0", "0.0", "0.0", "0.0", "0.0", "10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["0.0", "-0.0", "0.0", "0.0", "10.0", "0.0", "0.0", "0.0", "10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["0.0", "-10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "0.0", "10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["0.0", "-20.0", "0.0", "0.0", "10.0", "0.0", "0.0", "0.0", "10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "-0.0", "0.0", "0.0", "10.0", "0.0", "0.0", "10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "-10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "-20.0", "0.0", "0.0", "10.0", "0.0", "0.0", "10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "0.0", "-10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["10.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "0.0", "-10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["20.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "0.0", "-10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "-10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["0.0", "10.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "-10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["0.0", "20.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "-10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "-10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "10.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "-10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "20.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "-10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "10.0", "-(dis_0_1*slow_burn_a1)", "0.0", "(dis_0_1*slow_burn_a1)"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "20.0", "-(dis_0_2*slow_burn_a1)", "0.0", "(dis_0_2*slow_burn_a1)"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "10.0", "-(dis_1_2*slow_burn_a1)", "0.0", "(dis_1_2*slow_burn_a1)"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "-10.0", "-(dis_0_1*slow_burn_a1)", "0.0", "(dis_0_1*slow_burn_a1)"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "-20.0", "-(dis_0_2*slow_burn_a1)", "0.0", "(dis_0_2*slow_burn_a1)"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "-10.0", "-(dis_1_2*slow_burn_a1)", "0.0", "(dis_1_2*slow_burn_a1)"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "10.0", "-(dis_0_1*fast_burn_a1)", "0.0", "(dis_0_1*fast_burn_a1)"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "20.0", "-(dis_0_2*fast_burn_a1)", "0.0", "(dis_0_2*fast_burn_a1)"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "10.0", "-(dis_1_2*fast_burn_a1)", "0.0", "(dis_1_2*fast_burn_a1)"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "-10.0", "-(dis_0_1*fast_burn_a1)", "0.0", "(dis_0_1*fast_burn_a1)"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "-20.0", "-(dis_0_2*fast_burn_a1)", "0.0", "(dis_0_2*fast_burn_a1)"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "-10.0", "-(dis_1_2*fast_burn_a1)", "0.0", "(dis_1_2*fast_burn_a1)"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "fuel", "0.0", "0.0"], index=Zeno_variables_names)
    ]
Zeno_symbolic_effects = df_symbolic_effects(Zeno_variables_names, Zeno_symbolic_actions, Zeno_symbolic_effects_data)

   # Table of subsymbolic preconditions    
Zeno_subsymbolic_preconditions_data = [
        pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "<capacity_a1", ">=0.0", ">=0.0"], index=Zeno_variables_names),
    ]
Zeno_subsymbolic_precons = df_subsymbolic_preconditions(Zeno_learned_actions, Zeno_variables_names, Zeno_subsymbolic_preconditions_data)

    # Table of subsymbolic effects
Zeno_subsymbolic_effects_data = [
        pd.Series(["model(state_in)"], index=Zeno_learned_actions),
    ]
    #Zeno_subsymbolic_effects = df_subsymbolic_models(Zeno_learned_actions, Zeno_subsymbolic_effects_data)

Zeno_variable_parameters_data = [
        pd.Series(["10.0"], index=Zeno_var_parameters_names)
    ]

    # Table of initial- & goalstate
# Variables = ["location_p1","location_p2", "location_p3", "in_p1", "in_p2", "in_p3", "location_a1", "fuel_a1", "onboard_a1", "total_fuel_used"]
# compare two step
Zeno_init_1 = ["0.0", "0.0", "0.0", "0.0", "10.0", "0.0", "10.0", "100.0", "10.0", "0.0"]
Zeno_goal_1 = ["0.0", "0.0", "0.0", "0.0", "10.0", "0.0", "20.0", ">100.0", "10.0", ">=0.0"]
Zeno_1 = df_problem(Zeno_variables_names, Zeno_init_1, Zeno_goal_1)
        # Action_0     fs_c1_c2    0
        # Action_1  ML_refuel&1    0, ['ML_refuel'])

# compare three step
Zeno_init_2 = ["0.0", "0.0", "0.0", "10.0", "0.0", "0.0", "10.0", "100.0", "10.0", "0.0"]
Zeno_goal_2 = ["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", ">100.0", "0.0", ">=0.0"]
Zeno_2 = df_problem(Zeno_variables_names, Zeno_init_2, Zeno_goal_2)
        # Action_0         fs_c1_c0    0
        # Action_1  debark_a1_p1_c0    0
        # Action_2      ML_refuel&2    0

# compare input
Zeno_init_3 = ["0.0", "0.0", "0.0", "0.0", "10.0", "0.0", "10.0", "100.0", "10.0", "0.0"]
Zeno_goal_3 = ["10.0", "20.0", "10.0", "0.0", "0.0", "0.0", "20.0", ">100.0", "0.0", ">=0.0"]
Zeno_3 = df_problem(Zeno_variables_names, Zeno_init_3, Zeno_goal_3)
        # Action_0         fs_c1_c0    0
        # Action_1   board_a1_p3_c0    0
        # Action_2   board_a1_p1_c0    0
        # Action_3      ML_refuel&3    0
        # Action_4         fs_c0_c1    0
        # Action_5  debark_a1_p1_c1    0
        # Action_6  debark_a1_p3_c1    0
        # Action_7         fs_c1_c2    0
        # Action_8  debark_a1_p2_c2    0

recharge_listener_labels_data = [
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "10.0", "0.0", "0.0"], index=Zeno_variables_names)]
        # pd.Series(["10.0", "10.0", "10.0", "10.0", "10.0", "10.0", "10.0", "10.0", "10.0", "10.0"], index=Zeno_variables_names)]              # # include this to run the planer with no listener information
recharge_listener_under_bounds_data = [
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=Zeno_variables_names)]
recharge_listener_upper_bounds_data = [
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "190.0", "0.0", "0.0"], index=Zeno_variables_names)]

Zeno_subsymbolic_effects_parameters = variable_parameters(Zeno_learned_actions, Zeno_var_parameters_names, Zeno_variable_parameters_data)
Zeno_subsymbolic_labels = df_subsymbolic_label(Zeno_variables_names, Zeno_learned_actions, recharge_listener_labels_data)
Zeno_subsymbolic_under_bounds = df_subsymbolic_under_bound(Zeno_variables_names, Zeno_learned_actions, recharge_listener_under_bounds_data)
Zeno_subsymbolic_upper_bounds = df_subsymbolic_upper_bound(Zeno_variables_names, Zeno_learned_actions, recharge_listener_upper_bounds_data)

Zeno_subsymbolic_effects = subsymbolic_effetcs(Zeno_subsymbolic_labels, Zeno_subsymbolic_under_bounds, Zeno_subsymbolic_upper_bounds, Zeno_subsymbolic_effects_parameters)

Zeno_domain = domain(Zeno_variables_names, Zeno_symbolic_actions, Zeno_learned_actions, Zeno_fix_paraneters, Zeno_var_parameters, Zeno_symbolic_precons, Zeno_symbolic_effects, Zeno_subsymbolic_precons,Zeno_subsymbolic_effects)
Zeno_domains = input(Zeno_domain, Zeno_3)

def create_zeno_dataset(n_samples):
    Zeno_variables_names = ["location_p1","location_p2", "location_p3", "in_p1", "in_p2", "in_p3", "location_a1", "fuel_a1", "onboard_a1", "total_fuel_used"]
    Zeno_symbolic_actions = ["board_a1_p1_c0", "board_a1_p1_c1", "board_a1_p1_c2", "board_a1_p2c0", "board_a1_p2_c1", "board_a1_p2_c2", "board_a1_p3_c0", "board_a1_p3_c1", "board_a1_p3_c2",
                "debark_a1_p1_c0", "debark_a1_p1_c1", "debark_a1_p1_c2", "debark_a1_p2_c0", "debark_a1_p2_c1", "debark_a1_p2_c2", "debark_a1_p3_c0", "debark_a1_p3_c1", "debark_a1_p3_c2",
                "fs_c0_c1", "fs_c0_c2", "fs_c1_c2", "fs_c1_c0", "fs_c2_c0", "fs_c2_c1", 
                "ff_c0_c1", "ff_c0_c2", "ff_c1_c2", "ff_c1_c0", "ff_c2_c0", "ff_c2_c1",
                "refuel"]
    Zeno_learned_actions = ["ML_refuel"]

        # Table of fix parameters
    Zeno_fix_parameters_names = ["dis_0_1", "dis_0_2", "dis_1_2", "slow_burn_a1", "fast_burn_a1", "capacity_a1", "zoom_limit_a1"]
    Zeno_fix_parameters_values = ["6.0", "7.0", "8.0", "4.0", "1.50", "100.0", "8.0"]
    Zeno_fix_paraneters = df_fix_parameters(Zeno_fix_parameters_names, Zeno_fix_parameters_values)

        # Table of variable parameters
    Zeno_var_parameters_names = ["fuel"]
    Zeno_var_parameters_top = [100.0]
    Zeno_var_parameters_down = [0.0]
    Zeno_var_parameters = df_variable_parameters(Zeno_var_parameters_names, Zeno_var_parameters_top, Zeno_var_parameters_down)

    # Table of symbolic effects
    Zeno_symbolic_preconditions_data = [
        # Variables = ["location_p1","location_p2", "location_p3", "in_p1", "in_p2", "in_p3", "location_a1", "fuel_a1", "onboard_a1", "total_fuel_used"]
            pd.Series(["0.0", ">=0.0", ">=0.0", "0.0", ">=0.0", ">=0.0", "0.0", ">=0.0", ">=0.0", ">=0.0"], index=Zeno_variables_names),
            pd.Series(["10.0", ">=0.0", ">=0.0", "0.0", ">=0.0", ">=0.0", "10.0", ">=0.0", ">=0.0", ">=0.0"], index=Zeno_variables_names),
            pd.Series(["20.0", ">=0.0", ">=0.0", "0.0", ">=0.0", ">=0.0", "20.0", ">=0.0", ">=0.0", ">=0.0"], index=Zeno_variables_names),
            pd.Series([">=0.0", "00.0", ">=0.0", ">=0.0", "0.0", ">=0.0", "0.0", ">=0.0", ">=0.0", ">=0.0"], index=Zeno_variables_names),
            pd.Series([">=0.0", "10.0", ">=0.0", ">=0.0", "0.0", ">=0.0", "10.0", ">=0.0", ">=0.0", ">=0.0"], index=Zeno_variables_names),
            pd.Series([">=0.0", "20.0", ">=0.0", ">=0.0", "0.0", ">=0.0", "20.0", ">=0.0", ">=0.0", ">=0.0"], index=Zeno_variables_names),
            pd.Series([">=0.0", ">=0.0", "0.0", ">=0.0", ">=0.0", "0.0", "0.0", ">=0.0", ">=0.0", ">=0.0"], index=Zeno_variables_names),
            pd.Series([">=0.0", ">=0.0", "10.0", ">=0.0", ">=0.0", "0.0", "10.0", ">=0.0", ">=0.0", ">=0.0"], index=Zeno_variables_names),
            pd.Series([">=0.0", ">=0.0", "20.0", ">=0.0", ">=0.0", "0.0", "20.0", ">=0.0", ">=0.0", ">=0.0"], index=Zeno_variables_names),
            pd.Series([">=0.0", ">=0.0", ">=0.0", ">=10.0", ">=0.0", ">=0.0", "0.0", ">=0.0", ">=10.0", ">=0.0"], index=Zeno_variables_names),
            pd.Series([">=0.0", ">=0.0", ">=0.0", ">=10.0", ">=0.0", ">=0.0", "10.0", ">=0.0", ">=10.0", ">=0.0"], index=Zeno_variables_names),
            pd.Series([">=0.0", ">=0.0", ">=0.0", ">=10.0", ">=0.0", ">=0.0", "20.0", ">=0.0", ">=10.0", ">=0.0"], index=Zeno_variables_names),
            pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=10.0", ">=0.0", "0.0", ">=0.0", ">=10.0", ">=0.0"], index=Zeno_variables_names),
            pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=10.0", ">=0.0", "10.0", ">=0.0", ">=10.0", ">=0.0"], index=Zeno_variables_names),
            pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=10.0", ">=0.0", "20.0", ">=0.0", ">=10.0", ">=0.0"], index=Zeno_variables_names),
            pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=10.0", "0.0", ">=0.0", ">=10.0", ">=0.0"], index=Zeno_variables_names),
            pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=10.0", "10.0", ">=0.0", ">=10.0", ">=0.0"], index=Zeno_variables_names),
            pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=10.0", "20.0", ">=0.0", ">=10.0", ">=0.0"], index=Zeno_variables_names),
            pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "0.0", ">=(dis_0_1*slow_burn_a1)", ">=0.0", ">=0.0"], index=Zeno_variables_names),
            pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "0.0", ">=(dis_0_2*slow_burn_a1)", ">=0.0", ">=0.0"], index=Zeno_variables_names),
            pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "10.0", ">=(dis_1_2*slow_burn_a1)", ">=0.0", ">=0.0"], index=Zeno_variables_names),
            pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "10.0", ">=(dis_0_1*slow_burn_a1)", ">=0.0", ">=0.0"], index=Zeno_variables_names),
            pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "20.0", ">=(dis_0_2*slow_burn_a1)", ">=0.0", ">=0.0"], index=Zeno_variables_names),
            pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "20.0", ">=(dis_1_2*slow_burn_a1)", ">=0.0", ">=0.0"], index=Zeno_variables_names),
            pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "0.0", ">=(dis_0_1*fast_burn_a1)", "<=zoom_limit_a1", ">=0.0"], index=Zeno_variables_names),
            pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "0.0", ">=(dis_0_2*fast_burn_a1)", "<=zoom_limit_a1", ">=0.0"], index=Zeno_variables_names),
            pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "10.0", ">=(dis_1_2*fast_burn_a1)", "<=zoom_limit_a1", ">=0.0"], index=Zeno_variables_names),
            pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "10.0", ">=(dis_0_1*fast_burn_a1)", "<=zoom_limit_a1", ">=0.0"], index=Zeno_variables_names),
            pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "20.0", ">=(dis_0_2*fast_burn_a1)", "<=zoom_limit_a1", ">=0.0"], index=Zeno_variables_names),
            pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "20.0", ">=(dis_1_2*fast_burn_a1)", "<=zoom_limit_a1", ">=0.0"], index=Zeno_variables_names),
            pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "<capacity_a1", ">=0.0", ">=0.0"], index=Zeno_variables_names)
        ]
    Zeno_symbolic_precons = df_subsymbolic_preconditions(Zeno_symbolic_actions, Zeno_variables_names, Zeno_symbolic_preconditions_data)

        # Table of symbolic effects
    Zeno_symbolic_effects_data = [
            pd.Series(["-0.0", "0.0", "0.0", "10.0", "0.0", "0.0", "0.0", "0.0", "10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["-10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "0.0", "0.0", "10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["-20.0", "0.0", "0.0", "10.0", "0.0", "0.0", "0.0", "0.0", "10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["0.0", "-0.0", "0.0", "0.0", "10.0", "0.0", "0.0", "0.0", "10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["0.0", "-10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "0.0", "10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["0.0", "-20.0", "0.0", "0.0", "10.0", "0.0", "0.0", "0.0", "10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "-0.0", "0.0", "0.0", "10.0", "0.0", "0.0", "10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "-10.0", "0.0", "0.0", "10.0", "0.0", "0.0", "10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "-20.0", "0.0", "0.0", "10.0", "0.0", "0.0", "10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "0.0", "-10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["10.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "0.0", "-10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["20.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "0.0", "-10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "-10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["0.0", "10.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "-10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["0.0", "20.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "0.0", "-10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "-10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "10.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "-10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "20.0", "0.0", "0.0", "-10.0", "0.0", "0.0", "-10.0", "0.0"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "10.0", "-(dis_0_1*slow_burn_a1)", "0.0", "(dis_0_1*slow_burn_a1)"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "20.0", "-(dis_0_2*slow_burn_a1)", "0.0", "(dis_0_2*slow_burn_a1)"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "10.0", "-(dis_1_2*slow_burn_a1)", "0.0", "(dis_1_2*slow_burn_a1)"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "-10.0", "-(dis_0_1*slow_burn_a1)", "0.0", "(dis_0_1*slow_burn_a1)"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "-20.0", "-(dis_0_2*slow_burn_a1)", "0.0", "(dis_0_2*slow_burn_a1)"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "-10.0", "-(dis_1_2*slow_burn_a1)", "0.0", "(dis_1_2*slow_burn_a1)"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "10.0", "-(dis_0_1*fast_burn_a1)", "0.0", "(dis_0_1*fast_burn_a1)"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "20.0", "-(dis_0_2*fast_burn_a1)", "0.0", "(dis_0_2*fast_burn_a1)"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "10.0", "-(dis_1_2*fast_burn_a1)", "0.0", "(dis_1_2*fast_burn_a1)"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "-10.0", "-(dis_0_1*fast_burn_a1)", "0.0", "(dis_0_1*fast_burn_a1)"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "-20.0", "-(dis_0_2*fast_burn_a1)", "0.0", "(dis_0_2*fast_burn_a1)"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "-10.0", "-(dis_1_2*fast_burn_a1)", "0.0", "(dis_1_2*fast_burn_a1)"], index=Zeno_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "fuel", "0.0", "0.0"], index=Zeno_variables_names)
        ]
    Zeno_symbolic_effects = df_symbolic_effects(Zeno_variables_names, Zeno_symbolic_actions, Zeno_symbolic_effects_data)

    # Table of subsymbolic preconditions    
    Zeno_subsymbolic_preconditions_data = [
            pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=0.0", "<capacity_a1", ">=0.0", ">=0.0"], index=Zeno_variables_names),
        ]
    Zeno_subsymbolic_precons = df_subsymbolic_preconditions(Zeno_learned_actions, Zeno_variables_names, Zeno_subsymbolic_preconditions_data)

        # Table of subsymbolic effects
    Zeno_subsymbolic_effects_data = [
            pd.Series(["model(state_in)"], index=Zeno_learned_actions),
        ]
        #Zeno_subsymbolic_effects = df_subsymbolic_models(Zeno_learned_actions, Zeno_subsymbolic_effects_data)

    Zeno_variable_parameters_data = [
            pd.Series(["10.0"], index=Zeno_var_parameters_names)
        ]

        # Table of initial- & goalstate
    Zeno_init_1 = ["20.0", "0.0", "10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"]
    Zeno_goal_1 = ["10.0", "20.0", "0.0", "0.0", "0.0", "0.0", "0.0", ">=0.0", "0.0", ">=0.0"]
    Zeno_1 = df_problem(Zeno_variables_names, Zeno_init_1, Zeno_goal_1)

    recharge_listener_labels_data = [
            pd.Series(["20.0", "0.0", "10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=Zeno_variables_names)]
    recharge_listener_under_bounds_data = [
            pd.Series(["20.0", "0.0", "10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=Zeno_variables_names)]
    recharge_listener_upper_bounds_data = [
            pd.Series(["20.0", "0.0", "10.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=Zeno_variables_names)]

    Zeno_subsymbolic_effects_parameters = variable_parameters(Zeno_learned_actions, Zeno_var_parameters_names, Zeno_variable_parameters_data)
    Zeno_subsymbolic_labels = df_subsymbolic_label(Zeno_variables_names, Zeno_learned_actions, recharge_listener_labels_data)
    Zeno_subsymbolic_under_bounds = df_subsymbolic_under_bound(Zeno_variables_names, Zeno_learned_actions, recharge_listener_under_bounds_data)
    Zeno_subsymbolic_upper_bounds = df_subsymbolic_upper_bound(Zeno_variables_names, Zeno_learned_actions, recharge_listener_upper_bounds_data)

    Zeno_subsymbolic_effects = subsymbolic_effetcs(Zeno_subsymbolic_labels, Zeno_subsymbolic_under_bounds, Zeno_subsymbolic_upper_bounds, Zeno_subsymbolic_effects_parameters)

    Zeno_domain = domain(Zeno_variables_names, Zeno_symbolic_actions, Zeno_learned_actions, Zeno_fix_paraneters, Zeno_var_parameters, Zeno_symbolic_precons, Zeno_symbolic_effects, Zeno_subsymbolic_precons,Zeno_subsymbolic_effects)
    Zeno_domains = input(Zeno_domain, Zeno_1)

    set_1 = data_set(n_samples, Zeno_variables_names, Zeno_fix_parameters_names, Zeno_var_parameters_names)
    set_1_input, set2_output = set_1.generate()

    return set_1_input, set2_output

# Quicktest
if __name__ == "__main__":
    set_1_input, set2_output = create_zeno_dataset(500)
    print(set_1_input[1])
    print(set2_output[1])
    print("yippee")