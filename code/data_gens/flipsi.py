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

        Parameter_var = ["energy_input"]
        Parameter_fix = ["volume", "density", "specific_heat_capacity"]
        Features = ["coloured", "temperature", "milled", "drilled", "rounded"]

        bool_ls = [0, 10]

        volume_top = 2
        volume_down = 1
        density_top = 10000
        density_down = 2000
        specific_heat_capacity_top = 1000
        specific_heat_capacity_down = 250

        energy_input_down = 0

        State_in_temp_top = 100
        State_in_temp_down = 0

        for i in range(self.setlength):
            State_in[i][0] = random.choice(bool_ls)
            State_out[i][0] = State_in[i][0]
            State_in[i][2] = random.choice(bool_ls)
            State_out[i][2] = State_in[i][2]
            State_in[i][3] = random.choice(bool_ls)
            State_out[i][3] = State_in[i][3]
            State_in[i][4] = random.choice(bool_ls)
            State_out[i][4] = State_in[i][4]
            Parameters_fix[i][0] = random.uniform(volume_down, volume_top)
            Parameters_fix[i][1] = random.uniform(density_down, density_top)
            Parameters_fix[i][2] = random.uniform(specific_heat_capacity_down, specific_heat_capacity_top)
            State_in[i][1] = random.uniform(State_in_temp_down, State_in_temp_top)
            energy_input_top = (100 - State_in[i][1]) * (Parameters_fix[i][0] * Parameters_fix[i][1] * Parameters_fix[i][2])
            Parameters_var[i][0] = random.uniform(energy_input_down, energy_input_top)
            State_out[i][1] = State_in[i][1] + (Parameters_var[i][0] * (((Parameters_fix[i][0] * Parameters_fix[i][1] * Parameters_fix[i][2]) ** (-1))))

        Input = np.concatenate((State_in, Parameters_fix, Parameters_var), axis =1 )
        Output = State_out

        return Input, Output
    
FliPsi_variables_names = ["coloured", "temperature", "milled", "drilled", "rounded"]
FliPsi_symbolic_actions = ["Mill", "Drill", "CNC", "Paint", "Heat"]
FliPsi_learned_actions = ["L_Heat"]

    # Table of fix parameters
FliPsi_fix_parameters_names = ["volume", "density", "specific_heat_capacity"]
FliPsi_fix_parameters_values = ["1.5", "8730", "377"]
FliPsi_fix_paraneters = df_fix_parameters(FliPsi_fix_parameters_names, FliPsi_fix_parameters_values)

    # Table of variable parameters
FliPsi_var_parameters_names = ["energy_input"]
FliPsi_var_parameters_top_compare = [100]
FliPsi_var_parameters_down_compare = [0]
FliPsi_var_parameters_top_listener = [2000.0]
FliPsi_var_parameters_down_listener = [0.0]
FliPsi_var_parameters = df_variable_parameters(FliPsi_var_parameters_names, FliPsi_var_parameters_top_compare, FliPsi_var_parameters_down_compare)

FliPsi_symbolic_preconditions_data = [
        pd.Series(["0.0", "", "0.0", "", ""], index=FliPsi_variables_names),
        pd.Series(["", "", "", "0.0", "0.0"], index=FliPsi_variables_names),
        pd.Series(["", "", "", "", "0.0"], index=FliPsi_variables_names),
        pd.Series(["0.0", ">=20", "", "", ""], index=FliPsi_variables_names),
        pd.Series(["<=10", "<=0.0", "", "", ""], index=FliPsi_variables_names),
    ]
FliPsi_symbolic_precons = df_subsymbolic_preconditions(FliPsi_symbolic_actions, FliPsi_variables_names, FliPsi_symbolic_preconditions_data)

    # Table of symbolic effects
FliPsi_symbolic_effects_data = [
        pd.Series(["0.0", "0.0", "10", "0.0", "0.0"], index=FliPsi_variables_names),
        pd.Series(["0", "0", "0", "10", "0"], index=FliPsi_variables_names),
        pd.Series(["0", "0", "0", "0", "10"], index=FliPsi_variables_names),
        pd.Series(["10", "0", "0", "0", "0"], index=FliPsi_variables_names),
        pd.Series(["0", "(energy_input*1000000)*((volume*density*specific_heat_capacity*1000*100)**(-1))", "0", "0", "0"], index=FliPsi_variables_names)
    ]
FliPsi_symbolic_effects = df_symbolic_effects(FliPsi_variables_names, FliPsi_symbolic_actions, FliPsi_symbolic_effects_data)

# Table of subsymbolic preconditions    
FliPsi_subsymbolic_preconditions_data = [
        pd.Series(["", "<=100", "", "", ""], index=FliPsi_variables_names),
    ]
FliPsi_subsymbolic_precons = df_subsymbolic_preconditions(FliPsi_learned_actions, FliPsi_variables_names, FliPsi_subsymbolic_preconditions_data)

    # Table of subsymbolic effects
FliPsi_subsymbolic_effects_data = [
        pd.Series(["model(state_in)"], index=FliPsi_learned_actions),
    ]

FliPsi_variable_parameters_data = [
        pd.Series(["10.0"], index=FliPsi_var_parameters_names)
    ]

# compare 2 step 
FliPsi_init_1 = ["0.0", "5.0", "0.0", "0.0", "0.0"]
FliPsi_Goal_1 = ["10", "20.0", "0", "0", "0"]
FliPsi_1 = df_problem(FliPsi_variables_names, FliPsi_init_1, FliPsi_Goal_1)

# compare 3 step
FliPsi_init_2 = ["0.0", "5.0", "0.0", "0.0", "0.0"]
FliPsi_goal_2 = ["10", "20.0", "10", "0", "0"]
FliPsi_2 = df_problem(FliPsi_variables_names, FliPsi_init_2, FliPsi_goal_2)

# compare 5 steps (RAINER 1)
FliPsi_init_3 = ["0.0", "5.0", "0.0", "0.0", "0.0"]
FliPsi_goal_3 = ["10", "20.0", "10", "10", "10"]
FliPsi_3 = df_problem(FliPsi_variables_names, FliPsi_init_3, FliPsi_goal_3)

recharge_listener_labels_data = [
        pd.Series(["10.0", "10.0", "10.0", "10.0", "10.0"], index=FliPsi_variables_names)]
        # pd.Series(["0.0", "10.0", "0.0", "0.0", "0.0"], index=FliPsi_variables_names)]                # include this to run the planer with no listener information
recharge_listener_under_bounds_data = [
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0"], index=FliPsi_variables_names)]
recharge_listener_upper_bounds_data = [
        pd.Series(["0.0", "100.0", "0.0", "0.0", "0.0"], index=FliPsi_variables_names)]

FliPsi_subsymbolic_effects_parameters = variable_parameters(FliPsi_learned_actions, FliPsi_var_parameters_names, FliPsi_variable_parameters_data)
FliPsi_subsymbolic_labels = df_subsymbolic_label(FliPsi_variables_names, FliPsi_learned_actions, recharge_listener_labels_data)
FliPsi_subsymbolic_under_bounds = df_subsymbolic_under_bound(FliPsi_variables_names, FliPsi_learned_actions, recharge_listener_under_bounds_data)
FliPsi_subsymbolic_upper_bounds = df_subsymbolic_upper_bound(FliPsi_variables_names, FliPsi_learned_actions, recharge_listener_upper_bounds_data)

FliPsi_subsymbolic_effects = subsymbolic_effetcs(FliPsi_subsymbolic_labels, FliPsi_subsymbolic_under_bounds, FliPsi_subsymbolic_upper_bounds, FliPsi_subsymbolic_effects_parameters)

FliPsi_domain = domain(FliPsi_variables_names, FliPsi_symbolic_actions, FliPsi_learned_actions, FliPsi_fix_paraneters, FliPsi_var_parameters, FliPsi_symbolic_precons, FliPsi_symbolic_effects, FliPsi_subsymbolic_precons, FliPsi_subsymbolic_effects)
FliPsi_domains = input(FliPsi_domain, FliPsi_3)

def create_FliPsi_dataset(n_samples):
    FliPsi_variables_names = ["coloured", "temperature", "milled", "drilled", "rounded"]
    FliPsi_symbolic_actions = ["Mill", "Drill", "CNC", "Paint", "Heat"]
    FliPsi_learned_actions = ["L_Heat"]

        # Table of fix parameters
    FliPsi_fix_parameters_names = ["volume", "density", "specific_heat_capacity"]
    FliPsi_fix_parameters_values = ["1.5", "8730", "377"]
    FliPsi_fix_paraneters = df_fix_parameters(FliPsi_fix_parameters_names, FliPsi_fix_parameters_values)

        # Table of variable parameters
    FliPsi_var_parameters_names = ["energy_input"]
    FliPsi_var_parameters_top = [100000.0]
    FliPsi_var_parameters_down = [0.0]
    FliPsi_var_parameters = df_variable_parameters(FliPsi_var_parameters_names, FliPsi_var_parameters_top, FliPsi_var_parameters_down)

    FliPsi_symbolic_preconditions_data = [
            pd.Series(["0.0", "", "0.0", "", ""], index=FliPsi_variables_names),
            pd.Series(["", "", "", "0.0", "0.0"], index=FliPsi_variables_names),
            pd.Series(["", "", "", "", "0.0"], index=FliPsi_variables_names),
            pd.Series(["0.0", ">=20", "", "", ""], index=FliPsi_variables_names),
            pd.Series(["<=10", "<=100.0", "", "", ""], index=FliPsi_variables_names),
        ]
    FliPsi_symbolic_precons = df_subsymbolic_preconditions(FliPsi_symbolic_actions, FliPsi_variables_names, FliPsi_symbolic_preconditions_data)

        # Table of symbolic effects
    FliPsi_symbolic_effects_data = [
            pd.Series(["0.0", "0.0", "10", "0.0", "0.0"], index=FliPsi_variables_names),
            pd.Series(["0", "0", "0", "10", "0"], index=FliPsi_variables_names),
            pd.Series(["0", "0", "0", "0", "10"], index=FliPsi_variables_names),
            pd.Series(["10", "0", "0", "0", "0"], index=FliPsi_variables_names),
            pd.Series(["0", "(energy_input*1000000)*((volume*density*specific_heat_capacity*1000*100)**(-1))", "0", "0", "0"], index=FliPsi_variables_names)
        ]
    FliPsi_symbolic_effects = df_symbolic_effects(FliPsi_variables_names, FliPsi_symbolic_actions, FliPsi_symbolic_effects_data)

    # Table of subsymbolic preconditions    
    FliPsi_subsymbolic_preconditions_data = [
            pd.Series(["", "<=100", "", "", ""], index=FliPsi_variables_names),
        ]
    FliPsi_subsymbolic_precons = df_subsymbolic_preconditions(FliPsi_learned_actions, FliPsi_variables_names, FliPsi_subsymbolic_preconditions_data)

        # Table of subsymbolic effects
    FliPsi_subsymbolic_effects_data = [
            pd.Series(["model(state_in)"], index=FliPsi_learned_actions),
        ]

    FliPsi_variable_parameters_data = [
            pd.Series(["10.0"], index=FliPsi_var_parameters_names)
        ]

    FliPsi_init_1 = pd.Series(["0.0", "5.0", "0.0", "0.0", "0.0"], index=FliPsi_variables_names)
    FliPsi_Goal_1 = pd.Series(["10", "20.0", "10", "10", "10"], index=FliPsi_variables_names)
    FliPsi_1 = df_problem(FliPsi_variables_names, FliPsi_init_1, FliPsi_Goal_1)

    recharge_listener_labels_data = [
            pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0"], index=FliPsi_variables_names)]
    recharge_listener_under_bounds_data = [
            pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0"], index=FliPsi_variables_names)]
    recharge_listener_upper_bounds_data = [
            pd.Series(["0.0", "0.0", "0.0", "0.0", "100.0"], index=FliPsi_variables_names)]

    FliPsi_subsymbolic_effects_parameters = variable_parameters(FliPsi_learned_actions, FliPsi_var_parameters_names, FliPsi_variable_parameters_data)
    FliPsi_subsymbolic_labels = df_subsymbolic_label(FliPsi_variables_names, FliPsi_learned_actions, recharge_listener_labels_data)
    FliPsi_subsymbolic_under_bounds = df_subsymbolic_under_bound(FliPsi_variables_names, FliPsi_learned_actions, recharge_listener_under_bounds_data)
    FliPsi_subsymbolic_upper_bounds = df_subsymbolic_upper_bound(FliPsi_variables_names, FliPsi_learned_actions, recharge_listener_upper_bounds_data)

    FliPsi_subsymbolic_effects = subsymbolic_effetcs(FliPsi_subsymbolic_labels, FliPsi_subsymbolic_under_bounds, FliPsi_subsymbolic_upper_bounds, FliPsi_subsymbolic_effects_parameters)

    FliPsi_domain = domain(FliPsi_variables_names, FliPsi_symbolic_actions, FliPsi_learned_actions, FliPsi_fix_paraneters, FliPsi_var_parameters, FliPsi_symbolic_precons, FliPsi_symbolic_effects, FliPsi_subsymbolic_precons, FliPsi_subsymbolic_effects)
    FliPsi_domains = input(FliPsi_domain, FliPsi_1)

    set_1 = data_set(n_samples, FliPsi_variables_names, FliPsi_fix_parameters_names, FliPsi_var_parameters_names)
    set_1_input, set2_output = set_1.generate()

    return set_1_input, set2_output

# Quicktest
if __name__ == "__main__":
    set_1_input, set2_output = create_FliPsi_dataset(500)
    print("yippee")
