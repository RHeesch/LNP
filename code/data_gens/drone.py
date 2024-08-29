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

    def generate_charge(self):
        dimensions_States = (self.setlength, self.features)
        dimensions_fix_Parameters = (self.setlength, self.number_fix_Parameters)
        dimensions_var_Parameters = (self.setlength, self.number_variable_Parameters)

        State_in = np.zeros(dimensions_States)
        State_out = np.zeros(dimensions_States)
        Parameters_fix = np.zeros(dimensions_fix_Parameters)
        Parameters_var = np.zeros(dimensions_var_Parameters)

        ### start individual data generator
        # Parameter_var = ["charge"]
        # Parameter_fix = ["battery_level_full", "min_x", "min_y", "min_z", "max_x", "max_y", "max_z"]
        # Features = ["visited_loc1", "visited_loc2", "x-axis", "y-axis", "z-axis", "battery_level"]

        bool_ls = [0, 10]

        battery_level_full_top = 60
        battery_level_full_down = 10
        min_x_top = 20
        min_x_down = 0
        min_y_top = 20
        min_y_down = 0
        min_z_top = 20
        min_z_down = 0
        max_x_top = 50
        max_x_down = 30
        max_y_top = 50
        max_y_down = 30
        max_z_top = 50
        max_z_down = 30

        battery_level_top = 90
        battery_level_down = 0

        for i in range(len(State_out)):
            
            Parameters_fix[i][1] = random.randrange(min_x_down, min_x_top, 10)
            Parameters_fix[i][2] = random.randrange(min_y_down, min_y_top, 10)
            Parameters_fix[i][3] = random.randrange(min_z_down, min_z_top, 10)
            Parameters_fix[i][4] = random.randrange(max_x_down, max_x_top, 10)
            Parameters_fix[i][5] = random.randrange(max_y_down, max_y_top, 10)
            Parameters_fix[i][6] = random.randrange(max_z_down, max_z_top, 10)

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
            State_in[i][5] = random.randrange(battery_level_down, battery_level_top, 10)

            Parameters_var[i][0] = random.randrange(battery_level_full_down, battery_level_full_top, 10)

            Parameters_fix[i][0] = 150

            State_out[i][5] = State_in[i][5] + Parameters_var[i][0]

            #Parameters_var[i][0] = Parameters_fix[i][0] - State_in[i][5]
        ### end individual data generator

        Input = np.concatenate((State_in, Parameters_fix, Parameters_var), axis =1 )
        Output = State_out

        print(Input[1])
        print(Output[1])

        return Input, Output

Drone_variables_names = ["visited_loc1","visited_loc2", "x_axis", "y_axis", "z_axis", "battery_level"]
Drone_symbolic_actions = ["in_x", "in_y", "in_z", "de_x", "de_y", "de_z", "visit_l1", "visit_l2", "recharge"]
Drone_learned_actions = ["L_charge"]

    # Table of fix parameters
Drone_fix_parameters_names = ["battery_level_full", "min_x", "min_y", "min_z", "max_x", "max_y", "max_z"]
Drone_fix_parameters_values = ["200.0", "0.0", "0.0", "0.0", "40.0", "40.0", "40.0"]
Drone_fix_paraneters = df_fix_parameters(Drone_fix_parameters_names, Drone_fix_parameters_values)

    # Table of variable parameters
Drone_var_parameters_names = ["energy"]
Drone_var_parameters_top_compare = [130.0]
Drone_var_parameters_down_compare = [0.0]
Drone_var_parameters_top_listener = [100.0]
Drone_var_parameters_down_listener = [0.0]
Drone_var_parameters = df_variable_parameters(Drone_var_parameters_names, Drone_var_parameters_top_compare, Drone_var_parameters_down_compare)
# Drone_var_parameters = df_variable_parameters(Drone_var_parameters_names, Drone_var_parameters_top_listener, Drone_var_parameters_down_listener)

    # Table of symbolic preconditions
drone_symbolic_preconditions_data = [
        pd.Series([">=0.0", ">=0.0", "<=max_x-10.0", ">=0.0", ">=0.0", ">=10.0"], index=Drone_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", "<=max_y-10.0", ">=0.0", ">=10.0"], index=Drone_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", "<=max_z-10.0", ">=10.0"], index=Drone_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=min_x+10.0", ">=0.0", ">=0.0", ">=10.0"], index=Drone_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", ">=min_y+10.0", ">=0.0", ">=10.0"], index=Drone_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=min_z+10.0", ">=10.0"], index=Drone_variables_names),
        pd.Series([">=0.0", ">=0.0", "0.0", "0.0", "0.0", ">=10.0"], index=Drone_variables_names),
        # pd.Series([">=0.0", ">=0.0", "10.0", "0.0", "0.0", ">=10.0"], index=Drone_variables_names),
        pd.Series([">=0.0", ">=0.0", "20.0", "20.0", "20.0", ">=10.0"], index=Drone_variables_names),
        pd.Series([">=0.0", ">=0.0", "0.0", "0.0", "0.0", ">10000000"], index=Drone_variables_names)
        # pd.Series([">=0.0", ">=0.0", "0.0", "0.0", "0.0", "<battery_level_full"], index=Drone_variables_names)
    ]
drone_symbolic_precons = df_subsymbolic_preconditions(Drone_symbolic_actions, Drone_variables_names, drone_symbolic_preconditions_data)

    # Table of symbolic effects
drone_symbolic_effects_data = [
        pd.Series(["0.0", "0.0", "10.0", "0.0", "0.0", "-10.0"], index=Drone_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "10.0", "0.0", "-10.0"], index=Drone_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "10.0", "-10.0"], index=Drone_variables_names),
        pd.Series(["0.0", "0.0", "-10.0", "0.0", "0.0", "-10.0"], index=Drone_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "-10.0", "0.0", "-10.0"], index=Drone_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "-10.0", "-10.0"], index=Drone_variables_names),
        pd.Series(["10.0", "0.0", "0.0", "0.0", "0.0", "-10.0"], index=Drone_variables_names),
        pd.Series(["0.0", "10.0", "0.0", "0.0", "0.0", "-10.0"], index=Drone_variables_names),
        # pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "(battery_level_full-battery_level)"], index=Drone_variables_names)
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "energy"], index=Drone_variables_names)
    ]
drone_symbolic_effects = df_symbolic_effects(Drone_variables_names, Drone_symbolic_actions, drone_symbolic_effects_data)

    # Table of subsymbolic preconditions    
drone_subsymbolic_preconditions_data = [
        pd.Series([">=0.0", ">=0.0", "0.0", "0.0", "0.0", "<battery_level_full"], index=Drone_variables_names),
    ]
drone_subsymbolic_precons = df_subsymbolic_preconditions(Drone_learned_actions, Drone_variables_names, drone_subsymbolic_preconditions_data)

    # Table of subsymbolic effects
drone_subsymbolic_effects_data = [
        pd.Series(["model(state_in)"], index=Drone_learned_actions),
    ]
    #drone_subsymbolic_effects = df_subsymbolic_models(Drone_learned_actions, drone_subsymbolic_effects_data)

drone_variable_parameters_data = [
        pd.Series(["10.0"], index=Drone_var_parameters_names)
    ]

    # Table of initial- & goalstate
# compare 3 step
Drone_init_1 = ["0.0", "0.0", "10.0", "0.0", "0.0", "90"]
Drone_goal_1 = ["10.0", "0.0", "0.0", "0.0", "0.0", ">90"]
Drone_1 = df_problem(Drone_variables_names, Drone_init_1, Drone_goal_1)

# compare 2 step
Drone_init_2 = ["0.0", "0.0", "10.0", "0.0", "0.0", "90"]
Drone_goal_2 = ["0.0", "0.0", "0.0", "0.0", "0.0", ">90"]
Drone_2 = df_problem(Drone_variables_names, Drone_init_2, Drone_goal_2)

# compare input
Drone_init_3 = ["0.0", "0.0", "0.0", "0.0", "0.0", "90"]
Drone_goal_3 = ["10.0", "10.0", "0.0", "0.0", "0.0", ">90"]
Drone_3 = df_problem(Drone_variables_names, Drone_init_3, Drone_goal_3)


recharge_listener_labels_data = [
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "10.0"], index=Drone_variables_names)]
        # pd.Series(["10.0", "10.0", "10.0", "10.0", "10.0", "10.0"], index=Drone_variables_names)]         # include this to run the planer with no listener information
recharge_listener_under_bounds_data = [
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=Drone_variables_names)]
recharge_listener_upper_bounds_data = [
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "130.0"], index=Drone_variables_names)]

drone_subsymbolic_effects_parameters = variable_parameters(Drone_learned_actions, Drone_var_parameters_names, drone_variable_parameters_data)
drone_subsymbolic_labels = df_subsymbolic_label(Drone_variables_names, Drone_learned_actions, recharge_listener_labels_data)
drone_subsymbolic_under_bounds = df_subsymbolic_under_bound(Drone_variables_names, Drone_learned_actions, recharge_listener_under_bounds_data)
drone_subsymbolic_upper_bounds = df_subsymbolic_upper_bound(Drone_variables_names, Drone_learned_actions, recharge_listener_upper_bounds_data)

drone_subsymbolic_effects = subsymbolic_effetcs(drone_subsymbolic_labels, drone_subsymbolic_under_bounds, drone_subsymbolic_upper_bounds, drone_subsymbolic_effects_parameters)

drone_domain = domain(Drone_variables_names, Drone_symbolic_actions, Drone_learned_actions, Drone_fix_paraneters, Drone_var_parameters, drone_symbolic_precons, drone_symbolic_effects, drone_subsymbolic_precons,drone_subsymbolic_effects)
drone_domains = input(drone_domain, Drone_3)

def print_all():
        print(drone_domains.domain.fix_parameters.df)
        print(drone_domains.domain.variable_parameters.df)
        print(drone_domains.domain.symbolic_preconditions.df)
        print(drone_domains.domain.symbolic_effects.df)
        print(drone_domains.domain.subsymbolic_preconditions.df)
        print(drone_domains.domain.subsymbolic_effetcs.labels.df)
        print(drone_domains.domain.subsymbolic_effetcs.under_bounds.df)
        print(drone_domains.domain.subsymbolic_effetcs.upper_bounds.df)
        print(drone_subsymbolic_effects_parameters.df)
        print(drone_domains.problem.df)

    #print_all()

def create_drone_dataset(n_samples):
    Drone_variables_names = ["visited_loc1","visited_loc2", "x_axis", "y_axis", "z_axis", "battery_level"]
    Drone_symbolic_actions = ["in_x", "in_y", "in_z", "de_x", "de_y", "de_z", "visit_l1", "visit_l2", "recharge"]
    Drone_learned_actions = ["L_charge"]

    # Table of fix parameters
    Drone_fix_parameters_names = ["battery_level_full", "min_x", "min_y", "min_z", "max_x", "max_y", "max_z"]
    Drone_fix_parameters_values = ["100.0", "0.0", "0.0", "0.0", "40.0", "40.0", "40.0"]
    Drone_fix_paraneters = df_fix_parameters(Drone_fix_parameters_names, Drone_fix_parameters_values)

    # Table of variable parameters
    Drone_var_parameters_names = ["energy"]
    Drone_var_parameters_top = [50.0]
    Drone_var_parameters_down = [0.0]
    Drone_var_parameters = df_variable_parameters(Drone_var_parameters_names, Drone_var_parameters_top, Drone_var_parameters_down)

    # Table of symbolic preconditions
    drone_symbolic_preconditions_data = [
        pd.Series([">=0.0", ">=0.0", "<=max_x-10.0", ">=0.0", ">=0.0", ">=10.0"], index=Drone_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", "<=max_y-10.0", ">=0.0", ">=10.0"], index=Drone_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", "<=max_z-10.0", ">=10.0"], index=Drone_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=min_x+10.0", ">=0.0", ">=0.0", ">=10.0"], index=Drone_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", ">=min_y+10.0", ">=0.0", ">=10.0"], index=Drone_variables_names),
        pd.Series([">=0.0", ">=0.0", ">=0.0", ">=0.0", ">=min_z+10.0", ">=10.0"], index=Drone_variables_names),
        pd.Series([">=0.0", ">=0.0", "0.0", "0.0", "0.0", ">=10.0"], index=Drone_variables_names),
        pd.Series([">=0.0", ">=0.0", "10.0", "0.0", "0.0", ">=10.0"], index=Drone_variables_names),
        # pd.Series([">=0.0", ">=0.0", "20.0", "20.0", "20.0", ">=10.0"], index=Drone_variables_names),
        # pd.Series([">=0.0", ">=0.0", "0.0", "0.0", "0.0", ">battery_level_full"], index=Drone_variables_names)
        pd.Series([">=0.0", ">=0.0", "0.0", "0.0", "0.0", "<battery_level_full"], index=Drone_variables_names)
    ]
    drone_symbolic_precons = df_subsymbolic_preconditions(Drone_symbolic_actions, Drone_variables_names, drone_symbolic_preconditions_data)

    # Table of symbolic effects
    drone_symbolic_effects_data = [
        pd.Series(["0.0", "0.0", "10.0", "0.0", "0.0", "-10.0"], index=Drone_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "10.0", "0.0", "-10.0"], index=Drone_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "10.0", "-10.0"], index=Drone_variables_names),
        pd.Series(["0.0", "0.0", "-10.0", "0.0", "0.0", "-10.0"], index=Drone_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "-10.0", "0.0", "-10.0"], index=Drone_variables_names),
        pd.Series(["0.0", "0.0", "0.0", "0.0", "-10.0", "-10.0"], index=Drone_variables_names),
        pd.Series(["10.0", "0.0", "0.0", "0.0", "0.0", "-10.0"], index=Drone_variables_names),
        pd.Series(["0.0", "10.0", "0.0", "0.0", "0.0", "-10.0"], index=Drone_variables_names),
        #pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "(battery_level_full-battery_level)"], index=Drone_variables_names)
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "energy"], index=Drone_variables_names)
    ]
    drone_symbolic_effects = df_symbolic_effects(Drone_variables_names, Drone_symbolic_actions, drone_symbolic_effects_data)

    # Table of subsymbolic preconditions
    drone_subsymbolic_preconditions_data = [
        pd.Series([">=0.0", ">=0.0", "0.0", "0.0", "0.0", "<battery_level_full"], index=Drone_variables_names),
    ]
    drone_subsymbolic_precons = df_subsymbolic_preconditions(Drone_learned_actions, Drone_variables_names, drone_subsymbolic_preconditions_data)

    # Table of subsymbolic effects
    drone_subsymbolic_effects_data = [
        pd.Series(["model(state_in)"], index=Drone_learned_actions),
    ]
    #drone_subsymbolic_effects = df_subsymbolic_models(Drone_learned_actions, drone_subsymbolic_effects_data)

    # Table of initial- & goalstate
    Drone_init_1 = ["0.0", "0.0", "0.0", "0.0", "0.0", "90"]
    Drone_goal_1 = ["10.0", "10.0", "0.0", "0.0", "0.0", "battery_level_full"]
    Drone_1 = df_problem(Drone_variables_names, Drone_init_1, Drone_goal_1)

    drone_variable_parameters_data = [
        pd.Series(["10.0"], index=Drone_var_parameters_names)
    ]  

    recharge_listener_labels_data = [
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "10.0"], index=Drone_variables_names)]
    recharge_listener_under_bounds_data = [
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "0.0"], index=Drone_variables_names)]
    recharge_listener_upper_bounds_data = [
        pd.Series(["0.0", "0.0", "0.0", "0.0", "0.0", "130.0"], index=Drone_variables_names)]

    drone_subsymbolic_effects_parameters = variable_parameters(Drone_learned_actions, Drone_var_parameters_names, drone_variable_parameters_data)
    drone_subsymbolic_labels = df_subsymbolic_label(Drone_variables_names, Drone_learned_actions, recharge_listener_labels_data)
    drone_subsymbolic_under_bounds = df_subsymbolic_under_bound(Drone_variables_names, Drone_learned_actions, recharge_listener_under_bounds_data)
    drone_subsymbolic_upper_bounds = df_subsymbolic_upper_bound(Drone_variables_names, Drone_learned_actions, recharge_listener_upper_bounds_data)

    drone_subsymbolic_effects = subsymbolic_effetcs(drone_subsymbolic_labels, drone_subsymbolic_under_bounds, drone_subsymbolic_upper_bounds, drone_subsymbolic_effects_parameters)

    drone_domain = domain(Drone_variables_names, Drone_symbolic_actions, Drone_learned_actions, Drone_fix_paraneters, Drone_var_parameters, drone_symbolic_precons, drone_symbolic_effects, drone_subsymbolic_precons,drone_subsymbolic_effects)
    drone_domains = input(drone_domain, Drone_1)

    set_1 = data_set(n_samples, Drone_variables_names, Drone_fix_parameters_names, Drone_var_parameters_names)
    set_1_input, set2_output = set_1.generate_charge()

    return set_1_input, set2_output

# Quicktest
if __name__ == "__main__":
    set_1_input, set2_output = create_drone_dataset(500)
    print("yippee")
