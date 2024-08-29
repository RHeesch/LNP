import numpy as np
import pandas as pd
import math as math
from tqdm import tqdm, trange
import z3
import copy
import numpy as np
import json
from datetime import datetime
from train_module import *
from data_gens.drone import drone_domains as drone_input
from data_gens.flipsi import FliPsi_domains as flipsi_input
from data_gens.zeno import Zeno_domains as zeno_input
from data_gens.cashpoint import cashpoint_domains as cashpoint_input
from data_gens.drone_scale import Drone_scale_domains as drone_scale_input

# Load hyperparameters from JSON configuration file
with open('exp/exp_setup/hparams.json') as f:
    hparam = json.load(f)

# Set up various parameters from the hyperparameter file
max_step = hparam["Max_Step"]
inputdomain = '%s_input' % (hparam["DS_DOMAIN"])
input_object = eval(inputdomain)
threshold = hparam["opti_threshold"]

# integrate_learned_model = input_file.integrate_ml
if hparam["Int_ML"] == 1:
    integrate_learned_model = True
else: 
    integrate_learned_model = False

# Extract various column and row labels for symbolic and subsymbolic data
sym_precons_colums = list(input_object.domain.symbolic_preconditions.df.columns.values)
sym_precons_rows = list(input_object.domain.symbolic_preconditions.df.index.values)
sym_effect_colums = list(input_object.domain.symbolic_effects.df.columns.values)
sym_effect_rows = list(input_object.domain.symbolic_effects.df.index.values)
start_end_colums = list(input_object.problem.df.columns.values)
start_end_rows = list(input_object.problem.df.index.values)
L_precons_colums = list(input_object.domain.subsymbolic_preconditions.df.columns.values)
L_precons_rows = list(input_object.domain.subsymbolic_preconditions.df.index.values)
L_labels_colums = list(input_object.domain.subsymbolic_effetcs.labels.df.columns.values)
L_labels_rows = list(input_object.domain.subsymbolic_effetcs.labels.df.index.values)
L_under_bounds_colums = list(input_object.domain.subsymbolic_effetcs.under_bounds.df.columns.values)
L_under_bounds_rows = list(input_object.domain.subsymbolic_effetcs.under_bounds.df.index.values)
L_upper_bounds_colums = list(input_object.domain.subsymbolic_effetcs.upper_bounds.df.columns.values)
L_upper_bounds_rows = list(input_object.domain.subsymbolic_effetcs.upper_bounds.df.index.values)

# Determine which effects to consider based on learned model integration
if integrate_learned_model == True:
    all_effects = sym_effect_colums + L_labels_colums
else:
    all_effects = sym_effect_colums

# Validate input structure for initial and goal states
if len(start_end_colums) != 2:
    print('The Defintion of inital and goal state is not correct.')
    exit()

# Validate matching structure between preconditions and effects tables
for i in range(len(sym_precons_colums)):
    for l in range(len(sym_precons_colums[i])):
        if sym_precons_colums[i][l] != sym_effect_colums[i][l]:
            print('Tables of precons and effects do not match.')
            exit()
for i in range(len(sym_precons_rows)):
    for l in range(len(sym_precons_rows[i])):
        if sym_precons_rows[i][l] != sym_effect_rows[i][l]:
            print('Tables of precons and effects do not match.')
            exit()

# Build lists of fix parameters
para_fix_colums = list(input_object.domain.fix_parameters.df.columns.values)
para_fix_rows = list(input_object.domain.fix_parameters.df.index.values)

# Build lists of variable parameters
para_var_colums = list(input_object.domain.variable_parameters.df.columns.values)
para_var_rows = list(input_object.domain.variable_parameters.df.index.values)

# Build EnumSort of Variables
list_of_var = []
variables, list_of_var = z3.EnumSort('variables', sym_effect_rows)

# Build lists for vaules of fix parameters
list_of_fix_para = []
for i in range(len(para_fix_rows)):
    list_of_fix_para.append(para_fix_rows[i])
    list_of_fix_para[i] = z3.Real(para_fix_rows[i])

# Build lists for vaules of variable parameters
list_of_var_para = []
for i in range(len(para_var_rows)):
    list_of_var_para.append(para_var_rows[i])
    list_of_var_para[i] = []
    #list_of_var_para[i] = ['list_of_var_%s_%d' % (para_var_rows[i], index)]

if integrate_learned_model == True:
    # Build lists of paramters for learned actions
    learned_effect_parameters = []
    for i in range(len(L_labels_colums)):
        for j in range(len(L_labels_rows)):
            if input_object.domain.subsymbolic_effetcs.labels.df.iloc[j, i] != "0.0":
                name = '%s_%s' % (L_labels_rows[j], L_labels_colums[i])
                learned_effect_parameters.append(name)
    # print(learned_effect_parameters)

    # Build lists for values of parameters for learned actions
    list_of_learned_action_parameters = []
    for i in range(len(learned_effect_parameters)):
        list_of_learned_action_parameters.append(learned_effect_parameters[i])
        list_of_learned_action_parameters[i] = []

# Build lists of action instances
List_infra_actions = copy.deepcopy(list(all_effects))
for i in List_infra_actions:
    i = ['List_' + i + '_0']

# Build lists of instances of the actions
List_number_actions = copy.deepcopy(list(all_effects))
for i in List_number_actions:
    i = ['List_number_' + i + '_0']

# Build lists of instances of effects
List_infra_actions_effects = []
for i in range(len(all_effects)):
    List_infra_actions_effects.append('List_%s_eff' % all_effects[i])
    List_infra_actions_effects[i] = ['List_%s_eff_%d' % (all_effects[i], 0)]

# Build lists of instances of precondintions
List_infra_actions_pre = []
for i in range(len(all_effects)):
    List_infra_actions_pre.append('List_%s_pre' % all_effects[i])
    List_infra_actions_pre[i] = ['List_%s_pre_%d' % (all_effects[i], 0)]

# Initialize lists for instances and states
List_instances = ['instances_0']
List_instance_eff = ['List_instance_eff_0']
List_instance_pre = ['List_instance_pre_0']
List_states = ['List_States_0']
List_actions = ['Action_0']
List_of_ml_inputs = ['ML_in_0']
List_of_ml_outputs = ['ML_out_0']

# Initialize arrays for initial and goal state
Init = z3.Array('Init', variables, z3.RealSort())
Goal = z3.Array('Goal', variables, z3.RealSort())

# Function to add initial and goal state constraints to the solver
def add_init_and_goal(solver):
    for r in range(len(start_end_rows)):
        # Define Initial State
        if input_object.problem.df.iloc[r, 0] != "":
            inits = input_object.problem.df.iloc[r, 0]
            for i in range(len(para_fix_rows)):
                if para_fix_rows[i] == inits:
                    x = '(list_of_fix_para[' + str(i) + '])'
                    inits = inits.replace(para_fix_rows[i], x)
            solver.add(Init[list_of_var[r]] == eval(inits))
        # Define Goal State
        goal_long = input_object.problem.df.iloc[r, 1]
        for i in range(len(para_fix_rows)):
            if para_fix_rows[i] in goal_long:
                x = '(list_of_fix_para[' + str(i) + '])'
                goal_long = goal_long.replace(para_fix_rows[i], x)
        goal_sep = goal_long.split('&')
        for goal in goal_sep:
            if goal[0] == '<':
                if goal[1] == '=':
                    ts = goal[2:]
                    solver.add(Goal[list_of_var[r]] <= eval((ts)))
                else:
                    ts = goal[1:]
                    solver.add(Goal[list_of_var[r]] < eval((ts)))
            elif goal[0] == '>':
                if goal[1] == '=':
                    ts = goal[2:]
                    solver.add(Goal[list_of_var[r]] >= eval((ts)))
                else:
                    ts = goal[1:]
                    solver.add(Goal[list_of_var[r]] > eval((ts)))
            elif goal[0] == '=':
                if goal[1] == '=':
                    ts = goal[2:]
                    solver.add(Goal[list_of_var[r]] == eval((ts)))
                else:
                    ts = goal[1:]
                    solver.add(Goal[list_of_var[r]] == eval((ts)))
            else:
                solver.add(Goal[list_of_var[r]] == eval((goal)))

# Function to add parameter constraints to the solver
def add_parameter_constraints(solver):
    for k in range(len(para_fix_colums)):
        if para_fix_colums[k] == "Top_fix":
            for p in range(len(para_fix_rows)):
                solver.add(list_of_fix_para[p] <= float(input_object.domain.fix_parameters.df.iloc[p, k]))
        elif para_fix_colums[k] == "Down_fix":
            for p in range(len(para_fix_rows)):
                solver.add(list_of_fix_para[p] >= float(input_object.domain.fix_parameters.df.iloc[p, k]))
        else:
            for p in range(len(para_fix_rows)):
                solver.add(list_of_fix_para[p] == float(input_object.domain.fix_parameters.df.iloc[p, k]))

# Function to create a new instance with specific constraints
def new_instance(index, state, solver):
    # extend lists of variable parameters
    for p in range(len(list_of_var_para)):
        list_of_var_para[p].append('list_of_var_%s_%d' % (para_var_rows[p], index))
        list_of_var_para[p][index] = z3.Real('list_of_var_%s_%d' % (para_var_rows[p], index))

    # Instantiate all parameters including their constraints
    for k in range(len(para_var_colums)):
        if para_var_colums[k] == "Top_var":
            for p in range(len(para_var_rows)):
                solver.add(list_of_var_para[p][index] <= float(input_object.domain.variable_parameters.df.iloc[p, k]))
        elif para_var_colums[k] == "Down_var":
            for p in range(len(para_var_rows)):
                solver.add(list_of_var_para[p][index] >= float(input_object.domain.variable_parameters.df.iloc[p, k]))

    # Create a new instance of actions
    Number = z3.Datatype('current_instance_%d' % index)
    for i in range(len(List_number_actions)):
        List_number_actions[i] = Number.declare(all_effects[i])
    Number = Number.create()

    # Define arrays for effects and preconditions
    for i in range(len(all_effects)):
        List_infra_actions_effects[i][index] = z3.Array('List_%s_eff_%d' % (all_effects[i], index), variables,
                                                        z3.RealSort())
        List_infra_actions_pre[i][index] = z3.Array('List_%s_pre_%d' % (all_effects[i], index), variables,
                                                    z3.RealSort())

    # Define the preconditions of symbolic actions
    for c in range(len(sym_effect_colums)):
        for r in range(len(sym_effect_rows)):
            if input_object.domain.symbolic_preconditions.df.iloc[r, c] == "":
                break
            pre = input_object.domain.symbolic_preconditions.df.iloc[r, c]
            for i in range(len(para_fix_rows)):
                if para_fix_rows[i] in pre:
                    x = '(list_of_fix_para[' + str(i) + '])'
                    pre = pre.replace(para_fix_rows[i], x)
            for i in range(len(sym_effect_rows)):
                if sym_effect_rows[i] in pre:
                    x = '(state[list_of_var[' + str(i) + ']])'
                    pre = pre.replace(sym_effect_rows[i], x)
            sep_pre = pre.split('&')
            for s in sep_pre:
                if s[0] == '<':
                    if s[1] == '=':
                        ts = s[2:]
                        solver.add(List_infra_actions_pre[c][index][list_of_var[r]] <= eval((ts)))
                    else:
                        ts = s[1:]
                        solver.add(List_infra_actions_pre[c][index][list_of_var[r]] < eval((ts)))
                elif s[0] == '>':
                    if s[1] == '=':
                        ts = s[2:]
                        solver.add(List_infra_actions_pre[c][index][list_of_var[r]] >= eval((ts)))
                    else:
                        ts = s[1:]
                        solver.add(List_infra_actions_pre[c][index][list_of_var[r]] > eval((ts)))
                elif s[0] == '=':
                    print('Precon at Location %d and %d could not be considered' % (r, c))
                else:
                    solver.add(List_infra_actions_pre[c][index][list_of_var[r]] == eval((s)))

    # Define the effects of symbolic actions
    for c in range(len(sym_effect_colums)):
        for r in range(len(sym_effect_rows)):
            if input_object.domain.symbolic_effects.df.iloc[r, c] == "0.0":
                solver.add(List_infra_actions_effects[c][index][list_of_var[r]] == 0)
            elif input_object.domain.symbolic_effects.df.iloc[r, c] == "":
                solver.add(List_infra_actions_effects[c][index][list_of_var[r]] == 0)
            else:
                sep_eff = input_object.domain.symbolic_effects.df.iloc[r, c]
                for i in range(len(para_var_rows)):
                    y = '(list_of_var_para[' + str(i) + '][index])'
                    sep_eff = sep_eff.replace(para_var_rows[i], y)
                for i in range(len(para_fix_rows)):
                    x = '(list_of_fix_para[' + str(i) + '])'
                    sep_eff = sep_eff.replace(para_fix_rows[i], x)
                for i in range(len(sym_effect_rows)):
                    if sym_effect_rows[i] in sep_eff:
                        x = '(state[list_of_var[' + str(i) + ']])'
                        sep_eff = sep_eff.replace(sym_effect_rows[i], x)
                solver.add(List_infra_actions_effects[c][index][list_of_var[r]] == eval(sep_eff))

    # Define preconditions and effects of learned actions if learned model integration is enabled
    if integrate_learned_model == True:
        # Definition of the preconditions of learned actions
        for lc in range(len(L_precons_colums)):
            for lr in range(len(L_precons_rows)):
                if input_object.domain.subsymbolic_preconditions.df.iloc[lr, lc] == "":
                    break
                pre = input_object.domain.subsymbolic_preconditions.df.iloc[lr, lc]
                for i in range(len(para_fix_rows)):
                    if para_fix_rows[i] in pre:
                        x = '(list_of_fix_para[' + str(i) + '])'
                        pre = pre.replace(para_fix_rows[i], x)
                sep_pre = pre.split('&')
                for s in sep_pre:
                    if s[0] == '<':
                        if s[1] == '=':
                            ts = s[2:]
                            solver.add(
                                List_infra_actions_pre[(len(sym_effect_colums) + lc)][index][list_of_var[lr]] <= eval(ts))
                        else:
                            ts = s[1:]
                            solver.add(
                                List_infra_actions_pre[(len(sym_effect_colums) + lc)][index][list_of_var[lr]] < eval(ts))
                    elif s[0] == '>':
                        if s[1] == '=':
                            ts = s[2:]
                            solver.add(
                                List_infra_actions_pre[(len(sym_effect_colums) + lc)][index][list_of_var[lr]] >= eval(ts))
                        else:
                            ts = s[1:]
                            solver.add(
                                List_infra_actions_pre[(len(sym_effect_colums) + lc)][index][list_of_var[lr]] > eval(ts))
                    elif s[0] == '=':
                        print('Precon at Location %d and %d could not be considered' % (
                        len(sym_effect_rows) + lr, len(sym_effect_colums) + lc))
                    else:
                        solver.add(List_infra_actions_pre[(len(sym_effect_colums) + lc)][index][list_of_var[lr]] == eval(s))

        # Define learned effects for learned actions
        learned_effect_parameters_counter = 0
        for l_a in range(len(L_labels_colums)):
            for l_v in range(len(L_labels_rows)):
                if input_object.domain.subsymbolic_effetcs.labels.df.iloc[l_v, l_a] == "0.0":
                    solver.add(List_infra_actions_effects[(len(sym_effect_colums) + l_a)][index][list_of_var[l_v]] == 0)
                else:
                    list_of_learned_action_parameters[learned_effect_parameters_counter].append('list_of_learned_action_parameters_%s_%d' % (learned_effect_parameters[learned_effect_parameters_counter], index))
                    list_of_learned_action_parameters[learned_effect_parameters_counter][index] = z3.Real('list_of_learned_action_parameters_%s_%d' % (learned_effect_parameters[learned_effect_parameters_counter], index))
                    if hparam["Listener"] == "full":
                        solver.add(list_of_learned_action_parameters[learned_effect_parameters_counter][index] <= float(input_object.domain.subsymbolic_effetcs.upper_bounds.df.iloc[l_v, l_a]))
                        solver.add(list_of_learned_action_parameters[learned_effect_parameters_counter][index] >= float(input_object.domain.subsymbolic_effetcs.under_bounds.df.iloc[l_v, l_a]))
                    effect = '(list_of_learned_action_parameters[' + str(learned_effect_parameters_counter) + '][index])'
                    solver.add(List_infra_actions_effects[(len(sym_effect_colums) + l_a)][index][list_of_var[l_v]] == eval(effect))
                    learned_effect_parameters_counter += 1

    # Collect the effects and preconditions of instances
    Instance_Effect = z3.Array('Instance_Eff', Number, z3.ArraySort(variables, z3.RealSort()))
    Instance_Precon = z3.Array('Instance_Pre', Number, z3.ArraySort(variables, z3.RealSort()))

    for c in range(len(all_effects)):
        Number_effekt = 'Number.' + all_effects[c]
        solver.add(Instance_Effect[eval(Number_effekt)] == List_infra_actions_effects[c][index])
        solver.add(Instance_Precon[eval(Number_effekt)] == List_infra_actions_pre[c][index])
    return Number

# Extend the various lists with new instances
def extend_lists(index):
    List_instances.append("%d" % index)
    List_instance_eff.append('List_instance_eff_%d' % index)
    List_instance_pre.append('List_instance_pre_%d' % index)
    List_states.append('List_States_%d' % index)
    List_actions.append('Action_%d' % index)
    if integrate_learned_model == True:
        List_of_ml_inputs.append('ML_in_%d' % index)
        List_of_ml_outputs.append('ML_out_%d' % index)
    for i in range(len(all_effects)):
        List_infra_actions_effects[i].append('List_%s_eff_%d' % (all_effects[i], index))
        List_infra_actions_pre[i].append('List_%s_pre_%d' % (all_effects[i], index))

# Function to print learned parameters from the model
def print_all_learned_parameter(model, plan):
    # this function extracts the parameters describing the effects of the learned actions
    # all information are stored within a dataframe
    learned_actions = []
    for i in range(len(List_instances)):
        if str(model[plan[i]]) in L_labels_colums:
            x = str(model[plan[i]])
            learned_actions.append(f'{x}&{i}')
    out_learned_index = []
    for i in range(len(L_labels_rows)):
        out_learned_index.append(str(L_labels_rows[i]))
    out_learned = pd.DataFrame(index=out_learned_index, columns=learned_actions)
    for l_a in range(len(learned_actions)):
        sep_i = learned_actions[l_a].split('&')
        index_counter = int(sep_i[-1])
        counter = 0
        for l_act in range(len(L_labels_colums)):
            if L_labels_colums[l_act] == sep_i[0]:
                for l_r in range(len(L_labels_rows)):
                    if input_object.domain.subsymbolic_effetcs.labels.df.iloc[l_r, l_act] == "0.0":
                        out_learned.at[out_learned_index[l_r],learned_actions[l_a]] = 0
                    else:
                        for learned_par in range(len(list_of_learned_action_parameters[l_act])):
                            if sep_i[-1] == str(learned_par):
                                xy = model[list_of_learned_action_parameters[counter][index_counter]]
                                out_learned.at[out_learned_index[l_r],learned_actions[l_a]] = str(xy)
                                counter += 1
                                break
    return out_learned
    
# Function to print the schedule from the model    
def print_schedule(model, plan):
    # this function extracts the plan as well as the variable parameters from the satisfiability model
    # all information are stored within a dataframe as strings
    out_schedule = pd.DataFrame(index=range(len(plan)), columns=range(len(para_var_rows) + 1))
    NNs = []
    for i in range(len(plan)):
        if str(model[plan[i]]) in L_labels_colums:
            x = str(model[plan[i]])
            out_schedule[0][i] = str(f'{x}&{i}')
            NNs.append(x)
        else:
            out_schedule[0][i] = str(model[plan[i]])
    for i in range(len(List_instances)):
        counter_val = 1
        if str(model[plan[i]]) in sym_effect_colums:
            for p in list_of_var_para:
                var_par = False
                for v_p in range(len(para_var_rows)):
                    for r in range(len(sym_effect_rows)):
                        if str(para_var_rows[v_p]) in input_object.domain.symbolic_effects.df.iloc[r, input_object.domain.symbolic_effects.df.columns.get_loc(str(model[plan[i]]))]:
                            var_par = True
                if var_par == True:
                    out_schedule[counter_val][i] = str(model[p[i]])
                else:
                    out_schedule[counter_val][i] = 0
                counter_val += 1
        else:
            for p in list_of_var_para:
                out_schedule[counter_val][i] = 0
                counter_val += 1
    dc_para_var_rows = copy.deepcopy(para_var_rows)
    out_schedule_columns = dc_para_var_rows
    out_schedule_columns.insert(0, 'Action')
    out_schedule_index = []
    for i in plan:
        out_schedule_index.append(str(i))
    out_schedule.index = out_schedule_index
    out_schedule.columns = out_schedule_columns
    return out_schedule, NNs

# Function to print constant parameters from the model
def print_const_parameter(model):
    # this function extracts the fix parameters from the satisifiability model
    # all information are stored within a data frame, that contains string. 
    # this is needed as the strings are the most suitable intermediate type between the Z3 specefic satisifibility model and the types that are used later
    out_fix_para = pd.DataFrame(index=range(len(para_fix_rows)), columns=range(1))
    counter_val = 0
    for p in list_of_fix_para:
        out_fix_para[0][counter_val] = str(model[p])
        counter_val += 1
    out_fix_para_index = []
    for i in range(len(para_fix_rows)):
        out_fix_para_index.append(str(para_fix_rows[i]))
    out_fix_para.index = out_fix_para_index
    out_fix_para.columns = ['values']
    return out_fix_para

# Function to prepare data for training ML models based on the output of the planning algorithm
def to_ML_model(schedule, fix_paras, learned_parameters):
    # this function builds the input and output vectors, that are used by the ML-model later. 
    # all information are extracted from the satisifbility model and converted into non Z3 specefic types
    # the input vector is the concatination of the feature vector describing the state right before the learned action and all fix parameters
    # the output vector is the feature vector describing the state right after the learned action
    inputs = []
    outputs = []
    counter = 0
    counter_ml = 0
    for l_a in learned_parameters.columns:
        inputs.append(l_a)
        inputs[counter] = []
        outputs.append(l_a)
        outputs[counter] = []
        if counter == 0: 
            for r_out in range(len(L_labels_rows)):
            # each input-list is initialized as initial state
            # then, the effect of each action is featurewise added to the state
            # this is necessary, as the explicit information about the intermideate states could not be extraced from the satisfibility model
                inputs[counter].append(eval(input_object.problem.df.iloc[r_out, 0]))
                outputs[counter].append(0)
            for step in range(len(List_instances)):
                if schedule.iloc[step, 0] not in l_a:
                    for seff in range(len(sym_effect_colums)):
                        if sym_effect_colums[seff] == schedule.iloc[step, 0]:
                            for r in range(len(L_labels_rows)):
                                sep_eff = input_object.domain.symbolic_effects.df.iloc[r, seff]
                                for pvar in range(1,len(schedule.columns)):
                                    y = str(schedule.iloc[step, pvar])
                                    sep_eff = sep_eff.replace(schedule.columns[pvar], y)
                                for pfix in range(len(fix_paras.index)):
                                    x = str(fix_paras.iloc[pfix, 0])
                                    sep_eff = sep_eff.replace(fix_paras.index[pfix], x)
                                inputs[counter][r] = inputs[counter][r] + eval(sep_eff)
                else:
                    for r_op in range(len(L_labels_rows)):
                        if type(learned_parameters.at[L_labels_rows[r_op] ,l_a]) == int:
                            outputs[counter][r_op] = inputs[counter][r_op] + float(learned_parameters.at[L_labels_rows[r_op] ,l_a])
                        else:
                            outputs[counter][r_op] = inputs[counter][r_op] + eval(learned_parameters.at[L_labels_rows[r_op] ,l_a])
                    counter_ml = step
                    break
        else:
            for r_out in range(len(L_labels_rows)): 
                inputs[counter].append(eval(str(outputs[counter-1][r_out])))
                outputs[counter].append(0)
            for step in range(len(List_instances)-counter_ml):
                if schedule.iloc[(step+counter_ml), 0] not in l_a:
                    for seff in range(len(sym_effect_colums)):
                        if sym_effect_colums[seff] == schedule.iloc[step+counter_ml, 0]:
                            for r in range(len(L_labels_rows)):
                                sep_eff = input_object.domain.symbolic_effects.df.iloc[r, seff]
                                for pvar in range(1,len(schedule.columns)):
                                    y = str(schedule.iloc[step+counter_ml, pvar])
                                    sep_eff = sep_eff.replace(schedule.columns[pvar], y)
                                for pfix in range(len(fix_paras.index)):
                                    x = str(fix_paras.iloc[pfix, 0])
                                    sep_eff = sep_eff.replace(fix_paras.index[pfix], x)
                                inputs[counter][r] = inputs[counter][r] + eval(sep_eff)
                else:
                    for r_op in range(len(L_labels_rows)):
                        if type(learned_parameters.at[L_labels_rows[r_op] ,l_a]) == int:
                            outputs[counter][r_op] = inputs[counter][r_op] + float(learned_parameters.at[L_labels_rows[r_op] ,l_a])
                        else:
                            outputs[counter][r_op] = inputs[counter][r_op] + eval(learned_parameters.at[L_labels_rows[r_op] ,l_a])
                        counter_ml = counter_ml + step
                    break
        for pfix in range(len(fix_paras.index)):
            if '/' in fix_paras.iloc[pfix, 0]:
                fraction = '3/2'
                numerator, denominator = map(float, fraction.split('/'))
                result = numerator / denominator
                inputs[counter].append(float(result))
            else:
                inputs[counter].append(float(fix_paras.iloc[pfix, 0]))
        for p_var in para_var_rows:
            inputs[counter].append(0)
        counter += 1
    return inputs, outputs

# Simulate ML model prediction (stub function for testing)
def ml_model_simulation(input_vectors, output_vectors):
    ml_return_parameters = []
    for i in input_vectors:
        if i == [10.0, 10.0, 0.0, 0.0, 0.0, 50.0, 100.0, 0.0, 0.0, 0.0, 40.0, 40.0, 40.0, None]:
            ml_parameter = [1000]
        else: 
            ml_parameter = [40]
        ml_return_parameters.append(ml_parameter)
    return ml_return_parameters

# Function to validate ML model parameters and add constraints to the solver
def check_ml_parameters(learned_parameters, ml_return_parameters, solver, index_ml_return_parameters):
    ml_return_parameters_counter = 0
    l_a_sep = learned_parameters.columns[index_ml_return_parameters].split('&')
    for p_var in range(len(para_var_rows)):
        if input_object.domain.subsymbolic_effetcs.variable_parameters.df.at[para_var_rows[p_var], l_a_sep[0]] == '10.0':
            solver.add(list_of_var_para[p_var][int(l_a_sep[1])] == ml_return_parameters[ml_return_parameters_counter])
            ml_return_parameters_counter += 1

# Function to add a blocking lemma to the solver to prevent repeating the same solution
def new_blocking_lemma(model, schedule, solver, plan):
    blocking_lemma = z3.And(z3.Or([p[i] != model[p[i]] for p in list_of_var_para for i in range(len(schedule.index))]), z3.Or([x != model[x] for x in plan]),
                            z3.Or([lp[j] != model[lp[j]] for lp in list_of_learned_action_parameters for j in range(len(schedule.index))]))
    solver.add(blocking_lemma)
    
# Function to add a blocking lemma based on partial solutions
def blocking_lemma(model, schedule, schedule_learned, solver, index_learned, plan):
    index_lemma = 0
    for i in range(len(schedule.index)): 
        if schedule.index[i] == schedule_learned.index[index_learned]:
            index_lemma = i
    plan_so_far = []
    for i in range(0,index_lemma):
        plan_so_far.append(plan[i])
    blocking_lemma = z3.Or([x != model[x] for x in plan])
    solver.add(blocking_lemma)

# Main planning algorithm
def planning_alg(hparam, index, steps):
    plan = []
    start_time = datetime.now()
    finish = False
    solver = z3.Solver()
    add_init_and_goal(solver)
    add_parameter_constraints(solver)
    var = z3.Const('var', variables)
    List_instances[index] = new_instance(index, Init, solver)
    List_actions[index] = z3.Const('Action_%d' % index, List_instances[index])
    List_instance_pre[index] = z3.Array('Instance_Pre', List_instances[index], z3.ArraySort(variables, z3.RealSort()))
    List_instance_eff[index] = z3.Array('Instance_Eff', List_instances[index], z3.ArraySort(variables, z3.RealSort()))
    List_states[index] = z3.Array('States', List_instances[index], z3.ArraySort(variables, z3.RealSort()))
    plan.append(List_actions[index])
    solver.add(z3.ForAll([var], List_instance_pre[index][List_actions[index]][var] == Init[var]))
    solver.add(z3.ForAll([var], List_states[index][List_actions[index]][var] == Init[var] +
                         List_instance_eff[index][List_actions[index]][var]))
    while index < steps:
        solver.push()
        solver.add(z3.ForAll([var], List_states[index][List_actions[index]][var] == Goal[var]))
        if z3.sat == solver.check():
            if integrate_learned_model == True:
                solution_found = True
                while solution_found == True:
                    status = solver.check()
                    if status != z3.sat:
                        solution_found == False
                        break
                    model = solver.model()
                    model_schedule, NNs = print_schedule(model, plan)
                    print(model_schedule)
                    model_fix_parameters = print_const_parameter(model)
                    model_learned = print_all_learned_parameter(model, plan)
                    if model_learned.empty:
                        valid_parameters = True
                    else: 
                        input_vectors, output_vectors = to_ML_model(model_schedule, model_fix_parameters, model_learned)
                        print(input_vectors)
                        print(output_vectors)
                        ml_return_parameters = []
                        valid_parameters = False
                        for learned_action_in_schedule in range(len(input_vectors)):
                            print(NNs[learned_action_in_schedule])
                            ml_return_parameters.append(determine_parameter(hparam, input_vectors[learned_action_in_schedule], output_vectors[learned_action_in_schedule], NNs[learned_action_in_schedule]))
                            print(ml_return_parameters[learned_action_in_schedule])
                            solver.push()
                            print(ml_return_parameters[learned_action_in_schedule][0])
                            print(learned_action_in_schedule)
                            check_ml_parameters(model_learned, ml_return_parameters[learned_action_in_schedule][0], solver, learned_action_in_schedule)
                            if solver.check() == z3.sat and ml_return_parameters[learned_action_in_schedule][1] <= threshold:
                                print('Parameterset %d is valid.' %(learned_action_in_schedule+1))
                                valid_parameters = True
                            elif solver.check() == z3.sat: 
                                print('Parameterset %d is not valid. Threshold is violated. blocking lemma is added' %(learned_action_in_schedule+1))
                                solver.pop()
                                valid_parameters = False
                                new_blocking_lemma(model, model_schedule, solver, plan)
                                end_time = datetime.now()
                                dif_time = end_time - start_time
                                runtime = dif_time.total_seconds()
                                print(runtime)
                                break
                            else: 
                                print('Parameterset %d is not valid. Parameters are violating their limit. blocking lemma is added' %(learned_action_in_schedule+1))
                                solver.pop()
                                valid_parameters = False
                                new_blocking_lemma(model, model_schedule, solver, plan)
                                end_time = datetime.now()
                                dif_time = end_time - start_time
                                runtime = dif_time.total_seconds()
                                print(runtime)
                                break
                    if valid_parameters == True:
                        solution_found = False
                        finish = True
                        solver.pop()
                        break
            else:
                print('sat')
                model = solver.model()
                end_time = datetime.now()
                dif_time = end_time - start_time
                runtime = dif_time.total_seconds()
                print(runtime)
                schedule = print_schedule(model, plan)
                print(schedule)
                return runtime
        if finish == True: 
            end_time = datetime.now()
            dif_time = end_time - start_time
            runtime = dif_time.total_seconds()
            print(runtime)
            schedule = print_schedule(model, plan)
            print(schedule)
            return runtime
        solver.pop()
        index += 1
        end_time = datetime.now()
        print('There was no solution found in instance %d.' % index)
        extend_lists(index)
        List_instances[index] = new_instance(index, List_states[index-1][List_actions[index-1]], solver)
        List_actions[index] = z3.Const('Action_%d' % index, List_instances[index])
        List_instance_pre[index] = z3.Array('Instance_Pre', List_instances[index], z3.ArraySort(variables, z3.RealSort()))
        List_instance_eff[index] = z3.Array('Instance_Eff', List_instances[index], z3.ArraySort(variables, z3.RealSort()))
        List_states[index] = z3.Array('States', List_instances[index], z3.ArraySort(variables, z3.RealSort()))
        plan.append(List_actions[index])
        solver.add(z3.ForAll([var], List_instance_pre[index][List_actions[index]][var] ==
                             List_states[index - 1][List_actions[index - 1]][var]))
        solver.add(z3.ForAll([var], List_states[index][List_actions[index]][var] ==
                             List_states[index - 1][List_actions[index - 1]][var] +
                             List_instance_eff[index][List_actions[index]][var]))

    
# Quicktest
if __name__ == "__main__":
    model = planning_alg(hparam, 0, max_step)
    schedule = print_schedule(model)
    print(schedule)

