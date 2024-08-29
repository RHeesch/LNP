import os
import json
import random
import numpy as np
import pandas as pd
import torch
import z3
import torch.nn as nn
from LNP import *

index = 0
n_exp = 20

def train_nn(dataset, hparam):
    """
    This function takes generated data as input and trains an INN on it.
    Additionally, it returns the ranges in which the INN operates as heuristic for the SMT-based planner.
    """
    random.seed(hparam["SEED"])
    np.random.seed(hparam["SEED"])
    torch.manual_seed(hparam["SEED"])

    trained_nn = 0
    ranges = 0
    return trained_nn, ranges

def run_planner():
    runtime = planning_alg(hparam, index, 10) 
    # schedule = print_schedule(model)
    # print(schedule)
    results = open('Results_listener', 'a')
    results.write(str(runtime))
    results.write(' ,')
    results.close()
    return

def run_listener_exp(n_exp):
    results = []
    for i in range(n_exp):
        print('run_%d' %i)
        runtime = planning_alg(hparam, index, 10)
        # schedule = print_schedule(models_out[i], plans[i])
        # print(schedule)
        results.append(runtime)
    return results

if __name__ == "__main__":
    run_planner()
    # results = run_listener_exp(n_exp)
    #print(results)
