import os
import math
import json
import torch
import numpy as np

class AutoStop:
    def __init__(self, tolerance_e=40, min_delta=0.0001, max_delta=0.001):
        self.tolerance = tolerance_e
        self.max_delta = max_delta
        self.min_delta = min_delta
        self.counter_max = 0
        self.counter_min = 0
        self.min_loss = np.inf

    def auto_stop(self, loss):
        if loss < self.min_loss:
            if (loss - self.min_loss) < self.min_delta:
                self.counter_min += 1
                if self.counter_min > self.tolerance*2:
                    print("\n\nAutostop: Convergence criterion\n\n")
                    return True
            self.min_loss = max(loss)
            self.counter_min, self.counter_max = 0, 0
        elif loss > (self.min_loss + self.max_delta):
            self.counter_max += 1
            if self.counter_max >= self.tolerance:
                print("\n\nAutostop: Divergence criterion\n\n")
                return True
        elif loss == np.inf:
            return True
        elif math.isnan(loss):
            return True
        return False

def save_metrics(metrics: dict, hparam: dict):
    log_path = f"{hparam['LOG_DIR']}/{hparam['DS_DOMAIN']}"
    file_name = f"{hparam['MODEL']}_{hparam['ID']}_metrics.json"
    full_path = os.path.join(log_path, file_name)

    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    hparam["RESULTS"] = metrics
    with open(full_path, "w") as json_file:
        json.dump(hparam, json_file, indent=4)

    return print(f"Metrics saved in {full_path}.")

def listener(hparam: dict): # TODO change name into stasi-function
    """
    The listener function records soft boundaries of training parameters during the training process.
    The boundaries are subsequently passed as heuristics to the SMT planner to allow for a more efficient solving of
    the planning problem.
    """
    # TODO implement listener function
    return