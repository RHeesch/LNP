import numpy as np
import random
import json

def drone_n_a(input): 
    output = np.zeros(6)
    for i in range(len(output)):
        output[i] = input[i]
    
    output[5] = input[5] + input[-1]

    return output


def cash_n_a(input): 
    output = np.zeros(14)
    for i in range(len(output)):
        output[i] = input[i]
    
    output[11] = input[11] + input[-1]
    output[12] = input[12] - input[-1]

    return output


def zeno_n_a(input): 
    output = np.zeros(10)
    for i in range(len(output)):
        output[i] = input[i]
    
    output[7] = input[7] + input[-1]

    return output


def flipsi_n_a(input): 
    output = np.zeros(5)
    for i in range(len(output)):
        output[i] = input[i]
    
    x = ((1.5 * 8730 * 377*1000*100)**(-1))
    output[1] = input[1] + (input[-1]*1000000)*(x)

    return output

def no_access_recon_loss(a, b):
    return np.mean(np.abs(a - b))

# Quickrun
if __name__ == "__main__":

    input_data = [0.0, 5.0, 0.0, 0.0, 0.0, 1.5, 8730.0, 377.0, 0] # drone
    expected_output = [0.0, 20.0, 0.0, 0.0, 0.0] # drone

    with open('exp/exp_setup/hparams.json') as f:
        hparam = json.load(f)
    
    out_hat = flipsi_n_a(input_data)
    print("Output from drone_n_a:", out_hat)

    length_recon_in = len(input_data)

    split_index = length_recon_in - hparam["N_REC_VARS"]

    # Split the input array
    x = input_data[:split_index]
    print(x)
    param = np.array(input_data[split_index:])
    print(param)

    # Beam search parameters
    beam_width = hparam["BEAM_WIDTH"]
    perturbation_std = hparam["PERTURBATION_STD"]
    loss_threshold = hparam["opti_threshold"]
    max_steps = hparam["REC_STEPS"]
    
    # Define the min and max values for the parameter interval
    if hparam["DS_DOMAIN"] == "drone":
        param_min = 0
        param_max = 100
    elif hparam["DS_DOMAIN"] == "flipsi":
        param_min = 7405220
        param_max = 7405224
        # 14810445/2
    elif hparam["DS_DOMAIN"] == "zeno":
        param_min = 0
        param_max = 200
    elif hparam["DS_DOMAIN"] == "cashpoint":
        param_min = 5
        param_max = 20

    inital_value_param = (param_max - param_min)/2

    # Initialize the beam with random parameter sets within the specified interval
    beam = [np.clip(inital_value_param + np.random.randn(*param.shape) * perturbation_std, param_min, param_max) for _ in range(beam_width)]
    print(beam)
    
    best_param = None
    best_loss = float('inf')
    steps = 0
    
    while best_loss > loss_threshold and steps < max_steps:
        candidates = []
        
        for candidate_param in beam:
            # Perturb the candidate parameters to generate new candidates
            perturbations = [np.clip(candidate_param + np.random.randn(*candidate_param.shape) * perturbation_std, param_min, param_max) for _ in range(beam_width)]
            candidates.extend(perturbations)
        
        # Evaluate each candidate
        losses = []
        for candidate_param in candidates:
            y_hat = flipsi_n_a(np.concatenate((x, candidate_param), axis=0))
            loss = no_access_recon_loss(expected_output, y_hat)
            losses.append(loss.item())
        
        # Select the top-k candidates based on loss
        top_k_indices = np.argsort(losses)[:beam_width]
        beam = [candidates[i] for i in top_k_indices]
        
        # Update the best parameter if a better loss is found
        for idx in top_k_indices:
            if losses[idx] < best_loss:
                best_loss = losses[idx]
                best_param = candidates[idx]

        steps += 1

    rec_param = best_param

    if best_loss > loss_threshold:
        print('Maximum number of steps reached.')

    print("Best parameters:", best_param)
    print("Reconstruction loss:", best_loss)
