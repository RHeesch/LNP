import gc
import math
import torch
import numpy as np
from tqdm import tqdm
import json
import utils as u
import models as m
from data_module import *
from no_access_funtions import *

class training():
    def __init__(self, hparam):
        self.hparam = hparam
        self.autostop = u.AutoStop()
        self.loss_fn = nn.MSELoss()
        self.datamodule = DataModule(hparam, scaling=False)

        self.listener_change = []
        self.listener_upper = []
        self.listener_lower = []

    def train_nn(self):
        """
        First the model is regularly trained on a complete dataset.
        The reconstruction of missing parameters takes place only in the application function
        """
        # Impot data and get input + output shapes
        dl_train, dl_val, dl_test = self.datamodule.train_loader(), self.datamodule.val_loader(), self.datamodule.test_loader()
        in_, out_ = next(iter(dl_train))[0].shape[1], next(iter(dl_train))[1].shape[1]

        # for training on GPUs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # with torch.cuda.device(self.device) if torch.cuda.device_count() > 1 else torch.cuda.device(0):
        # print(f"run on {self.device} {torch.cuda.get_device_name()}...\n")

        # init model + optimizer
        self.model = m.NeuralAction(input_dim=in_, hidden_dim=self.hparam["N_HIDDEN"], output_dim=out_, n_layers=self.hparam["N_LAYERS"], dropout=self.hparam["DROPOUT"]).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparam["LEARNING_RATE"], weight_decay=self.hparam["WEIGHT_DECAY"])
        print(self.model)

        with tqdm(range(self.hparam["EPOCHS"]), unit="epoch") as tepoch:
            for e in tepoch:
                # train_step
                l_train = self.train_step(dl_train, optimizer)

                # val step
                if e % 50 == 0:
                    l_val = self.val_step(dl_val)

                #if self.autostop.auto_stop(l_val): # TODO fix autostop
                    #    break

                tepoch.set_postfix(train_loss=l_train, val_loss=l_val)

        # test step
        l_test = self.val_step(dl_test)

        # save results
        results = {"train_loss": l_train, "val_loss": l_val, "test_loss": l_test}
        u.save_metrics(metrics=results, hparam=self.hparam)
        torch.save(self.model.state_dict(), self.hparam["LOG_DIR"] + "/" + hparam['DS_DOMAIN'] + "/" + self.hparam['DS_DOMAIN'] + "_L_charge" + ".pth")
        print("Model weights saved in " + self.hparam["LOG_DIR"] + "/" + hparam['DS_DOMAIN'] + "/model.pth .")

        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        return l_test

    def train_step(self, dl_train, optimizer):
        self.model.train()
        for x, y in dl_train:
            self.listener(x, y)

            x, y = x.to(self.device), y.to(self.device)

            y_hat = self.model(x, torch.nan, training=True)
            loss = self.loss_fn(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss.item()

    def val_step(self, dl_val):
        self.model.eval()
        with torch.no_grad():
            for x, y in dl_val:
                self.listener(x, y)

                x, y = x.to(self.device), y.to(self.device)

                y_hat = self.model(x, torch.nan, training=True)
                loss = self.loss_fn(y_hat, y)
        return loss.item()

    def listener(self, in_state, out_state):
        """
        The listener function records during the training of an NN which parameters of a state are changed by the
        effects of an action. It returns three vecotors:
            - "changes": denoting what parameter changed (indicated by 10, else 0),
            - "lower_bound": denoting the lower bound of the changes
            - "upper_bound": denoting the upper bound of the changes
        """
        in_state = in_state[:, :out_state.size(1)]

        delta = in_state != out_state
        change = (delta.sum(dim=0) > 0).int() * 10
        upper = torch.where(delta, torch.max(in_state, out_state), torch.full_like(in_state, 0)).max(dim=0)[0]
        lower = torch.where(delta, torch.min(in_state, out_state), torch.full_like(in_state, 0)).min(dim=0)[0]

        self.listener_change, self.listener_lower, self.listener_upper = change.tolist(), lower.tolist(), upper.tolist()
        return

class reconstruction():
    def __init__(self, hparam, in_, out_, action_):
        self.hparam, self.in_, self.out_, self.action_ = hparam, in_, out_, action_
        self.device = self.device = self.hparam["DEVICE"] # "cuda" if torch.cuda.is_available() else "cpu" # TODO change back
        self.model = self.load_model()
        self.loss = nn.L1Loss()

    def load_model(self):
        model = m.NeuralAction(input_dim=self.in_, hidden_dim=self.hparam["N_HIDDEN"], output_dim=self.out_, n_layers=self.hparam["N_LAYERS"], dropout=self.hparam["DROPOUT"])
        model.load_state_dict(torch.load(self.hparam["LOG_DIR"] + "/" + self.hparam['DS_DOMAIN'] + "/" + self.hparam['DS_DOMAIN'] + "_" + self.action_ + ".pth"))
        model.to(self.device)
        return model

    def reconstruction(self, x_in, y):
        x, param = torch.split(x_in, [self.in_ - self.hparam["N_REC_VARS"], self.hparam["N_REC_VARS"]], dim=1)

        rec_param = torch.zeros_like(param, dtype=torch.float32, requires_grad=True, device=self.device)

        optimizer = torch.optim.SGD([rec_param], lr=self.hparam["REC_LR"])
        # optimizer = torch.optim.Adam([rec_param], lr=self.hparam["REC_LR"])

        for epoch in range(self.hparam["REC_EPOCHS"]):
            optimizer.zero_grad()

            y_hat = self.model(x, rec_param, training=False)
            loss = self.loss(y, y_hat)
            loss.backward()
            optimizer.step()

        rec_loss = self.loss(x_in, torch.cat((x, rec_param), dim=1))
        return {"param": param.cpu().detach().numpy(), "rec_param": rec_param.cpu().detach().numpy(), "loss": loss.cpu().detach().numpy(), "rec_loss": rec_loss.cpu().detach().numpy()}

    def sampling(self, x, y):
        res_dict = {}

        with tqdm(range(self.hparam["N_MODELS"]), unit="sampling") as n_sampling:
            for n in n_sampling:
                torch.manual_seed(n)
                res_dict[n] = self.reconstruction(x, y)
        return res_dict

    def results(self, res_dict):
        min_key = min(res_dict, key=lambda k: res_dict[k]["loss"].item())
        return res_dict[min_key]


def no_access_recon(hparam, input_data, expected_output):

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
        param_min = 0
        param_max = 20000000
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
            y_hat = zeno_n_a(np.concatenate((x, candidate_param), axis=0))
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

    if best_loss > loss_threshold:
        print('Maximum number of steps reached.')

    return best_param, best_loss


def determine_parameter(hparam, input, output, action):

    length_recon_in = len(input)
    length_recon_out = len(output)

    if hparam["access"] == "no":
        final_res, loss = no_access_recon(hparam, input, output)
        n_rec_param = hparam['N_REC_VARS']

        back_to_smt = []
        for i in range(n_rec_param):
            back_to_smt.append(float(final_res))
    else:
        input = [input]
        output = [output]
        if hparam["SCALE"] == 1:
            scaler_x = joblib.load("../exp/exp_setup/scaler_x.pkl")
            scaler_y = joblib.load("../exp/exp_setup/scaler_y.pkl")
            input = torch.Tensor(scaler_x.transform(input)).to("cpu")
            output = torch.Tensor(scaler_y.transform(output)).to("cpu")
        elif hparam["SCALE"] == 0:
            input = torch.Tensor(input).to("cpu")
            output = torch.Tensor(output).to("cpu")

        apl = reconstruction(hparam, in_=length_recon_in, out_=length_recon_out, action_=action)
        res_dict = apl.sampling(input, output)

        final_res = apl.results(res_dict)

        new_param = input
        
        loss = final_res['rec_loss']

        n_rec_param = hparam['N_REC_VARS']
        new_param[0,(len(input[0])-n_rec_param):len(input[0])] = torch.from_numpy(final_res['rec_param'])
        back_to_smt = []

        if hparam["SCALE"] == 1:
            new_x_original = scaler_x.inverse_transform(new_param)
            for i in range(n_rec_param):
                back_to_smt.append(float(new_x_original[0,(len(input[0])-n_rec_param+i)]))
        else:
            for i in range(n_rec_param):
                back_to_smt.append(float(new_param[0,(len(input[0])-n_rec_param+i)]))

    print(back_to_smt)

    return back_to_smt, loss


# Quickrun
if __name__ == "__main__":

    with open('exp/exp_setup/hparams.json') as f:
        hparam = json.load(f)

    # testing the training functionality
    """ print("\nTraining the neural network on the dataset...")
    pl = training(hparam)
    test_loss = pl.train_nn() """

    #print(f"listener change: {pl.listener_change}")
    #print(f"listener upper: {pl.listener_upper}")
    #print(f"listener lower: {pl.listener_lower}")

    # only for testing on models trained on "drone" dataset
    print("\nReconstruct the missing variables from the input tensor...")
    if hparam["DS_DOMAIN"] == "drone_scale":
        x = np.array([[0.0, 10.0, 0.0, 0.0, 0.0, 60.0, 150.0, 0.0, 0.0, 0.0, 40.0, 40.0, 40.0, 30., 0.0, 0.0, 0.0]]) # drone
        y = np.array([[0.0, 10.0, 0.0, 0.0, 0.0, 80]]) # drone
    elif hparam["DS_DOMAIN"] == "drone":
        x = np.array([[10.0, 0.0, 0.0, 0.0, 0.0, 80.0, 200.0, 0.0, 0.0, 0.0, 40.0, 40.0, 40.0, 0]]) # drone
        y = np.array([[10.0, 0.0, 0.0, 0.0, 0.0, 171.25]]) # drone
    elif hparam["DS_DOMAIN"] == "drone_scale":
        x = np.array([[0.0, 10.0, 0.0, 0.0, 0.0, 60.0, 150.0, 0.0, 0.0, 0.0, 40.0, 40.0, 40.0, 30., 10., 0., 0.]]) # drone
        y = np.array([[0.0, 10.0, 0.0, 10.0, 0.0, 50]]) # drone
    elif hparam["DS_DOMAIN"] == "flipsi":
        x = np.array([[0., 5., 0., 0., 0., 1.5, 8730, 377, 0]]) # flipsi
        y = np.array([[0., 20., 0., 0., 0.]]) # flipsi
    elif hparam["DS_DOMAIN"] == "zeno":
        x = np.array([[20.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 7.0, 8.0, 4.0, 1.50, 60.0, 8.0, 50]]) # zeno
        y = np.array([[20.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0]]) # zeno
    elif hparam["DS_DOMAIN"] == "cashpoint":
        x = np.array([[0.0, 0.0, 0.0, 10.0, 0.0, 10.0, 0.0, 0.0, 10.0, 10.0, 10.0, 0.0, 200, 200, 50]]) # cash
        y = np.array([[0.0, 0.0, 0.0, 10.0, 0.0, 10.0, 0.0, 0.0, 10.0, 10.0, 10.0, 150.0, 150, 200]]) # cash
    print(x)
    print(y)

    bts = determine_parameter(hparam,x[0],y[0],"L_charge")

    print(bts)