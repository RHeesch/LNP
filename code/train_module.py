import gc
import math
import os
import logging
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json
import joblib
import utils as u
import models as m
from data_module import *
from path_utils import get_exp_setup_dir, normalize_log_dir, get_log_dir, to_repo_relative, get_repo_root
from data_gens.json_loader import domain_json_path

logger = logging.getLogger("lazy_planner")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

class training():
    def __init__(self, hparam, data_path=None):
        self.hparam = hparam
        normalize_log_dir(self.hparam)
        self.autostop = u.AutoStop()
        self.loss_fn = nn.MSELoss()
        self.datamodule = DataModule(hparam, scaling=False, data_path=data_path)
        self.predict_delta = self.hparam.get("PREDICT_DELTA", 0) == 1
        self.model_name = self.hparam.get("MODEL", "autoencoder").lower()
        self.listener_eps = self.hparam.get("LISTENER_EPS", 1e-6)
        self.bool_loss = self.hparam.get("BOOL_LOSS", "bce").lower()
        self.bool_encoding = self.hparam.get("BOOL_ENCODING", "01")
        self.bool_weight = float(self.hparam.get("BOOL_LOSS_WEIGHT", 1.0))
        self.bool_indices = []

        self.listener_change = []
        self.listener_upper = []
        self.listener_lower = []

    def train_nn(self, early_stop=None):
        """
        First the model is regularly trained on a complete dataset.
        The reconstruction of missing parameters takes place only in the application function
        """
        # Import data and get input + output shapes
        dl_train, dl_val, dl_test = self.datamodule.train_loader(), self.datamodule.val_loader(), self.datamodule.test_loader()
        in_, out_ = next(iter(dl_train))[0].shape[1], next(iter(dl_train))[1].shape[1]
        self._init_listener_stats(out_)
        self._init_bool_indices(out_)

        # for training on GPUs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # with torch.cuda.device(self.device) if torch.cuda.device_count() > 1 else torch.cuda.device(0):
        # print(f"run on {self.device} {torch.cuda.get_device_name()}...\n")

        # init model + optimizer
        if self.model_name in ["autoencoder", "neuralaction"]:
            self.model = m.NeuralAction(
                input_dim=in_,
                hidden_dim=self.hparam["N_HIDDEN"],
                output_dim=out_,
                n_layers=self.hparam["N_LAYERS"],
                dropout=self.hparam["DROPOUT"],
            ).to(self.device)
        elif self.model_name in ["ffn", "feedforward", "mlp"]:
            self.model = m.FeedForwardNN(
                input_dim=in_,
                hidden_dim=self.hparam["N_HIDDEN"],
                output_dim=out_,
                n_layers=self.hparam["N_LAYERS"],
                dropout=self.hparam["DROPOUT"],
            ).to(self.device)
        else:
            raise ValueError(f"Unknown MODEL: {self.hparam.get('MODEL')}")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparam["LEARNING_RATE"], weight_decay=self.hparam["WEIGHT_DECAY"])
        logger.info("Model:\n%s", self.model)

        best_val = float("inf")
        no_improve = 0
        l_val = float("inf")
        with tqdm(range(self.hparam["EPOCHS"]), unit="epoch", disable=self.hparam.get("QUIET", False)) as tepoch:
            for e in tepoch:
                # train_step
                l_train = self.train_step(dl_train, optimizer)

                # val step
                if e % 50 == 0:
                    l_val = self.val_step(dl_val)
                    if l_val < best_val:
                        best_val = l_val
                        no_improve = 0
                    else:
                        no_improve += 1

                    if early_stop:
                        min_epochs = early_stop.get("min_epochs", 0)
                        patience = early_stop.get("patience", None)
                        diverge_factor = early_stop.get("diverge_factor", None)
                        if math.isnan(l_val) or l_val == float("inf"):
                            break
                        if (
                            diverge_factor
                            and best_val < float("inf")
                            and l_val > best_val * diverge_factor
                            and e >= min_epochs
                        ):
                            break
                        if patience is not None and no_improve >= patience and e >= min_epochs:
                            break

                #if self.autostop.auto_stop(l_val): # TODO fix autostop
                    #    break

                tepoch.set_postfix(train_loss=l_train, val_loss=l_val)

        # test step
        l_test = self.val_step(dl_test)

        # save results
        results = {
            "train_loss": l_train,
            "val_loss": l_val,
            "test_loss": l_test,
            "model": self.model_name,
            "predict_delta": int(self.predict_delta),
        }
        u.save_metrics(metrics=results, hparam=self.hparam)
        self.save_listener_bounds()
        ml_action = self.hparam.get("ML_ACTION", "L_charge")
        log_dir = get_log_dir(self.hparam)
        weights_path = (
            log_dir
            / self.hparam["DS_DOMAIN"]
            / f"{self.hparam['DS_DOMAIN']}_{self.model_name}_{ml_action}.pth"
        )
        torch.save(self.model.state_dict(), weights_path)
        logger.info("Model weights saved in %s.", to_repo_relative(weights_path))

        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        return results

    def _init_bool_indices(self, out_dim):
        try:
            cfg_path = domain_json_path(self.hparam["DS_DOMAIN"])
            with open(cfg_path) as f:
                cfg = json.load(f)
            out_state = cfg.get("dataset", {}).get("output_layout", {}).get("state", cfg["variables"])
            bool_vars = {v for v, t in cfg.get("variable_types", {}).items() if t == "boolean"}
            self.bool_indices = [i for i, v in enumerate(out_state[:out_dim]) if v in bool_vars]
        except Exception:
            self.bool_indices = []

    def _compute_loss(self, y_hat, y):
        if not self.bool_indices:
            return self.loss_fn(y_hat, y)
        idx = torch.tensor(self.bool_indices, device=y_hat.device)
        mask = torch.ones(y_hat.size(1), dtype=torch.bool, device=y_hat.device)
        mask[idx] = False
        y_hat_bool = y_hat[:, idx]
        y_bool = y[:, idx]
        y_hat_cont = y_hat[:, mask]
        y_cont = y[:, mask]
        loss_cont = self.loss_fn(y_hat_cont, y_cont) if y_hat_cont.numel() else 0.0
        if self.bool_loss == "bce":
            # map targets to 0/1 if encoded as 0/10
            if self.bool_encoding == "010":
                y_bool = y_bool / 10.0
                y_hat_bool = y_hat_bool / 10.0
            bce = nn.BCEWithLogitsLoss()
            loss_bool = bce(y_hat_bool, y_bool)
        else:
            # weighted MSE
            loss_bool = self.loss_fn(y_hat_bool, y_bool) * self.bool_weight
        return loss_cont + loss_bool

    def train_step(self, dl_train, optimizer):
        self.model.train()
        for x, y in dl_train:
            self.listener(x, y)

            x, y = x.to(self.device), y.to(self.device)

            y_hat = self.model(x, torch.nan, training=True)
            if self.predict_delta:
                x_state = x[:, : y.size(1)]
                y = y - x_state
            loss = self._compute_loss(y_hat, y)

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
                if self.predict_delta:
                    x_state = x[:, : y.size(1)]
                    y = y - x_state
                loss = self._compute_loss(y_hat, y)
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

        delta = out_state - in_state
        change_mask = delta.abs() > self.listener_eps
        self.listener_change = self.listener_change | change_mask.any(dim=0)

        batch_min = torch.where(change_mask, delta, torch.full_like(delta, float("inf"))).min(dim=0)[0]
        batch_max = torch.where(change_mask, delta, torch.full_like(delta, float("-inf"))).max(dim=0)[0]
        self.listener_lower = torch.minimum(self.listener_lower, batch_min)
        self.listener_upper = torch.maximum(self.listener_upper, batch_max)
        return

    def _init_listener_stats(self, out_dim):
        self.listener_change = torch.zeros(out_dim, dtype=torch.bool)
        self.listener_lower = torch.full((out_dim,), float("inf"))
        self.listener_upper = torch.full((out_dim,), float("-inf"))

    def save_listener_bounds(self):
        log_dir = normalize_log_dir(self.hparam)
        log_path = log_dir / self.hparam["DS_DOMAIN"]
        os.makedirs(log_path, exist_ok=True)
        file_name = f"listener_bounds_{self.model_name}.json"
        full_path = log_path / file_name

        lower = self.listener_lower.clone()
        upper = self.listener_upper.clone()
        lower[torch.isinf(lower)] = 0.0
        upper[torch.isinf(upper)] = 0.0

        payload = {
            "ml_action": self.hparam.get("ML_ACTION", "L_charge"),
            "model": self.model_name,
            "predict_delta": int(self.predict_delta),
            "eps": self.listener_eps,
            "change": self.listener_change.int().tolist(),
            "lower": lower.tolist(),
            "upper": upper.tolist(),
        }
        with open(full_path, "w") as f:
            json.dump(payload, f, indent=2)
        return

class reconstruction():
    def __init__(self, hparam, in_, out_, action_):
        self.hparam, self.in_, self.out_, self.action_ = hparam, in_, out_, action_
        normalize_log_dir(self.hparam)
        self.device = self.hparam.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.loss = nn.L1Loss()
        self.predict_delta = self.hparam.get("PREDICT_DELTA", 0) == 1
        self.model_name = self.hparam.get("MODEL", "autoencoder").lower()

    def load_model(self):
        if self.hparam.get("MODEL", "autoencoder").lower() in ["autoencoder", "neuralaction"]:
            model = m.NeuralAction(
                input_dim=self.in_,
                hidden_dim=self.hparam["N_HIDDEN"],
                output_dim=self.out_,
                n_layers=self.hparam["N_LAYERS"],
                dropout=self.hparam["DROPOUT"],
            )
        elif self.hparam.get("MODEL", "autoencoder").lower() in ["ffn", "feedforward", "mlp"]:
            model = m.FeedForwardNN(
                input_dim=self.in_,
                hidden_dim=self.hparam["N_HIDDEN"],
                output_dim=self.out_,
                n_layers=self.hparam["N_LAYERS"],
                dropout=self.hparam["DROPOUT"],
            )
        else:
            raise ValueError(f"Unknown MODEL: {self.hparam.get('MODEL')}")
        log_dir = get_log_dir(self.hparam)
        weights_path = (
            log_dir
            / self.hparam["DS_DOMAIN"]
            / f"{self.hparam['DS_DOMAIN']}_{self.hparam.get('MODEL', 'autoencoder').lower()}_{self.action_}.pth"
        )
        model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=self.device))
        model.to(self.device)
        return model

    def reconstruction(self, x_in, y):
        x, param = torch.split(x_in, [self.in_ - self.hparam["N_REC_VARS"], self.hparam["N_REC_VARS"]], dim=1)

        rec_param = torch.zeros_like(param, dtype=torch.float32, requires_grad=True, device=self.device)

        optimizer = torch.optim.SGD([rec_param], lr=self.hparam["REC_LR"])
        # optimizer = torch.optim.Adam([rec_param], lr=self.hparam["REC_LR"])

        for epoch in range(self.hparam["REC_EPOCHS"]):
            optimizer.zero_grad()

            if self.model_name in ["ffn", "feedforward", "mlp"]:
                x_full = torch.cat((x, rec_param), dim=1)
                y_hat = self.model(x_full, None, training=False)
            else:
                y_hat = self.model(x, rec_param, training=False)
            if self.predict_delta:
                x_state = x_in[:, : self.out_]
                y = y - x_state
            loss = self.loss(y, y_hat)
            loss.backward()
            optimizer.step()

        rec_loss = self.loss(x_in, torch.cat((x, rec_param), dim=1))
        return {"param": param.cpu().detach().numpy(), "rec_param": rec_param.cpu().detach().numpy(), "loss": loss.cpu().detach().numpy(), "rec_loss": rec_loss.cpu().detach().numpy()}

    def sampling(self, x, y):
        res_dict = {}

        with tqdm(range(self.hparam["N_MODELS"]), unit="sampling", disable=self.hparam.get("QUIET", False)) as n_sampling:
            for n in n_sampling:
                torch.manual_seed(n)
                res_dict[n] = self.reconstruction(x, y)
        return res_dict

    def results(self, res_dict):
        min_key = min(res_dict, key=lambda k: res_dict[k]["loss"].item())
        return res_dict[min_key]


def determine_parameter(hparam, input, output, action):

    length_recon_in = len(input)
    length_recon_out = len(output)

    input = [input]
    output = [output]
    device = hparam.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    if hparam["SCALE"] == 1:
        cache_dir = get_repo_root() / "exp" / "data_cache"
        scaler_x = joblib.load(cache_dir / "scaler_x.pkl")
        scaler_y = joblib.load(cache_dir / "scaler_y.pkl")
        input = torch.Tensor(scaler_x.transform(input)).to(device)
        output = torch.Tensor(scaler_y.transform(output)).to(device)
    elif hparam["SCALE"] == 0:
        input = torch.Tensor(input).to(device)
        output = torch.Tensor(output).to(device)

    apl = reconstruction(hparam, in_=length_recon_in, out_=length_recon_out, action_=action)
    mode = hparam.get("PARAM_SEARCH", "gradient-descent").lower()
    n_rec_param = hparam['N_REC_VARS']

    def _param_bounds():
        try:
            cfg_path = domain_json_path(hparam["DS_DOMAIN"])
            with open(cfg_path) as f:
                cfg = json.load(f)
            downs = cfg.get("variable_parameters", {}).get("down", [])
            tops = cfg.get("variable_parameters", {}).get("top", [])
            bounds = []
            for i in range(n_rec_param):
                if i < len(downs) and i < len(tops):
                    bounds.append((float(downs[i]), float(tops[i])))
                else:
                    bounds.append((-1.0, 1.0))
            return bounds
        except Exception:
            return [(-1.0, 1.0) for _ in range(n_rec_param)]

    bounds = _param_bounds()

    def _eval_loss(param_vec):
        x_in = input.clone()
        x_in[0, -n_rec_param:] = torch.tensor(param_vec, dtype=x_in.dtype)
        x_fixed = x_in[:, : x_in.size(1) - n_rec_param]
        rec_param = x_in[:, x_in.size(1) - n_rec_param :]
        if apl.model_name in ["ffn", "feedforward", "mlp"]:
            x_full = torch.cat((x_fixed, rec_param), dim=1)
            y_hat = apl.model(x_full, None, training=False)
        else:
            y_hat = apl.model(x_fixed, rec_param, training=False)
        y = output.clone()
        if apl.predict_delta:
            x_state = x_in[:, : length_recon_out]
            y = y - x_state
        loss_val = apl.loss(y, y_hat).item()
        return loss_val

    if mode == "gradient-descent":
        res_dict = apl.sampling(input, output)
        final_res = apl.results(res_dict)
        new_param = input
        loss = final_res['rec_loss']
        new_param[0,(len(input[0])-n_rec_param):len(input[0])] = torch.from_numpy(final_res['rec_param'])
        back_to_smt = []
        if hparam["SCALE"] == 1:
            new_x_original = scaler_x.inverse_transform(new_param)
            for i in range(n_rec_param):
                back_to_smt.append(float(new_x_original[0,(len(input[0])-n_rec_param+i)]))
        else:
            for i in range(n_rec_param):
                back_to_smt.append(float(new_param[0,(len(input[0])-n_rec_param+i)]))
    elif mode == "beam":
        beam_width = int(hparam.get("BEAM_WIDTH", 8))
        beam_steps = int(hparam.get("BEAM_STEPS", 20))
        beam_std = float(hparam.get("BEAM_STD", 0.2))
        # init beam
        beam = []
        for _ in range(beam_width):
            vec = []
            for lo, hi in bounds:
                vec.append(lo + (hi - lo) * np.random.rand())
            beam.append(np.array(vec, dtype=float))
        best = None
        best_loss = float("inf")
        for _ in range(beam_steps):
            candidates = []
            for b in beam:
                for _ in range(beam_width):
                    pert = b + np.random.randn(*b.shape) * beam_std
                    for i, (lo, hi) in enumerate(bounds):
                        pert[i] = float(np.clip(pert[i], lo, hi))
                    candidates.append(pert)
            losses = [(_eval_loss(c), c) for c in candidates]
            losses.sort(key=lambda x: x[0])
            beam = [c for _, c in losses[:beam_width]]
            if losses[0][0] < best_loss:
                best_loss = losses[0][0]
                best = losses[0][1]
        back_to_smt = list(best)
        loss = best_loss
    elif mode == "finite_diff":
        fd_steps = int(hparam.get("FD_STEPS", 30))
        fd_step = float(hparam.get("FD_STEP_SIZE", 0.1))
        # init
        vec = []
        for lo, hi in bounds:
            vec.append(lo + (hi - lo) * np.random.rand())
        vec = np.array(vec, dtype=float)
        for _ in range(fd_steps):
            grad = np.zeros_like(vec)
            base_loss = _eval_loss(vec)
            for i, (lo, hi) in enumerate(bounds):
                eps = fd_step * 0.1
                vec_up = vec.copy()
                vec_dn = vec.copy()
                vec_up[i] = min(hi, vec_up[i] + eps)
                vec_dn[i] = max(lo, vec_dn[i] - eps)
                grad[i] = (_eval_loss(vec_up) - _eval_loss(vec_dn)) / (2 * eps)
            vec = vec - fd_step * grad
            for i, (lo, hi) in enumerate(bounds):
                vec[i] = float(np.clip(vec[i], lo, hi))
        back_to_smt = list(vec)
        loss = _eval_loss(vec)
    else:
        raise ValueError(f"Unknown PARAM_SEARCH mode: {mode}")

    if not hparam.get("QUIET", False):
        print(back_to_smt)

    return back_to_smt, loss


# Quickrun
if __name__ == "__main__":

    exp_setup_dir = get_exp_setup_dir()
    with open(exp_setup_dir / "hparams.json") as f:
        hparam = json.load(f)
    normalize_log_dir(hparam)

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
