import json
import random
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from data_module import DataModule
from path_utils import get_exp_setup_dir
from train_module import determine_parameter, reconstruction
import torch
import joblib


def _evaluate_loss(hparam, x, y, params, model):
    n_rec = hparam["N_REC_VARS"]
    x_with = x.copy()
    x_with[-n_rec:] = params
    if hparam.get("SCALE", 0) == 1:
        from path_utils import get_repo_root
        cache_dir = get_repo_root() / "exp" / "data_cache"
        scaler_x = joblib.load(cache_dir / "scaler_x.pkl")
        scaler_y = joblib.load(cache_dir / "scaler_y.pkl")
        x_in = torch.Tensor(scaler_x.transform([x_with]))
        y_target = torch.Tensor(scaler_y.transform([y]))
    else:
        x_in = torch.Tensor([x_with])
        y_target = torch.Tensor([y])

    x_fixed = x_in[:, : x_in.size(1) - n_rec]
    rec_param = x_in[:, x_in.size(1) - n_rec :]
    if model.model_name in ["ffn", "feedforward", "mlp"]:
        x_full = torch.cat((x_fixed, rec_param), dim=1)
        y_hat = model.model(x_full, None, training=False)
    else:
        y_hat = model.model(x_fixed, rec_param, training=False)
    if model.predict_delta:
        x_state = x_in[:, : y_hat.size(1)]
        y_target = y_target - x_state
    return (y_hat - y_target).abs().max().item()


def main():
    exp_setup_dir = get_exp_setup_dir()
    with open(exp_setup_dir / "hparams.json") as f:
        hparam = json.load(f)

    data_path = DataModule.data_cache_path(hparam)
    x, y = DataModule(hparam, data_path=data_path).load_cached_data()
    idx = random.randrange(len(x))

    ml_action = hparam.get("ML_ACTION", "L_charge")
    recon = reconstruction(hparam, in_=x.shape[1], out_=y.shape[1], action_=ml_action)

    for mode in ["gradient-descent", "beam", "finite_diff"]:
        hparam_mode = deepcopy(hparam)
        hparam_mode["PARAM_SEARCH"] = mode
        params, _ = determine_parameter(hparam_mode, x[idx].copy(), y[idx].copy(), ml_action)
        loss = _evaluate_loss(hparam_mode, x[idx].copy(), y[idx].copy(), params, recon)
        print(f"{mode}: max_abs_dist={loss:.4f}")


if __name__ == "__main__":
    main()
