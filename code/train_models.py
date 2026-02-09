import argparse
import json
import random
from copy import deepcopy

from data_module import DataModule
from path_utils import get_exp_setup_dir, normalize_log_dir, to_repo_relative, get_log_dir, get_repo_root
from train_module import training, reconstruction, determine_parameter
import numpy as np
import torch
import joblib
import os
import tempfile


def parse_args():
    parser = argparse.ArgumentParser(description="Train models on cached data.")
    parser.add_argument("--tune", action="store_true", help="Run random hyperparameter tuning.")
    parser.add_argument("--trials", type=int, default=10, help="Number of tuning trials (random search).")
    parser.add_argument("--trial-patience", type=int, default=4, help="Val checks without improvement before stopping a trial.")
    parser.add_argument("--trial-diverge-factor", type=float, default=5.0, help="Stop trial if val_loss exceeds best*factor.")
    parser.add_argument("--trial-min-epochs", type=int, default=50, help="Minimum epochs before early-stop criteria apply.")
    parser.add_argument("--tune-concretization", action="store_true", help="Tune concretization hyperparameters.")
    parser.add_argument("--concretization-samples", type=int, default=64, help="Number of samples for concretization tuning.")
    parser.add_argument("--auto-write-hparams", action="store_true", help="Write best hyperparameters to hparams.json.")
    return parser.parse_args()


def main():
    args = parse_args()
    exp_setup_dir = get_exp_setup_dir()
    hparam_path = exp_setup_dir / "hparams.json"
    with open(hparam_path) as f:
        hparam = json.load(f)
    normalize_log_dir(hparam)

    data_path = DataModule.data_cache_path(hparam)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Cached dataset not found at {data_path}. "
            "Generate it first with: python code/generate_data.py"
        )

    def _atomic_write_json(path, payload):
        path = os.fspath(path)
        dir_name = os.path.dirname(path)
        os.makedirs(dir_name, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_hparams_", dir=dir_name)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _apply_best_hparams(best_hparams):
        if not best_hparams:
            return
        if not args.auto_write_hparams:
            return
        updated = dict(hparam)
        updated.update(best_hparams)
        _atomic_write_json(hparam_path, updated)
        hparam.update(best_hparams)

    if args.tune:
        x, _ = DataModule(hparam, data_path=data_path).load_cached_data()
        input_dim = x.shape[1]
        max_hidden = max(1, 3 * input_dim)
        candidates_hidden = [
            max(8, input_dim),
            max(8, 2 * input_dim),
            max(8, 3 * input_dim),
        ]
        candidates_hidden = sorted({h for h in candidates_hidden if h <= max_hidden})
        search_space = {
            "LEARNING_RATE": [1e-4, 3e-4, 1e-3],
            "N_HIDDEN": candidates_hidden,
            "N_LAYERS": [2, 3, 4],
            "DROPOUT": [0.0, 0.1, 0.2, 0.3],
            "WEIGHT_DECAY": [0.0, 1e-5, 1e-4],
            # gradient-descent concretization hyperparameters
            "REC_LR": [1e-1, 1.0, 1e1, 1e2],
            "REC_EPOCHS": [500, 1000, 3000],
            "N_MODELS": [4, 8, 16, 32],
            "opti_threshold": [1.0, 2.5, 5.0, 10.0],
        }
        best = {"val_loss": float("inf")}
        trials = []
        for i in range(args.trials):
            trial_hparam = deepcopy(hparam)
            for k, v in search_space.items():
                trial_hparam[k] = random.choice(v)
            trainer = training(trial_hparam, data_path=data_path)
            early_stop = {
                "patience": args.trial_patience,
                "diverge_factor": args.trial_diverge_factor,
                "min_epochs": args.trial_min_epochs,
            }
            results = trainer.train_nn(early_stop=early_stop)
            trials.append({"trial": i, **results, "hparam": {k: trial_hparam[k] for k in search_space}})
            if results["val_loss"] < best["val_loss"]:
                best = {"trial": i, **results, "hparam": {k: trial_hparam[k] for k in search_space}}
        log_dir = get_log_dir(trial_hparam)
        summary_path = log_dir / trial_hparam["DS_DOMAIN"] / "tuning_summary.json"
        with open(summary_path, "w") as f:
            json.dump({"best": best, "trials": trials}, f, indent=2)
        best_hparams = best.get("hparam", {})
        _apply_best_hparams(best_hparams)
        print(f"Tuning finished. Best val_loss={best['val_loss']} trial={best['trial']}")
        if best_hparams:
            print("Best hyperparameters:")
            for k, v in best_hparams.items():
                print(f"  {k}={v}")
            if args.auto_write_hparams:
                print("Best values were written to hparams.json. Rerun single training:")
                print("  python code/train_models.py")
            else:
                print("Run with --auto-write-hparams to update hparams.json automatically.")
        print(f"Tuning summary saved at: {to_repo_relative(summary_path)}")
    elif args.tune_concretization:
        x, y = DataModule(hparam, data_path=data_path).load_cached_data()
        n_samples = min(args.concretization_samples, len(x))
        indices = random.sample(range(len(x)), n_samples)
        search_space = {
            "REC_LR": [1e-1, 1.0, 1e1, 1e2],
            "REC_EPOCHS": [500, 1000, 3000],
            "N_MODELS": [4, 8, 16, 32],
        }
        best = {"mean_max_dist": float("inf")}
        trials = []
        for i in range(args.trials):
            print(f"Concretization tuning trial {i + 1}/{args.trials}...")
            trial_hparam = deepcopy(hparam)
            for k, v in search_space.items():
                trial_hparam[k] = random.choice(v)
            ml_action = trial_hparam.get("ML_ACTION", "L_charge")
            recon = reconstruction(trial_hparam, in_=x.shape[1], out_=y.shape[1], action_=ml_action)

            max_dists = []
            passes = 0
            for idx in indices:
                x_i = x[idx].copy()
                y_i = y[idx].copy()
                params, _ = determine_parameter(trial_hparam, x_i, y_i, ml_action)
                n_rec = trial_hparam["N_REC_VARS"]
                x_with = x_i.copy()
                x_with[-n_rec:] = params

                # scale if needed
                if trial_hparam.get("SCALE", 0) == 1:
                    cache_dir = get_repo_root() / "exp" / "data_cache"
                    scaler_x = joblib.load(cache_dir / "scaler_x.pkl")
                    scaler_y = joblib.load(cache_dir / "scaler_y.pkl")
                    x_in = torch.Tensor(scaler_x.transform([x_with]))
                    y_target = torch.Tensor(scaler_y.transform([y_i]))
                else:
                    x_in = torch.Tensor([x_with])
                    y_target = torch.Tensor([y_i])

                x_fixed = x_in[:, : x_in.size(1) - n_rec]
                rec_param = x_in[:, x_in.size(1) - n_rec :]
                if recon.model_name in ["ffn", "feedforward", "mlp"]:
                    x_full = torch.cat((x_fixed, rec_param), dim=1)
                    y_hat = recon.model(x_full, None, training=False)
                else:
                    y_hat = recon.model(x_fixed, rec_param, training=False)
                if recon.predict_delta:
                    x_state = x_in[:, : y_hat.size(1)]
                    y_target = y_target - x_state

                diff = (y_hat - y_target).abs().max().item()
                max_dists.append(diff)
                if diff <= trial_hparam["opti_threshold"]:
                    passes += 1

            mean_max_dist = float(np.mean(max_dists))
            pass_rate = passes / max(1, len(max_dists))
            trials.append({
                "trial": i,
                "mean_max_dist": mean_max_dist,
                "pass_rate": pass_rate,
                "hparam": {k: trial_hparam[k] for k in search_space},
            })
            if mean_max_dist < best["mean_max_dist"]:
                best = {"trial": i, "mean_max_dist": mean_max_dist, "pass_rate": pass_rate, "hparam": {k: trial_hparam[k] for k in search_space}}

        log_dir = get_log_dir(hparam)
        summary_path = log_dir / hparam["DS_DOMAIN"] / "concretization_tuning.json"
        with open(summary_path, "w") as f:
            json.dump({"best": best, "trials": trials}, f, indent=2)
        best_hparams = best.get("hparam", {})
        _apply_best_hparams(best_hparams)
        print(
            "Concretization tuning finished. "
            f"Best mean_max_dist={best['mean_max_dist']} "
            f"trial={best['trial']} pass_rate={best['pass_rate']}"
        )
        best_hparams = best.get("hparam", {})
        if best_hparams:
            print("Best concretization hyperparameters:")
            for k, v in best_hparams.items():
                print(f"  {k}={v}")
            if args.auto_write_hparams:
                print("Best values were written to hparams.json.")
            else:
                print("Run with --auto-write-hparams to update hparams.json automatically.")
        print(f"Tuning summary saved at: {to_repo_relative(summary_path)}")
    else:
        trainer = training(hparam, data_path=data_path)
        results = trainer.train_nn()
        delta = hparam.get("PREDICT_DELTA", 0) == 1
        model_name = hparam.get("MODEL", "autoencoder")
        print(f"Training finished. test_loss={results['test_loss']}")
        print(f"PREDICT_DELTA={int(delta)} (training used delta targets if 1)")
        print(f"MODEL={model_name}")


if __name__ == "__main__":
    main()
