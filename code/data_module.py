import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_gens.json_loader import domain_json_path, generate_dataset_from_json
from path_utils import get_exp_setup_dir, get_repo_root

class DataModule(nn.Module):
    def __init__(self, hparam, scaling=False, data_path=None):
        super(DataModule, self).__init__()
        self.hparam = hparam
        self.scaling = scaling
        self.data_path = Path(data_path) if data_path else None
        self._split_cache = None

        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    @staticmethod
    def data_cache_path(hparam) -> Path:
        repo_root = get_repo_root()
        cache_dir = repo_root / "exp" / "data_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{hparam['DS_DOMAIN']}_n{hparam['N_SAMPLES']}_seed{hparam['SEED']}.npz"
        return cache_dir / file_name

    @staticmethod
    def generate_and_store(hparam, force=False) -> Path:
        """
        Generate data once and store it. Returns the stored file path.
        """
        path = DataModule.data_cache_path(hparam)
        if path.exists() and not force:
            return path

        seed = hparam.get("SEED", None)
        if seed is not None:
            import random
            random.seed(seed)
            np.random.seed(seed)
            try:
                import torch
                torch.manual_seed(seed)
            except Exception:
                pass

        domain_path = domain_json_path(hparam["DS_DOMAIN"])
        if not domain_path.exists():
            raise FileNotFoundError(f"Domain JSON not found at {domain_path}")
        x, y = generate_dataset_from_json(domain_path, n_samples=hparam["N_SAMPLES"])

        np.savez(path, x=x, y=y)
        return path

    def load_cached_data(self):
        path = self.data_path or self.data_cache_path(self.hparam)
        if not path.exists():
            raise FileNotFoundError(
                f"Data cache not found at {path}. "
                "Generate it first via DataModule.generate_and_store(hparam)."
            )
        with np.load(path) as data:
            return data["x"], data["y"]

    def scale_ds(self, x, y):
        x_scaled = self.scaler_x.fit_transform(x)
        y_scaled = self.scaler_y.fit_transform(y)

        repo_root = get_repo_root()
        cache_dir = repo_root / "exp" / "data_cache"
        os.makedirs(cache_dir, exist_ok=True)
        joblib.dump(self.scaler_x, cache_dir / "scaler_x.pkl")
        joblib.dump(self.scaler_y, cache_dir / "scaler_y.pkl")
        return x_scaled, y_scaled

    def sampler(self):
        if self._split_cache is not None:
            return self._split_cache

        x, y = self.load_cached_data()

        if self.scaling:
            x, y = self.scale_ds(x, y)

        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=0.4, random_state=self.hparam["SEED"]
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_val, y_val, test_size=0.5, random_state=self.hparam["SEED"]
        )
        self._split_cache = (x_train, x_val, x_test, y_train, y_val, y_test)
        return self._split_cache

    def train_loader(self):
        x_train, _, _, y_train, _, _ = self.sampler()
        ds_train = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
        return DataLoader(ds_train, batch_size=self.hparam["BATCH_SIZE"],shuffle=True, drop_last=True, )

    def val_loader(self):
        _, x_val, _, _, y_val, _ = self.sampler()
        ds_train = TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))
        return DataLoader(ds_train, batch_size=self.hparam["BATCH_SIZE"], shuffle=True, drop_last=True)

    def test_loader(self):
        _, _, x_test, _, _, y_test = self.sampler()
        ds_train = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
        return DataLoader(ds_train, batch_size=self.hparam["BATCH_SIZE"], shuffle=True, drop_last=True)


# Quicktest
if __name__ == "__main__":
    exp_setup_dir = get_exp_setup_dir()
    with open(exp_setup_dir / "hparams.json") as f:
        hparam = json.load(f)

    print(hparam)

    dm = DataModule(hparam, scaling=True)
    dl_train = dm.train_loader()
    dl_val = dm.val_loader()
    dl_test = dm.test_loader()

    sample_train = next(iter(dl_train))
    sample_val = next(iter(dl_val))
    sample_test = next(iter(dl_test))

    print("dl lens:\n")
    print(f"dl_train: {len(dl_train)}")
    print(f"dl_val: {len(dl_val)}")
    print(f"dl_test: {len(dl_test)}")

    print("sample dimensions:\n")
    print(f"training_x: {sample_train[0].shape}, training_y: {sample_train[1].shape}")
    print(f"val_x: {sample_val[0].shape}, val_y: {sample_val[1].shape}")
    print(f"test_x: {sample_test[0].shape}, test_y: {sample_test[1].shape}")

    print(sample_test[0])
    print(sample_test[1])
