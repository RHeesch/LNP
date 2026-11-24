import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_gens import drone as drone
from data_gens import flipsi as flipsi
from data_gens import zeno as zeno
from data_gens import cashpoint as cashpoint
from data_gens import drone_scale as drone_scale

class DataModule(nn.Module):
    def __init__(self, hparam, scaling=False):
        super(DataModule, self).__init__()
        self.hparam = hparam
        self.scaling = scaling

        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def gen_data(self):
        """
        Generating training data from problem domains.
        """
        if "drone_scale" in self.hparam["DS_DOMAIN"]:
            x, y =  drone_scale.create_Drone_scale_dataset(n_samples=self.hparam["N_SAMPLES"])
        elif "flipsi" in self.hparam["DS_DOMAIN"]:
            x, y = flipsi.create_FliPsi_dataset(n_samples=self.hparam["N_SAMPLES"])
        elif "zeno" in self.hparam["DS_DOMAIN"]:
            x, y = zeno.create_zeno_dataset(n_samples=self.hparam["N_SAMPLES"])
        elif "cashpoint" in self.hparam["DS_DOMAIN"]:
            x, y = cashpoint.create_cashpoint_dataset(n_samples=self.hparam["N_SAMPLES"])
        elif "drone" in self.hparam["DS_DOMAIN"]:
            x, y =  drone.create_drone_dataset(n_samples=self.hparam["N_SAMPLES"])
        return x, y

    def scale_ds(self, x, y):
        x_scaled = self.scaler_x.fit_transform(x)
        y_scaled = self.scaler_y.fit_transform(y)

        directory = "../exp/exp_setup/"
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.scaler_x, os.path.join(directory, "scaler_x.pkl"))
        joblib.dump(self.scaler_y, os.path.join(directory, "scaler_y.pkl"))
        return x_scaled, y_scaled

    def sampler(self):
        x, y = self.gen_data()

        if self.scaling:
            x, y = self.scale_ds(x, y)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.4, random_state=self.hparam["SEED"])
        x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, random_state=self.hparam["SEED"])
        return x_train, x_val, x_test, y_train, y_val, y_test

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
    with open('exp/exp_setup/hparams.json') as f:
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