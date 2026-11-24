import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential()
        for layer in range(n_layers - 1):
            self.encoder.add_module(f"linear_{layer}", nn.Linear(input_dim, hidden_dim)) if layer == 0 \
                else self.encoder.add_module(f"linear_{layer}", nn.Linear(hidden_dim, hidden_dim))
            self.encoder.add_module(f"relu_{layer}", nn.ReLU())
            self.encoder.add_module(f"dropout_{layer}", nn.Dropout(p=dropout))
        self.encoder.add_module(f"linear_{n_layers}", nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, n_layers, dropout):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential()
        for layer in range(n_layers - 1):
            self.decoder.add_module(f"linear_{layer}", nn.Linear(hidden_dim, hidden_dim))
            self.decoder.add_module(f"relu_{layer}", nn.ReLU())
            self.decoder.add_module(f"dropout_{layer}", nn.Dropout(p=dropout))
        self.decoder.add_module(f"linear_{n_layers}", nn.Linear(hidden_dim, output_dim))

    def forward(self, z):
        return self.decoder(z)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, n_layers // 2, dropout)
        self.decoder = Decoder(hidden_dim, output_dim, n_layers // 2, dropout)

    def forward(self, x, params, training=True): # while the x tensor is excluded from the gradient computation, the params tensor (as subset of x) is included in the gradient computation.
        if training is False:
            x = torch.cat([x, params], dim=1)

        z = self.encoder(x)
        y = self.decoder(z)
        return y



# Quicktest
if __name__ == "__main__":
    input_dim = 4  # Dimension of the input vector
    hparam = {"N_BLOCKS": 3, "N_HIDDEN": 2}

    model = Autoencoder(hparam, input_dim)
    x = torch.randn(10, input_dim)

    # Forward
    y = model(x)

    # Inverse
    x_hat = model(y, reverse=True)
    print(x)
    print(x_hat)
    print(y)
