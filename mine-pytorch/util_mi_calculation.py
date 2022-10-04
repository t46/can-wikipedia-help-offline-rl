import torch
import torch.nn as nn
import numpy as np
from mine.models.mine import Mine


def calc_mi(X, Y, device):
    x_dim = X.shape[1]
    y_dim = Y.shape[1]

    statistics_network = nn.Sequential(
    nn.Linear(x_dim + y_dim, 400),
    nn.ReLU(),
    nn.Linear(400, 400),
    nn.ReLU(),
    nn.Linear(400, 400),
    nn.ReLU(),
    nn.Linear(400, 1)
)

    mine = Mine(
        T = statistics_network,
        loss = 'mine', #mine_biased, fdiv
        method = 'concat'
    ).to(device)

    mi = mine.optimize(X, Y, iters = 1000, batch_size=50, opt=torch.optim.Adam(mine.parameters(), lr=1e-4))

    del mine

    return mi

def load_data_and_activation(
    path_to_d4rl_data_sample,
    path_to_activation,
    env_name,
    dataset_name,
    seed,
    batch_size,
    epoch,
    model_name,
    device
    ):
    data = np.load(f'{path_to_d4rl_data_sample}/data_{env_name}_{dataset_name}_{seed}_{batch_size}.npy', allow_pickle=True).item()
    activations = np.load(f'{path_to_activation}/activation_{epoch}_{model_name}_{env_name}_{dataset_name}_{seed}_{batch_size}.npy', allow_pickle=True).item()

    rtg = data['rtg'][:, :-1].to(device)  # (batch_size, K, dim)
    states = data['states'].to(device)  # (batch_size, K, dim)
    actions = data['actions'].to(device)  # (batch_size, K, dim)

    return rtg, states, actions, activations