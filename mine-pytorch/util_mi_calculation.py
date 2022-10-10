'''
Utility functions to calculate mutual information.
'''
import numpy as np
import torch
import torch.nn as nn

from mine.models.mine import Mine


def calc_mi(X, Y, device):
    """Estimate mutual information by MINE

    Args:
        X (torch.Tensor): Observed value of a random variable.
        Y (torch.Tensor): Observed value of another random variable.
        device (str): cpu/cuda.

    Returns:
        torch.Tensor: Estimated mutual information.
    """
    x_dim = X.shape[1]
    y_dim = Y.shape[1]

    statistics_network = nn.Sequential(
        nn.Linear(x_dim + y_dim, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 1),
    )

    mine = Mine(
        T=statistics_network, loss="mine", method="concat"  # mine_biased, fdiv
    ).to(device)

    mi = mine.optimize(
        X,
        Y,
        iters=1000,
        batch_size=50,
        opt=torch.optim.Adam(mine.parameters(), lr=1e-4),
    )

    del mine

    return mi


def load_data_and_activation(
    path_to_d4rl_data_sample,
    path_to_activation,
    env_name,
    dataset_name,
    seed,
    epoch,
    model_name,
    device,
):
    """Load D4RL data and activation.

    Args:
        path_to_d4rl_data_sample (str): Path to load D4RL data sample.
        path_to_activation (str): Path to load activation with the data.
        env_name (str): hopper, halfcheetah, or walker2d.
        dataset_name (str): medium, expert, or random.
        seed (int): Random seed.
        epoch (int): Model checkpoint.
        model_name (str): dt, gpt2, or igpt.
        device (str): cpu/cuda.

    Returns:
        tuple: return-to-go, states, actions, activations
    """
    data = np.load(
        f"{path_to_d4rl_data_sample}/data_{env_name}_{dataset_name}_{seed}.npy",
        allow_pickle=True,
    ).item()
    activations = np.load(
        f"{path_to_activation}/activation_{epoch}_{model_name}_{env_name}_{dataset_name}_{seed}.npy",
        allow_pickle=True,
    ).item()

    rtg = data["rtg"][:, :-1].to(device)  # (batch_size, K, dim)
    states = data["states"].to(device)  # (batch_size, K, dim)
    actions = data["actions"].to(device)  # (batch_size, K, dim)

    return rtg, states, actions, activations
