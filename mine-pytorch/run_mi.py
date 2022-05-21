from mine.models.mine import Mine

import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch

device = 'cuda'

def calc_mi(X, Y):
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


def main():

    env_name = 'hopper'
    dataset_name = 'medium'
    seed = 666
    epoch = 40
    model_name = 'igpt'  # 'gpt2

    data = np.load(f'/root/projects/can-wikipedia-help-offline-rl/code/notebooks/data/data_{env_name}_{dataset_name}_{seed}.npy', allow_pickle=True).item()
    activations = np.load(f'/root/projects/can-wikipedia-help-offline-rl/code/notebooks/results/activation_{epoch}_{model_name}_{env_name}_{dataset_name}_{seed}.npy', allow_pickle=True).item()

    rtg = data['rtg'][:, :-1].to(device)
    states = data['states'].to(device)
    actions = data['actions'].to(device)

    if model_name == 'igpt':
        keys = ['0.mlp.dropout', '12.mlp.dropout', '23.mlp.dropout']
    else:
        keys = ['6.mlp.dropout']

    for key in keys:

        mi_dict = {}

        value = activations[key]
        activation = value.to(device)

        mi_dict[key] = []
        x_mi_list = []
        y_mi_list = []
        for step in tqdm(range(states.shape[1])):
            for i in range(3):
                if (step == states.shape[1] - 1) and i == 2:  # Exclude action of K=1
                    pass
                else:
                    if i == 0:
                        rsa = rtg
                    elif i == 1:
                        rsa = states
                    else:
                        rsa = actions

                    try:
                        x_mi = calc_mi(rsa[:, step, :], activation[:, -2, :]).cpu().numpy()  # I(X; T)
                    except:
                        x_mi = np.nan
                        print(f'{key}: x_mi is None')
                    try:
                        y_mi = calc_mi(actions[:, -1, :], activation[:, (3 * step + i), :]).cpu().numpy()  # I(Y; T)
                    except:
                        y_mi = np.nan
                        print(f'{key}: y_mi is None')
                    x_mi_list.append(x_mi)
                    y_mi_list.append(y_mi)
        mi_dict[key].append(x_mi_list)
        mi_dict[key].append(y_mi_list)

        np.save(f'results/mi_xy_{key}_{epoch}_{model_name}_{env_name}_{dataset_name}_666.npy', mi_dict)

if __name__ == '__main__':
    main()