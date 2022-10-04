from mine.models.mine import Mine

import numpy as np
import torch.nn as nn
from tqdm import tqdm
import os
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
    env_names = ['hopper', 'halfcheetah', 'walker2d']
    dataset_name = 'medium'
    seed = 666
    batch_size = 100
    path_to_save_mi = 'path_to_save_mi'
    '''
    The following is the path to data saved by 
    .../can-wikipdia-help-offline-rl/notebooks/section-52-mutual-information/save_activation.ipynb.
    So refer to the notebook for the path.
    '''
    path_to_d4rl_data_sample = 'path_to_d4rl_data_sample'

    for env_name in env_names:

        data = np.load(f'{path_to_d4rl_data_sample}/data_{env_name}_{dataset_name}_{seed}_{batch_size}.npy', allow_pickle=True).item()

        rtg = data['rtg'][:, :-1].to(device)  # (batch_size, K, dim)
        states = data['states'].to(device)  # (batch_size, K, dim)
        actions = data['actions'].to(device)  # (batch_size, K, dim)

        mi_state_action_list = []
        mi_rtg_action_list = []
        for step in tqdm(range(states.shape[1])):
            try:
                mi_state_action = calc_mi(states[:, step, :], actions[:, step, :])
                mi_rtg_action = calc_mi(rtg[:, step, :], actions[:, step, :])
            except:
                mi_state_action = None
                mi_rtg_action = None
            mi_state_action_list.append(mi_state_action)
            mi_rtg_action_list.append(mi_rtg_action)

        np.save(f'{path_to_save_mi}/mi_data_state_action_{env_name}_{dataset_name}_{seed}_{batch_size}.npy', mi_state_action_list)
        np.save(f'{path_to_save_mi}/mi_data_rtg_action_{env_name}_{dataset_name}_{seed}_{batch_size}.npy', mi_rtg_action_list)


if __name__ == '__main__':
    main()