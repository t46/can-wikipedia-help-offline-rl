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
    model_name = 'gpt2'
    batch_size = 100
    path_to_save_mi = 'path_to_save_mi'
    '''
    Followings are the paths to data and activation saved by 
    .../can-wikipdia-help-offline-rl/notebooks/section-52-mutual-information/save_activation.ipynb.
    So refer to the notebook for the path.
    '''
    path_to_d4rl_data_sample = 'path_to_d4rl_data_sample'
    path_to_activation = 'path_to_activation'

    data = np.load(f'{path_to_d4rl_data_sample}/data_{env_name}_{dataset_name}_{seed}_{batch_size}.npy', allow_pickle=True).item()
    activation = np.load(f'{path_to_activation}/activation_{epoch}_{model_name}_{env_name}_{dataset_name}_{seed}_{batch_size}.npy', allow_pickle=True).item()

    rtg = data['rtg'][:, :-1].to(device)  # (batch_size, K, dim)
    states = data['states'].to(device)  # (batch_size, K, dim)
    actions = data['actions'].to(device)  # (batch_size, K, dim)

    mi_dict = {}

    for key, value in tqdm(activation.items()):
        if 'mlp.dropout' in key:
            activation = value.to(device)
            mi_dict[key] = []
            state_mi_list = []
            action_mi_list = []
            for step in tqdm(range(states.shape[1])):
                try:
                    state_mi = calc_mi(states[:, step, :], activation[:, 3 * step + 1, :]).cpu().numpy()  # I(X; T)
                except:
                    state_mi = np.nan
                    print(f'{key}: state_mi is None')
                try:
                    action_mi = calc_mi(actions[:, step, :], activation[:, 3 * step + 1, :]).cpu().numpy()  # I(Y; T)
                except:
                    action_mi = np.nan
                    print(f'{key}: action_mi is None')
                state_mi_list.append(state_mi)
                action_mi_list.append(action_mi)
            mi_dict[key].append(state_mi_list)
            mi_dict[key].append(action_mi_list)

    np.save(f'{path_to_save_mi}/mi_{epoch}_{model_name}_{env_name}_{dataset_name}_{seed}_{batch_size}.npy', mi_dict)

if __name__ == '__main__':
    main()