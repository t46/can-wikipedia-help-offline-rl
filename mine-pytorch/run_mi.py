import argparse
import numpy as np
from tqdm import tqdm
from util_mi_calculation import calc_mi, load_data_and_activation


device = 'cuda'

def main(variant):

    env_name = variant['env_name']
    dataset_name = variant['dataset_name']
    seed = variant['seed']
    epoch = variant['epoch']
    model_name = variant['model_name']
    batch_size = variant['batch_size']
    path_to_save_mi = variant['path_to_save_mi']
    path_to_d4rl_data_sample = variant['path_to_data']
    path_to_activation = variant['path_to_activation']

    rtg, states, actions, activations = load_data_and_activation(path_to_d4rl_data_sample,
                                                                 path_to_activation,
                                                                 env_name,
                                                                 dataset_name,
                                                                 seed,
                                                                 batch_size,
                                                                 epoch,
                                                                 model_name,
                                                                 device
                                                                )

    if model_name == 'igpt':
        keys = ['0.mlp.dropout', '12.mlp.dropout', '23.mlp.dropout']
    else:
        keys = ['6.mlp.dropout']

    for key in tqdm(keys):

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
                        x_mi = calc_mi(rsa[:, step, :], activation[:, -2, :], device).cpu().numpy()  # I(X; T)
                    except:
                        x_mi = np.nan
                        print(f'{key}: x_mi is None')
                    try:
                        y_mi = calc_mi(actions[:, -1, :], activation[:, (3 * step + i), :], device).cpu().numpy()  # I(Y; T)
                    except:
                        y_mi = np.nan
                        print(f'{key}: y_mi is None')
                    x_mi_list.append(x_mi)
                    y_mi_list.append(y_mi)
        mi_dict[key].append(x_mi_list)
        mi_dict[key].append(y_mi_list)

        np.save(f'{path_to_save_mi}/mi_xy_{key}_{epoch}_{model_name}_{env_name}_{dataset_name}_{seed}_{batch_size}.npy', mi_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_save_mi", type=str)
    parser.add_argument("--path_to_data", type=str)
    parser.add_argument("--path_to_activation", type=str)
    parser.add_argument("--env_name", type=str, default="hopper")
    parser.add_argument("--dataset_name", type=str, default="medium")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--model_name", type=str, default='gpt2')
    args = parser.parse_args()
    main(variant=vars(args))