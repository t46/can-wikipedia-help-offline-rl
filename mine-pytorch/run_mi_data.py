import argparse
import numpy as np
from tqdm import tqdm
from util_mi_calculation import calc_mi, load_data_and_activation


device = 'cuda'

def main(variant):
    env_names = ['hopper', 'halfcheetah', 'walker2d']
    dataset_name = variant['dataset_name']
    seed = variant['seed']
    epoch = variant['epoch']
    model_name = variant['model_name']
    batch_size = variant['batch_size']
    path_to_save_mi = variant['path_to_save_mi']
    path_to_d4rl_data_sample = variant['path_to_data']
    path_to_activation = variant['path_to_activation']

    for env_name in env_names:

        rtg, states, actions, _ = load_data_and_activation(path_to_d4rl_data_sample,
                                                           path_to_activation,
                                                           env_name,
                                                           dataset_name,
                                                           seed,
                                                           batch_size,
                                                           epoch,
                                                           model_name,
                                                           device
                                                           )

        mi_state_action_list = []
        mi_rtg_action_list = []
        for step in tqdm(range(states.shape[1])):
            try:
                mi_state_action = calc_mi(states[:, step, :], actions[:, step, :], device)
                mi_rtg_action = calc_mi(rtg[:, step, :], actions[:, step, :], device)
            except:
                mi_state_action = None
                mi_rtg_action = None
            mi_state_action_list.append(mi_state_action)
            mi_rtg_action_list.append(mi_rtg_action)

        np.save(f'{path_to_save_mi}/mi_data_state_action_{env_name}_{dataset_name}_{seed}_{batch_size}.npy', mi_state_action_list)
        np.save(f'{path_to_save_mi}/mi_data_rtg_action_{env_name}_{dataset_name}_{seed}_{batch_size}.npy', mi_rtg_action_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_save_mi", type=str)
    parser.add_argument("--path_to_data", type=str)
    parser.add_argument("--path_to_activation", type=str)
    parser.add_argument("--dataset_name", type=str, default="medium")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--model_name", type=str, default='gpt2')
    args = parser.parse_args()
    main(variant=vars(args))