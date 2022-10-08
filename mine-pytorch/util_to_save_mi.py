import numpy as np
from tqdm import tqdm

from util_mi_calculation import calc_mi


def run_to_save_mi(
    path_to_save_mi,
    env_name,
    dataset_name,
    seed,
    epoch,
    model_name,
    device,
    rtg,
    states,
    actions,
    activations
):
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

        np.save(f'{path_to_save_mi}/mi_xy_{key}_{epoch}_{model_name}_{env_name}_{dataset_name}_{seed}.npy', mi_dict)


def run_to_save_mi_no_context(
    path_to_save_mi,
    env_name,
    dataset_name,
    seed,
    epoch,
    model_name,
    device,
    states,
    actions,
    activations
):
    mi_dict = {}

    for key, value in tqdm(activations.items()):
        if 'mlp.dropout' in key:
            activation = value.to(device)
            mi_dict[key] = []
            state_mi_list = []
            action_mi_list = []
            for step in tqdm(range(states.shape[1])):
                try:
                    state_mi = calc_mi(states[:, step, :], activation[:, 3 * step + 1, :], device).cpu().numpy()  # I(X; T)
                except:
                    state_mi = np.nan
                    print(f'{key}: state_mi is None')
                try:
                    action_mi = calc_mi(actions[:, step, :], activation[:, 3 * step + 1, :], device).cpu().numpy()  # I(Y; T)
                except:
                    action_mi = np.nan
                    print(f'{key}: action_mi is None')
                state_mi_list.append(state_mi)
                action_mi_list.append(action_mi)
            mi_dict[key].append(state_mi_list)
            mi_dict[key].append(action_mi_list)

    np.save(f'{path_to_save_mi}/mi_{epoch}_{model_name}_{env_name}_{dataset_name}_{seed}.npy', mi_dict)

def run_to_save_mi_data(
    path_to_save_mi,
    env_name,
    dataset_name,
    seed,
    device,
    rtg,
    states,
    actions
):
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

    np.save(f'{path_to_save_mi}/mi_data_state_action_{env_name}_{dataset_name}_{seed}.npy', mi_state_action_list)
    np.save(f'{path_to_save_mi}/mi_data_rtg_action_{env_name}_{dataset_name}_{seed}.npy', mi_rtg_action_list)