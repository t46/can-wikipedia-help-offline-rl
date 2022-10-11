'''
Utility function to run the mutual information estimation experiment and save the mutual information.
'''
import numpy as np
import os
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
    activations,
):
    """Run the experiment to save the estimated mutual informaiton.

    Args:
        path_to_save_mi (str): Path to save the estimated mutual information.
        env_name (str): hopper, halfcheetah, or walker2d.
        dataset_name (str): medirum, expert, or random.
        seed (int): Random seed.
        epoch (int): Model checkpoint.
        model_name (str): dt, gpt2, or igpt.
        device (str): cpu/cuda.
        rtg (np.ndarray): Return-to-go.
        states (np.ndarray): States.
        actions (np.ndarray): Actions.
        activations (np.ndarray): Activations.
    """
    os.makedirs(path_to_save_mi, exist_ok=True)

    if model_name == "igpt":
        keys = ["0.mlp.dropout", "12.mlp.dropout", "23.mlp.dropout"]
    else:
        keys = ["6.mlp.dropout"]

    for key in tqdm(keys):

        mi_dict = {}

        value = activations[key]
        activation = value.to(device)

        mi_dict[key] = []
        x_mi_list = []  # Mutual information I(X; T) b/w input data X and activation.
        y_mi_list = []  # Mutual information I(Y; T) b/w label data Y and activation.
        for step in tqdm(range(states.shape[1])):
            for i in range(3):
                if (step == states.shape[1] - 1) and i == 2:  # Exclude action of K=1
                    pass
                else:  # Input to decision transformer is tuple of (R, s, a, R, s, a, ...)
                    if i == 0:
                        input_data = rtg
                    elif i == 1:
                        input_data = states
                    else:
                        input_data = actions

                    try:
                        x_mi = (
                            calc_mi(input_data[:, step, :], activation[:, -2, :], device)
                            .cpu()
                            .numpy()
                        )  # I(X; T)
                    except:
                        x_mi = np.nan
                        print(f"{key}: x_mi is None")
                    try:
                        y_mi = (
                            calc_mi(
                                actions[:, -1, :],
                                activation[:, (3 * step + i), :],
                                device,
                            )
                            .cpu()
                            .numpy()
                        )  # I(Y; T)
                    except:
                        y_mi = np.nan
                        print(f"{key}: y_mi is None")
                    x_mi_list.append(x_mi)
                    y_mi_list.append(y_mi)
        mi_dict[key].append(x_mi_list)
        mi_dict[key].append(y_mi_list)

        np.save(
            f"{path_to_save_mi}/mi_xy_{key}_{epoch}_{model_name}_{env_name}_{dataset_name}_{seed}.npy",
            mi_dict,
        )


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
    activations,
):
    """Run the experiment to save the estimated mutual informaiton b/w data and activation of the current step.

    Args:
        path_to_save_mi (str): Path to save the estimated mutual information.
        env_name (str): hopper, halfcheetah, or walker2d.
        dataset_name (str): medirum, expert, or random.
        seed (int): Random seed.
        epoch (int): Model checkpoint.
        model_name (str): dt, gpt2, or igpt.
        device (str): cpu/cuda.
        rtg (np.ndarray): Return-to-go.
        states (np.ndarray): States.
        actions (np.ndarray): Actions.
        activations (np.ndarray): Activations.
    """
    os.makedirs(path_to_save_mi, exist_ok=True)

    mi_dict = {}

    for key, value in tqdm(activations.items()):
        if "mlp.dropout" in key:
            activation = value.to(device)
            mi_dict[key] = []
            state_mi_list = []  # Mutual information I(X; T) b/w state and activation.
            action_mi_list = []  # Mutual information I(Y; T) b/w action and activation.
            for step in tqdm(range(states.shape[1])):
                try:
                    # Input to decision transformer is tuple of (R, s, a, R, s, a, ...)
                    state_mi = (
                        calc_mi(
                            states[:, step, :], activation[:, 3 * step + 1, :], device
                        )
                        .cpu()
                        .numpy()
                    )  # I(X; T)
                except:
                    state_mi = np.nan
                    print(f"{key}: state_mi is None")
                try:
                    action_mi = (
                        calc_mi(
                            actions[:, step, :], activation[:, 3 * step + 1, :], device
                        )
                        .cpu()
                        .numpy()
                    )  # I(Y; T)
                except:
                    action_mi = np.nan
                    print(f"{key}: action_mi is None")
                state_mi_list.append(state_mi)
                action_mi_list.append(action_mi)
            mi_dict[key].append(state_mi_list)
            mi_dict[key].append(action_mi_list)

    np.save(
        f"{path_to_save_mi}/mi_{epoch}_{model_name}_{env_name}_{dataset_name}_{seed}.npy",
        mi_dict,
    )


def run_to_save_mi_data(
    path_to_save_mi, env_name, dataset_name, seed, device, rtg, states, actions
):
    """Run the experiment to save the estimated mutual informaiton b/w data.

    Args:
        path_to_save_mi (str): Path to save the estimated mutual information.
        env_name (str): hopper, halfcheetah, or walker2d.
        dataset_name (str): medirum, expert, or random.
        seed (int): Random seed.
        device (str): cpu/cuda.
        rtg (np.ndarray): Return-to-go.
        states (np.ndarray): States.
        actions (np.ndarray): Actions.
    """
    os.makedirs(path_to_save_mi, exist_ok=True)

    mi_state_action_list = []  # Mutual information b/w state and action.
    mi_rtg_action_list = []  # Mutual information b/w return-to-go and action.
    for step in tqdm(range(states.shape[1])):
        try:
            mi_state_action = calc_mi(states[:, step, :], actions[:, step, :], device)
            mi_rtg_action = calc_mi(rtg[:, step, :], actions[:, step, :], device)
        except:
            mi_state_action = None
            mi_rtg_action = None
        mi_state_action_list.append(mi_state_action)
        mi_rtg_action_list.append(mi_rtg_action)

    np.save(
        f"{path_to_save_mi}/mi_data_state_action_{env_name}_{dataset_name}_{seed}.npy",
        mi_state_action_list,
    )
    np.save(
        f"{path_to_save_mi}/mi_data_rtg_action_{env_name}_{dataset_name}_{seed}.npy",
        mi_rtg_action_list,
    )
