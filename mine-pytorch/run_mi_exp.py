import argparse

import numpy as np
from tqdm import tqdm

from util_mi_calculation import load_data_and_activation
from util_to_save_mi import (
    run_to_save_mi,
    run_to_save_mi_data,
    run_to_save_mi_no_context,
)

device = "cuda"


def main(variant):

    env_name = variant["env_name"]
    dataset_name = variant["dataset_name"]
    seed = variant["seed"]
    epoch = variant["epoch"]
    model_name = variant["model_name"]
    path_to_save_mi = variant["path_to_save_mi"]
    path_to_d4rl_data_sample = variant["path_to_data"]
    path_to_activation = variant["path_to_activation"]
    exp_type = variant["exp_type"]

    rtg, states, actions, activations = load_data_and_activation(
        path_to_d4rl_data_sample,
        path_to_activation,
        env_name,
        dataset_name,
        seed,
        epoch,
        model_name,
        device,
    )
    if exp_type == "normal":
        run_to_save_mi(
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
        )
    elif exp_type == "no_context":
        run_to_save_mi_no_context(
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
        )
    elif exp_type == "data":
        run_to_save_mi_data(
            path_to_save_mi, env_name, dataset_name, seed, device, rtg, states, actions
        )
    else:
        print("No such option")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_save_mi", type=str)
    parser.add_argument("--path_to_data", type=str)
    parser.add_argument("--path_to_activation", type=str)
    parser.add_argument("--exp_type", type=str, default="normal")
    parser.add_argument("--env_name", type=str, default="hopper")
    parser.add_argument("--dataset_name", type=str, default="medium")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--model_name", type=str, default="gpt2")
    args = parser.parse_args()
    main(variant=vars(args))
