import argparse
import sys

import numpy as np
import torch

sys.path.append("../")
from sample_batch_data import get_batch, get_data_info
from set_config import generate_variant
from signal_propagation import get_activation


def save_data_and_activation(
    path_to_load_dataset,
    path_to_model_checkpoint,
    path_to_save_d4rl_data_sample,
    path_to_save_activation,
    seed=666,
    model_name="gpt2",
    epoch=40,
    env_name_list=["hopper", "halfcheetah", "walker2d"],
):
    """Save activation and associated data.

    Args:
        path_to_load_dataset: Path to dataset to load from.
        path_to_model_checkpoint: Path to model checkpoint.
        path_to_save_d4rl_data_sample (str): Path to save a batch of sampled D4RL data.
        path_to_save_activation (str): Path to save a batch of activations of neural networks.
        seed (int, optional): random seed. Defaults to 666.
        model_name (str, optional): 'gpt2', 'igpt', or 'dt'. Defaults to 'gpt2'.
        epoch (int, optional): 0 or 40. Defaults to 40.
        env_name_list (list, optional): environment name list. Defaults to ['hopper', 'halfcheetah', 'walker2d'].
    """

    for env_name in env_name_list:

        torch.manual_seed(seed)

        dataset_name = "medium"

        variant = generate_variant(
            epoch, path_to_model_checkpoint, model_name, env_name, seed, dataset_name
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state_dim, act_dim, max_ep_len, scale = get_data_info(variant)
        states, actions, rewards, dones, rtg, timesteps, attention_mask = get_batch(
            variant, state_dim, act_dim, max_ep_len, scale, device, path_to_load_dataset
        )

        data = {
            "states": states,
            "actions": actions,
            "rtg": rtg,
            "timesteps": timesteps,
            "attention_mask": attention_mask,
        }

        activation = get_activation(
            variant,
            state_dim,
            act_dim,
            max_ep_len,
            states,
            actions,
            rewards,
            rtg,
            timesteps,
            attention_mask,
            device,
        )
        np.save(
            f"{path_to_save_activation}/activation_{epoch}_{model_name}_{env_name}_{dataset_name}_{seed}.npy",
            activation,
        )
        np.save(
            f"{path_to_save_d4rl_data_sample}/data_{env_name}_{dataset_name}_{seed}.npy",
            data,
        )


def main(args):
    save_data_and_activation(
        path_to_load_dataset=args["path_to_load_data"],
        path_to_model_checkpoint=args["path_to_load_model"],
        path_to_save_d4rl_data_sample=args["path_to_save_data"],
        path_to_save_activation=args["path_to_save_activation"],
        seed=args["seed"],
        model_name=args["model_name"],
        epoch=args["epoch"],
        env_name_list=["hopper"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_load_data", type=str)
    parser.add_argument("--path_to_load_model", type=str)
    parser.add_argument("--path_to_save_data", type=str)
    parser.add_argument("--path_to_save_activation", type=str)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument(
        "--env_name_list", nargs="+", default=["hopper", "halfcheetah", "walker2d"]
    )
    args = parser.parse_args()
    main(vars(args))
