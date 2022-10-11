'''
Compute, save, and plot the difference in attention distance.
'''
import argparse
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm.notebook import tqdm

sys.path.append("../")
from sample_batch_data import get_batch, get_data_info
from set_config import generate_variant
from signal_propagation import get_activation

sns.set_style("ticks")
sns.set_context("paper", 1.5, {"lines.linewidth": 2})


def main(args):

    models = ["gpt2", "dt"]
    seed = args["seed"]
    epoch1 = args["epoch1"]
    epoch2 = args["epoch2"]
    env_name = args["env_name"]
    dataset_name = args["dataset_name"]
    path_to_model_checkpoint = args["path_to_load_model"]
    path_to_dataset = args["path_to_load_data"]
    path_to_save_att_dist_diff = args["path_to_save_att_dist_diff"]
    path_to_save_figure = args["path_to_save_figure"]

    os.makedirs(path_to_save_att_dist_diff, exist_ok=True)
    os.makedirs(path_to_save_figure, exist_ok=True)

    att_dist_diff_abs_list = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model_name in tqdm(models):

        torch.manual_seed(seed)

        variant = generate_variant(
            epoch1, path_to_model_checkpoint, model_name, env_name, seed, dataset_name
        )

        state_dim, act_dim, max_ep_len, scale = get_data_info(variant)
        states, actions, rewards, dones, rtg, timesteps, attention_mask = get_batch(
            variant, state_dim, act_dim, max_ep_len, scale, device, path_to_dataset
        )

        # Get activations of epoch1 and epoch1, respectvely.
        att_dist_mat_list = []
        for _ in tqdm(range(2)):

            # For the first iteration, ues the `variant` defined above (epoch1).
            # For the second iteration, ues the `variant` defined below (epoch2).
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

            variant = generate_variant(
                epoch2,
                path_to_model_checkpoint,
                model_name,
                env_name,
                seed,
                dataset_name,
            )

            att_dist_mat = []
            layers = np.arange(24) if model_name == "igpt" else np.arange(12)

            for batchid in range(variant["batch_size"]):

                att_dist_list = []

                for layer in layers:

                    query = activation[f"{layer}.attn.c_attn"][
                        batchid, :, :768
                    ]  # 0, 5, 11
                    key = activation[f"{layer}.attn.c_attn"][
                        batchid, :, 768:1536
                    ]  # 0, 5, 11

                    att_weight_mat = (
                        torch.tril(
                            torch.softmax(
                                torch.matmul(query, key.t()) / np.sqrt(768), dim=1
                            )
                        )
                        .cpu()
                        .numpy()
                    )

                    dist_mat = np.zeros_like(att_weight_mat)
                    for i in range(dist_mat.shape[0]):
                        for j in range(dist_mat.shape[1]):
                            dist_mat[i, j] = abs(i - j)

                    dist_mat *= np.array(att_weight_mat)
                    att_dist = np.mean(dist_mat)

                    att_dist_list.append(att_dist)

                att_dist_mat.append(att_dist_list)

            att_dist_mat_list.append(att_dist_mat)

        att_dist_diff_abs = np.abs(
            np.array(att_dist_mat_list[0]) - np.array(att_dist_mat_list[1])
        )

        np.save(
            f"{path_to_save_att_dist_diff}/att_dist_diff_{epoch1}_{epoch2}_{model_name}_{env_name}_{dataset_name}_{seed}_K1.npy",
            att_dist_diff_abs,
        )

        if model_name == "igpt":
            att_dist_diff_abs_normalized = np.zeros(
                (variant["batch_size"], len(layers) // 2)
            )
            for batch_id in range(variant["batch_size"]):
                att_dist_diff_abs_normalized_persample = []
                for i in range(att_dist_diff_abs_normalized.shape[1]):
                    att_dist_diff_abs_normalized_persample.append(
                        att_dist_diff_abs[batch_id, i * 2: (i + 1) * 2].mean()
                    )
                att_dist_diff_abs_normalized[
                    batch_id, :
                ] = att_dist_diff_abs_normalized_persample
            att_dist_diff_abs = att_dist_diff_abs_normalized

        att_dist_diff_abs_list.append(att_dist_diff_abs)

    att_dist_diff_abs_cat = list(np.concatenate(att_dist_diff_abs_list).flatten())
    model_name = ["GPT2" for _ in range(1200)] + ["Random Init" for _ in range(1200)]
    block_id = list(np.array([[i for i in range(12)] for _ in range(200)]).flatten())

    df = pd.DataFrame(
        {
            "att_dist": att_dist_diff_abs_cat,
            "model_name": model_name,
            "block_id": block_id,
        }
    )

    sns.boxplot(
        x="block_id",
        y="att_dist",
        data=df,
        hue="model_name",
        palette={"GPT2": (0.372, 0.537, 0.537), "Random Init": (0.733, 0.737, 0.870)},
    )
    plt.xlabel("Transformer Block")
    plt.ylabel(r"$|d_{att}$" + f"({epoch2}) - " + r"$d_{att}$" + f"({epoch1})|")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(
        f"{path_to_save_figure}/att_dist_diff_{epoch1}_{epoch2}_gpt2_dt_{env_name}_{dataset_name}_{seed}_K1.pdf"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_load_data", type=str, default="../../data")
    parser.add_argument("--path_to_load_model", type=str, default="../../checkpoints")
    parser.add_argument("--path_to_save_att_dist_diff", type=str, default="results")
    parser.add_argument("--path_to_save_figure", type=str, default="figs")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--epoch1", type=int, default=0, help="A model checkpoint to compare attention distance.")
    parser.add_argument("--epoch2", type=int, default=4, help="Another model checkpoint to compare attention distance.")
    parser.add_argument("--env_name", type=str, default="hopper", help="hopper, halfcheetah, or walker2d")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="medium",
        help="Only medium is used for the experiments.",
    )  # medium, medium-replay, medium-expert, expert
    args = parser.parse_args()
    main(vars(args))
