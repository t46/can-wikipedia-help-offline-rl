"""
Compute and plot L2 distance and cosine similarity of parameters from two different models.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

sys.path.append("../")
from sample_batch_data import get_data_info
from set_config import generate_variant

sys.path.append("../../")
from decision_transformer.models.decision_transformer import \
    DecisionTransformer

sns.set_style("ticks")
sns.set_context("paper", 1.5, {"lines.linewidth": 2})


def main(args):
    seed = args["seed"]
    epoch1 = args["epoch1"]
    epoch2 = args["epoch2"]
    env_name = args["env_name"]
    dataset_name = args["dataset_name"]
    path_to_model_checkpoint = args["path_to_load_model"]
    path_to_save_result = args["path_to_save_result"]
    path_to_save_figure = args["path_to_save_figure"]

    os.makedirs(path_to_save_result, exist_ok=True)
    os.makedirs(path_to_save_figure, exist_ok=True)

    model_names = ["gpt2", "igpt", "dt"]

    for model_name in model_names:

        torch.manual_seed(seed)

        variant = generate_variant(
            epoch1, path_to_model_checkpoint, model_name, env_name, seed, dataset_name
        )

        state_dim, act_dim, max_ep_len, scale = get_data_info(variant)

        model1 = DecisionTransformer(
            args=variant,
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=variant["K"],
            max_ep_len=max_ep_len,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=0.1,
        )
        if variant["load_checkpoint"]:
            state_dict = torch.load(
                variant["load_checkpoint"], map_location=torch.device("cpu")
            )
            model1.load_state_dict(state_dict)
            print(f"Loaded from {variant['load_checkpoint']}")

            model1.eval()

        variant = generate_variant(
            epoch2, path_to_model_checkpoint, model_name, env_name, seed, dataset_name
        )

        model2 = DecisionTransformer(
            args=variant,
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=variant["K"],
            max_ep_len=max_ep_len,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=0.1,
        )
        if variant["load_checkpoint"]:
            state_dict = torch.load(
                variant["load_checkpoint"], map_location=torch.device("cpu")
            )
            model2.load_state_dict(state_dict)
            print(f"Loaded from {variant['load_checkpoint']}")

            model2.eval()

        param_dist = {}
        param_cos = {}
        for (name1, param1), (name2, param2) in zip(
            model1.transformer.h.named_parameters(),
            model2.transformer.h.named_parameters(),
        ):
            param_dist[name1] = torch.linalg.norm(param2 - param1).detach().numpy()
            param_cos[name1] = (
                (
                    torch.dot(param1.view(-1), param2.view(-1))
                    / (torch.linalg.norm(param1) * torch.linalg.norm(param2) + 1e-6)
                )
                .detach()
                .numpy()
            )

        np.save(
            f"{path_to_save_result}/paramdist_{epoch1}_{epoch2}_{model_name}_{env_name}_{dataset_name}_{seed}.npy",
            param_dist,
        )
        np.save(
            f"{path_to_save_result}/paramcos_{epoch1}_{epoch2}_{model_name}_{env_name}_{dataset_name}_{seed}.npy",
            param_cos,
        )

    sim_metrics_list = ["paramdist", "paramcos"]
    param_sim_dicts = {
        sim_metrics: {
            model_name: np.load(
                f"{path_to_save_result}/{sim_metrics}_{epoch1}_{epoch2}_{model_name}_{env_name}_{dataset_name}_{seed}.npy",
                allow_pickle=True,
            ).item()
            for model_name in model_names
        }
        for sim_metrics in sim_metrics_list
    }

    colors = {
        "gpt2": (0.372, 0.537, 0.537),
        "igpt": (0.627, 0.352, 0.470),
        "dt": (0.733, 0.737, 0.870),
    }
    labels = {"gpt2": "GPT2", "igpt": "iGPT", "dt": "Random Init"}

    for dist_of_cos, paramsim_dict in param_sim_dicts.items():
        if dist_of_cos == "paramdist":
            fig_y_label = r"$||\theta_T - \theta_0||_2$"
            y_min, y_max = 0, 80
        else:
            fig_y_label = r"$cossim(\theta_T, \theta_0)$"
            y_min, y_max = 0, 1
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 7))
        fig.subplots_adjust(bottom=0.1)
        for i, ax in enumerate(axes):
            model_name = model_names[i]
            ax.bar(
                x=range(len(paramsim_dict[model_name])),
                height=list(paramsim_dict[model_name].values()),
                color=colors[model_name],
                label=labels[model_name],
            )
            ax.set_ylim(y_min, y_max)
            ax.set_ylabel(fig_y_label, fontsize=15)
            ax.legend(loc="upper right")
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.15)
        plt.xlabel("Parameter of Each Layer", fontsize=15)
        plt.savefig(
            f"{path_to_save_figure}/{dist_of_cos}_{epoch1}_{epoch2}_gpt2_igpt_dt_{env_name}_{dataset_name}_{seed}.pdf"
        )
        plt.close()
    print(f"Labels of xticks: \n {list(paramsim_dict[model_name].keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_load_data", "-pld", type=str, default="../../data")
    parser.add_argument(
        "--path_to_load_model", "-plm", type=str, default="../../checkpoints"
    )
    parser.add_argument("--path_to_save_result", "-psr", type=str, default="results")
    parser.add_argument("--path_to_save_figure", "-psf", type=str, default="figs")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument(
        "--epoch1",
        type=int,
        default=0,
        help="A model checkpoint to measure parameter similarity.",
    )
    parser.add_argument(
        "--epoch2",
        type=int,
        default=40,
        help="Another model checkpoint to measure parameter similarity.",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="hopper",
        help="hopper, halfcheetah, or walker2d",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="medium",
        help="Only medium is used for the experiments.",
    )  # medium, medium-replay, medium-expert, expert
    args = parser.parse_args()
    main(vars(args))
