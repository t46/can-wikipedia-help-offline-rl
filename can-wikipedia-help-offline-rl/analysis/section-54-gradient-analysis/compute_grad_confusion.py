'''
Compute, save, and plot the minimum cosine similarity of gradients.
'''
import argparse
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm

sys.path.append("../")
from sample_batch_data import get_batch, get_data_info
from set_config import generate_variant
from signal_propagation import get_gradients

sns.set_style("ticks")
sns.set_context("paper", 1.5, {"lines.linewidth": 2})


def main(args):
    model_names = ["gpt2", "igpt", "dt"]
    seed = args["seed"]
    epoch = args["epoch"]
    dataset_name = args["dataset_name"]
    env_name = args["env_name"]
    path_to_dataset = args["path_to_load_data"]
    path_to_model_checkpoint = args["path_to_load_model"]
    path_to_save_grad_cossim = args["path_to_save_gradcossim"]
    path_to_save_figure = args["path_to_save_figure"]

    os.makedirs(path_to_save_grad_cossim, exist_ok=True)
    os.makedirs(path_to_save_figure, exist_ok=True)

    min_gradcossims_list = []

    device = torch.device(args["device"])
    for model_name in tqdm(model_names):

        torch.manual_seed(seed)

        variant = generate_variant(
            epoch,
            path_to_model_checkpoint,
            model_name,
            env_name,
            seed,
            dataset_name,
            batch_size=50,
        )

        state_dim, act_dim, max_ep_len, scale = get_data_info(variant)
        states, actions, rewards, dones, rtg, timesteps, attention_mask = get_batch(
            variant, state_dim, act_dim, max_ep_len, scale, device, path_to_dataset
        )
        grads_list = get_gradients(
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
        # Compute cosine similarity between gradients of different data samples.
        gradcossim_list = []
        for grads1 in tqdm(grads_list):
            for grads2 in grads_list:
                gradcossim_list.append(
                    (
                        torch.dot(grads1, grads2)
                        / (1e-6 + torch.norm(grads1) * torch.norm(grads2))
                    ).numpy()
                )
        gradcossim = np.array(gradcossim_list)
        np.save(
            f"{path_to_save_grad_cossim}/gradcossim_{epoch}_{model_name}_{env_name}_{dataset_name}_{seed}.npy",
            gradcossim,
        )

        # Measure gradient confusion by the minimum gradient cosine similarity.
        min_gradcossims_list.append([min(gradcossim_list)])

    model_name_label = ["GPT2", "iGPT", "Random Init"]
    colors = [(0.372, 0.537, 0.537), (0.627, 0.352, 0.470), (0.733, 0.737, 0.870)]
    my_palette = sns.color_palette(colors)
    sns.barplot(data=min_gradcossims_list, palette=my_palette)
    plt.xticks(np.arange(3), model_name_label)
    plt.ylabel(r"Min of $cossim(\nabla_{\theta}\ell_1, \nabla_{\theta}\ell_2)$")
    plt.savefig(
        f"{path_to_save_figure}/mingradcossim_{epoch}_gpt2_igpt_dt_{env_name}_{dataset_name}_{seed}.pdf"
    )
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_load_data", "-pld", type=str, default="../../data")
    parser.add_argument("--path_to_load_model", "-plm", type=str, default="../../checkpoints")
    parser.add_argument("--path_to_save_gradcossim", "-psg", type=str, default="results")
    parser.add_argument("--path_to_save_figure", "-psf", type=str, default="figs")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--env_name", type=str, default="hopper", help="hopper, halfcheetah, or walker2d")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="medium",
        help="Only medium is used for the experiments.",
    )  # medium, medium-replay, medium-expert, expert
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    main(vars(args))
