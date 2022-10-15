"""
Plot diagonal element of CKA matrix and compare them among different models
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("ticks")
sns.set_context("paper", 1.5, {"lines.linewidth": 2})


def smooth_sequence(sequence):
    """Take average over two adjacent elements starting from the 0th item for model comparison.

    Args:
        sequence (list): list to be smoothed.

    Returns:
        np.ndarray: smoothed sequence.
    """
    sequence_normalized = []
    for i in range(len(sequence) // 2):
        sequence_normalized.append(sequence[i * 2 : (i + 1) * 2].mean())
    sequence = np.array(sequence_normalized)
    return sequence


def get_large_cka_module_name(cka_diag, cka_threshold):
    """Get layer names of CKAs that are larger than a threshold.

    Args:
        cka_diag (np.ndarray): CKA matrix.
        cka_threshold (float): threshold above which layer names are returned.

    Returns:
        list: layer names.
    """
    block_name_list = [
        "ln_1",
        "attn.c_attn",
        "attn.c_proj",
        "attn.resid_dropout",
        "ln_2",
        "mlp.c_fc",
        "mlp.c_proj",
        "mlp.act",
        "mlp.dropout",
    ]
    layer_name_list = []
    for i in range(12):
        for block_name in block_name_list:
            layer_name_list.append(str(i) + "." + block_name)
    module_idx_large_cka = list(np.where(cka_diag > cka_threshold)[0])
    module_name_large_cka = [
        layer_name_list[module_id] for module_id in module_idx_large_cka
    ]

    return module_name_large_cka


def main(args):
    seed = args["seed"]
    epoch = args["epoch"]
    env_name = args["env_name"]
    rtg_state_action = args["rtg_state_action"]
    cka_threshold_rand_init = args["cka_threshold_rand_init"]
    path_to_cka = args["path_to_load_cka"]
    path_to_save_figure = args["path_to_save_figure"]

    model_names = ["dt", "gpt2", "igpt"]
    colors = {
        "gpt2": (0.372, 0.537, 0.537),
        "igpt": (0.627, 0.352, 0.470),
        "dt": (0.733, 0.737, 0.870),
    }
    labels = {"gpt2": "GPT2", "igpt": "iGPT", "dt": "Random Init"}

    for model_name in model_names:
        cka_diag = np.diag(
            np.load(
                f"{path_to_cka}/cka_0_{epoch}_{model_name}{model_name}_{env_name}_medium_{seed}_{rtg_state_action}.npy"
            )
        )
        if model_name == "igpt":
            cka_diag = smooth_sequence(cka_diag)

        plt.plot(cka_diag, color=colors[model_name], label=labels[model_name])
        plt.scatter(np.arange(len(cka_diag)), cka_diag, color=colors[model_name])

        if model_name == "dt" and env_name == "hopper" and rtg_state_action == "state":
            plt.hlines(
                y=cka_threshold_rand_init,
                xmin=-5,
                xmax=110,
                color="gray",
                linestyles="dashed",
                alpha=0.5,
            )
            module_name_large_cka = get_large_cka_module_name(
                cka_diag, cka_threshold_rand_init
            )
            print(f"Large CKA modules: \n {module_name_large_cka}")

    plt.ylim(0, 1)
    plt.xlim(-5, 110)

    plt.xlabel("Normalized Layer")
    plt.ylabel("CKA")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(
        f"{path_to_save_figure}/cka_plot_{epoch}_gpt2_igpt_dt_{env_name}_medium_{seed}_{rtg_state_action}.pdf"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_load_cka", "-plc", type=str, default="results")
    parser.add_argument("--path_to_save_figure", "-psf", type=str, default="figs")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument(
        "--epoch",
        type=int,
        default=40,
        help="CKA similarity b/w the model at epoch 0 and that at this epoch will be plotted.",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="hopper",
        help="hopper, halfcheetah, or walker2d",
    )
    parser.add_argument(
        "--rtg_state_action",
        "-rsa",
        type=str,
        default="state",
        help="reward, state, or action",
    )
    parser.add_argument(
        "--cka_threshold_rand_init",
        "-thr",
        type=float,
        default=0.23,
        help="Plot horizontal line at this value.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    main(vars(args))
