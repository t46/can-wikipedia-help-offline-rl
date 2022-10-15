"""
Plot mutual information between data.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("ticks")
sns.set_context("paper", 1.5, {"lines.linewidth": 2})


def main(args):
    env_names = ["hopper", "halfcheetah", "walker2d"]
    dataset_name = args["dataset_name"]
    seed = args["seed"]
    path_to_saved_mi = args["path_to_load_mi"]
    path_to_save_figure = args["path_to_save_figure"]

    state_rtg = ["state", "rtg"]

    os.makedirs(path_to_saved_mi, exist_ok=True)
    os.makedirs(path_to_save_figure, exist_ok=True)

    mi_action_dict = {}
    for sr in state_rtg:

        mi_action_list = []
        for env_name in env_names:

            mi_action = np.load(
                f"{path_to_saved_mi}/mi_data_{sr}_action_{env_name}_{dataset_name}_{seed}.npy",
                allow_pickle=True,
            )
            mi_action_np = [float(mi.cpu().numpy()) for mi in mi_action]
            mi_action_list.append(mi_action_np)

        mi_action_dict[sr] = mi_action_list

    y_labels = {"state": r"$\hat{I}(S; T)$", "rtg": r"$\hat{I}(\hat{R}; T)$"}
    for data_type in state_rtg:
        df = pd.DataFrame(
            {
                "Hopper": mi_action_dict[data_type][0],
                "Halfcheetah": mi_action_dict[data_type][1],
                "Walker2d": mi_action_dict[data_type][2],
            }
        )
        sns.boxplot(
            data=df,
            palette={
                "Hopper": (0.372, 0.537, 0.537),
                "Halfcheetah": (0.733, 0.737, 0.870),
                "Walker2d": (0.627, 0.352, 0.470),
            },
        )
        plt.ylabel(y_labels[data_type])
        plt.tight_layout()
        plt.savefig(
            f"{path_to_save_figure}/mi_data_{data_type}_action_{dataset_name}_{seed}.pdf"
        )
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_load_mi", "-plm", type=str, default="results")
    parser.add_argument("--path_to_save_figure", "-psf", type=str, default="figs")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="medium",
        help="Only medium is used for the experiments.",
    )  # medium, medium-replay, medium-expert, expert
    args = parser.parse_args()
    main(vars(args))
