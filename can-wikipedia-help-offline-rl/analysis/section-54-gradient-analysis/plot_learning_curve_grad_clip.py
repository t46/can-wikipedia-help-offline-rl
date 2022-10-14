"""
Compare learning curves between w/ grad clip and w/o grad clip.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb

sns.set_style("ticks")
sns.set_context("paper", 1.5, {"lines.linewidth": 2})


def get_return_from_wandb(model_name, env_name, seed, remove_grad, wandb_project_name):
    """Get mean return from wandb

    Args:
        model_name (str): 'gpt2', 'igpt', or 'dt'.
        env_name (str): 'hopper', 'halfcheetah', or 'walker2d'.
        seed (int): Random seed used for experiments.
        remove_grad (bool): If True, return the results without grad clip.
        wandb_project_name (str): Your Wandb project name.

    Returns:
        list:  mean return over epocs for K=20
    """
    api = wandb.Api()
    runs = api.runs(path=wandb_project_name, filters={"state": "finished"})

    if env_name == "hopper":
        rtg_conditioning = 3600
    elif env_name == "halfcheetah":
        rtg_conditioning = 6000
    elif env_name == "walker2d":
        rtg_conditioning = 5000
    else:
        rtg_conditioning = None

    if remove_grad:
        for run in runs:
            if (
                run.name
                == f"gym-experiment-{env_name}-medium-{model_name}-{seed}-no-grad-clip"
            ):
                break
        return_mean = run.history()[
            f"evaluation/target_{rtg_conditioning}_return_mean"
        ][:10]
        return return_mean
    else:
        for run in runs:
            if run.name == f"gym-experiment-{env_name}-medium-{model_name}-{seed}":
                break
        return_mean = run.history()[
            f"evaluation/target_{rtg_conditioning}_return_mean"
        ][:10]
        return return_mean


def get_action_error_from_wandb(
    model_name, env_name, seed, remove_grad, wandb_project_name
):
    """Get action errors.

    Args:
        model_name (str): 'gpt2', 'igpt', or 'dt'.
        env_name (str): 'hopper', 'halfcheetah', or 'walker2d'.
        seed (int): Random seed used for experiments.
        remove_grad (bool): If True, return the results without grad clip.
        wandb_project_name (str): Your Wandb project name.

    Returns:
        list: action error over epocs for K=20
    """
    api = wandb.Api()
    runs = api.runs(path=wandb_project_name, filters={"state": "finished"})

    if remove_grad:
        for run in runs:
            if (
                run.name
                == f"gym-experiment-{env_name}-medium-{model_name}-{seed}-no-grad-clip"
            ):
                break
        action_error = run.history()["training/action_error"][:10]
        return action_error
    else:
        for run in runs:
            if run.name == f"gym-experiment-{env_name}-medium-{model_name}-{seed}":
                break
        action_error = run.history()["training/action_error"][:10]
        return action_error


def main(args):

    seed = args["seed"]
    env_name = args["env_name"]
    model_name = args["model_name"]
    dataset_name = args["dataset_name"]
    from_wandb = args["from_wandb"]
    path_to_return = args["path_to_save_return"]
    path_to_action_error = args["path_to_save_action_error"]
    path_to_save_figure = args["path_to_save_figure"]
    wandb_project_name = args["wandb_project_name"]

    if from_wandb:
        return_mean = get_return_from_wandb(
            model_name, env_name, seed, False, wandb_project_name
        )
        return_mean_no_grad_clip = get_return_from_wandb(
            model_name, env_name, seed, True, wandb_project_name
        )
        np.save(
            f"{path_to_return}/returnmean_{model_name}_{env_name}_{dataset_name}_{seed}.npy",
            return_mean,
        )
        np.save(
            f"{path_to_return}/returnmean_{model_name}_no_grad_clip_{env_name}_{dataset_name}_{seed}.npy",
            return_mean_no_grad_clip,
        )

        action_error = get_action_error_from_wandb(
            model_name, env_name, seed, False, wandb_project_name
        )
        action_error_no_grad_clip = get_action_error_from_wandb(
            model_name, env_name, seed, True, wandb_project_name
        )
        np.save(
            f"{path_to_action_error}/action_error_{model_name}_{env_name}_{dataset_name}_{seed}.npy",
            action_error,
        )
        np.save(
            f"{path_to_action_error}/action_error_{model_name}_no_grad_clip_{env_name}_{dataset_name}_{seed}.npy",
            action_error_no_grad_clip,
        )
    else:
        return_mean = np.load(
            f"{path_to_return}/returnmean_{model_name}_{env_name}_{dataset_name}_{seed}.npy"
        )
        return_mean_no_grad_clip = np.load(
            f"{path_to_return}/returnmean_{model_name}_no_grad_clip_{env_name}_{dataset_name}_{seed}.npy"
        )

        action_error = np.load(
            f"{path_to_action_error}/action_error_{model_name}_{env_name}_{dataset_name}_{seed}.npy"
        )
        action_error_no_grad_clip = np.load(
            f"{path_to_action_error}/action_error_{model_name}_no_grad_clip_{env_name}_{dataset_name}_{seed}.npy"
        )

    # Return Mean
    plt.plot(return_mean, color=(0.372, 0.537, 0.537), label="w/ Grad Clip")
    plt.scatter(np.arange(len(return_mean)), return_mean, color=(0.372, 0.537, 0.537))

    plt.plot(
        return_mean_no_grad_clip, color=(0.627, 0.352, 0.470), label="w/o Grad Clip"
    )
    plt.scatter(
        np.arange(len(return_mean_no_grad_clip)),
        return_mean_no_grad_clip,
        color=(0.627, 0.352, 0.470),
    )

    plt.legend(loc="upper right")
    plt.ylabel("Mean Return")
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(
        f"{path_to_save_figure}/returnmean_{model_name}_grad_clip_{env_name}_{dataset_name}_{seed}.pdf"
    )
    plt.close()

    # Action Error
    plt.plot(action_error, color=(0.372, 0.537, 0.537), label="w/ Grad Clip")
    plt.scatter(np.arange(len(action_error)), action_error, color=(0.372, 0.537, 0.537))

    plt.plot(
        action_error_no_grad_clip, color=(0.627, 0.352, 0.470), label="w/o Grad Clip"
    )
    plt.scatter(
        np.arange(len(action_error_no_grad_clip)),
        action_error_no_grad_clip,
        color=(0.627, 0.352, 0.470),
    )

    plt.legend(loc="upper right")
    plt.ylabel("Action Error")
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(
        f"{path_to_save_figure}/action_error_{model_name}_grad_clip_{env_name}_{dataset_name}_{seed}.pdf"
    )
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_save_return", "-psr", type=str, default="results")
    parser.add_argument(
        "--path_to_save_action_error", "-psa", type=str, default="results"
    )
    parser.add_argument("--path_to_save_figure", "-psf", type=str, default="figs")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument(
        "--env_name",
        type=str,
        default="hopper",
        help="hopper, halfcheetah, or walker2d",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="dt, gpt2, or igpt.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="medium",
        help="Only medium is used for the experiments.",
    )  # medium, medium-replay, medium-expert, expert
    parser.add_argument(
        "--from_wandb",
        action="store_true",
        default=True,
        help="Load data from wandb. If False, load from the directory data is saved.",
    )
    parser.add_argument(
        "--wandb_project_name",
        "-wpn",
        type=str,
        help="user_name/wandb_project_name",
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    main(vars(args))
