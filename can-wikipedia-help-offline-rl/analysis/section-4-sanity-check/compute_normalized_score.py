'''
Sanity Check
'''

import wandb
import argparse
from pprint import pprint

import gym
import d4rl
import pickle
import numpy as np


def main(args):

    model_name = args["model_name"]
    env_name = args["env_name"]
    dataset_name = args["dataset_name"]
    seed = args["seed"]
    wandb_project_name = args["wandb_project_name"]

    # Get Run
    api = wandb.Api()
    runs = api.runs(
        path=wandb_project_name,
        filters={'state': 'finished'}
        )

    for run in runs:
        if run.name == f'gym-experiment-{env_name}-{dataset_name}-{model_name}-{seed}':
            break

    if env_name == 'hopper':
        rtg_conditioning = 3600
    elif env_name == 'halfcheetah':
        rtg_conditioning = 6000
    elif env_name == 'walker2d':
        rtg_conditioning = 5000
    else:
        rtg_conditioning = None

    return_map = {}
    return_map['medium'] = max(run.history()[f'evaluation/target_{rtg_conditioning}_return_mean'])
    best_checkpoint_epoch = np.argmax(run.history()[f'evaluation/target_{rtg_conditioning}_return_mean']) + 1

    # Get Return
    datasets = ["random", "expert"]
    path_to_dataset = args["path_to_load_data"]

    for dataset in datasets:

        if env_name == "hopper":
            env = gym.make("Hopper-v3")
            max_ep_len = 1000
            env_targets = [3600, 1800]  # evaluation conditioning targets
            scale = 1000.0  # normalization for rewards/returns
        elif env_name == "halfcheetah":
            env = gym.make("HalfCheetah-v3")
            max_ep_len = 1000
            env_targets = [12000, 6000]
            scale = 1000.0
        elif env_name == "walker2d":
            env = gym.make("Walker2d-v3")
            max_ep_len = 1000
            env_targets = [5000, 2500]
            scale = 1000.0
        elif env_name == "reacher2d":
            from decision_transformer.envs.reacher_2d import Reacher2dEnv

            env = Reacher2dEnv()
            max_ep_len = 100
            env_targets = [76, 40]
            scale = 10.0
        else:
            raise NotImplementedError

        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        # load dataset
        dataset_path = f"{path_to_dataset}/{env_name}-{dataset}-v2.pkl"
        with open(dataset_path, "rb") as f:
            trajectories = pickle.load(f)

        # save all path information into separate lists
        mode = "normal"
        states, traj_lens, returns = [], [], []
        for path in trajectories:
            if mode == "delayed":  # delayed: all rewards moved to end of trajectory
                path["rewards"][-1] = path["rewards"].sum()
                path["rewards"][:-1] = 0.0
            states.append(path["observations"])
            traj_lens.append(len(path["observations"]))
            returns.append(path["rewards"].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # used for input normalization
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        num_timesteps = sum(traj_lens)

        print("=" * 50)
        print(f"Starting new experiment: {env_name} {dataset}")
        print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
        print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
        print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
        print("=" * 50)

        if dataset == 'random':
            return_map['random'] = np.mean(returns)
        elif dataset == 'expert':
            return_map['expert'] = np.mean(returns)

    pprint(f'{model_name}-{env_name}-{seed}')
    pprint(return_map)
    normalized_score = 100 * (return_map['medium'] - return_map['random']) / (return_map['expert'] - return_map['random'])
    print(f'Epoch: {best_checkpoint_epoch}')
    print(f'Normalized Score: {normalized_score}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_load_data", "-pld", type=str, default="../../data")
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
    args = parser.parse_args()
    main(vars(args))
