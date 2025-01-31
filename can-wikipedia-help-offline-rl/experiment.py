"""
Run fine-tuning.
"""
import argparse
import os
import pickle
import random

import gym
import numpy as np
import torch

import wandb
from decision_transformer.evaluation.evaluate_episodes import (
    evaluate_episode,
    evaluate_episode_rtg,
)
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer
from utils import get_optimizer


def discount_cumsum(x, gamma):
    """Discount commulative summation of rewards.

    Args:
        x (np.ndarray): Trajectory of rewards.
        gamma (float): Discount factor.

    Returns:
       np.ndarray: Discounted cummulative summation of rewards.
    """
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def experiment(
    exp_prefix,
    variant,
):
    """Run fine-tuning.

    Args:
        exp_prefix (str): Prefix of wandb's run.
        variant (dict): Arguments.

    Raises:
        NotImplementedError: Environment is not implemented.
        NotImplementedError: Model type is not implemented.

    """
    assert (variant["pretrained_lm"] == "gpt2") or (
        variant["pretrained_block"] is None
    )
    # NOTE: (variant["pretrained_lm"] == "gpt2" AND variant["pretrained_block"] is not None) is not acceptable.
    # NOTE: When variant["pretrained_lm"] == False,
    # NOTE: a Transformer block of rand-init model is just replaced by that of another rand-init model.
    # NOTE: When variant["pretrained_lm"] == openai/imagegpt-small, the number of layers can be different b/w
    # NOTE: rand-init model and image-pre-trained model.

    seed = variant["seed"]
    torch.manual_seed(seed)
    device = variant.get("device", "cuda")
    log_to_wandb = variant.get("log_to_wandb", False)

    env_name, dataset = variant["env"], variant["dataset"]
    model_type = variant["model_type"]
    K = variant["K"]
    pretrained_block = variant["pretrained_block"]
    group_name = f"{exp_prefix}-{env_name}-{dataset}"

    if (variant["pretrained_lm"] is None) or (pretrained_block is not None):
        model_name = "dt"
    elif variant["pretrained_lm"] == "openai/imagegpt-small":
        model_name = "igpt"
    else:
        model_name = variant["pretrained_lm"]
    exp_name = f"{group_name}-{model_name}-{seed}"
    out_dir = variant["outdir"] + f"/{model_name}_{dataset}_{env_name}_{seed}"

    if K != 20:
        exp_name += f"-K{K}"
        out_dir += f"_K{K}"
    if variant["remove_grad_clip"]:
        exp_name += "-no-grad-clip"
        out_dir += "_no_grad_clip"
    if pretrained_block is not None:
        exp_name += f"-block{pretrained_block}"
        out_dir += f"_block{pretrained_block}"

    os.makedirs(out_dir, exist_ok=True)

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

    if model_type == "bc":
        env_targets = env_targets[
            :1
        ]  # since BC ignores target, no need for different evaluations

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    data_path = variant["data_path"]
    dataset_path = f"{data_path}/{env_name}-{dataset}-v2.pkl"
    with open(dataset_path, "rb") as f:
        trajectories = pickle.load(f)

    # save all path information into separate lists
    mode = variant.get("mode", "normal")
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

    batch_size = variant["batch_size"]
    num_eval_episodes = variant["num_eval_episodes"]
    pct_traj = variant.get("pct_traj", 1.0)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        """Get a batch of sample data from D4RL.

        Args:
            batch_size (int, optional): Batch size. Defaults to 256.
            max_len (_type_, optional): Maximum length of trajectories. Defaults to K.

        Returns:
            tuple: Batch of states, actions, rewards, done, return-to-go, timesteps, masks
        """
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj["rewards"].shape[0] - 1)

            # get sequences from dataset
            s.append(traj["observations"][si : si + max_len].reshape(1, -1, state_dim))
            a.append(traj["actions"][si : si + max_len].reshape(1, -1, act_dim))
            r.append(traj["rewards"][si : si + max_len].reshape(1, -1, 1))
            if "terminals" in traj:
                d.append(traj["terminals"][si : si + max_len].reshape(1, -1))
            else:
                d.append(traj["dones"][si : si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = (
                max_ep_len - 1
            )  # padding cutoff
            rtg.append(
                discount_cumsum(traj["rewards"][si:], gamma=1.0)[
                    : s[-1].shape[1] + 1
                ].reshape(1, -1, 1)
            )
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1
            )
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate(
                [np.ones((1, max_len - tlen, act_dim)) * -10.0, a[-1]], axis=1
            )
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = (
                np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1)
                / scale
            )
            timesteps[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1
            )
            mask.append(
                np.concatenate(
                    [np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1
                )
            )

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(
            dtype=torch.float32, device=device
        )
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(
            dtype=torch.float32, device=device
        )
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(
            dtype=torch.float32, device=device
        )
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(
            dtype=torch.long, device=device
        )
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(
            dtype=torch.float32, device=device
        )
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
            dtype=torch.long, device=device
        )
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        """Evaluate agent's performance for episodes.

        Args:
            target_rew (_type_): Evaluation conditioning target
        """

        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == "dt":
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew / scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew / scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
            return {
                f"target_{target_rew}_return_mean": np.mean(returns),
                f"target_{target_rew}_return_std": np.std(returns),
                f"target_{target_rew}_length_mean": np.mean(lengths),
                f"target_{target_rew}_length_std": np.std(lengths),
            }

        return fn

    # Wehn pretrained_block is not None, first instantiate the pre-trained model as `model_pretrained` and then
    # instantiate the randomly initialized model as `model`.
    # After that, a transformer block of `model` is replaced by that of `model_pretrained`.
    if pretrained_block:
        if model_type == "dt":
            model_pretrained = DecisionTransformer(
                args=variant,
                state_dim=state_dim,
                act_dim=act_dim,
                max_length=K,
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
                state_dict = torch.load(variant["load_checkpoint"])
                model_pretrained.load_state_dict(state_dict)
                print(f"Loaded from {variant['load_checkpoint']}")
        elif model_type == "bc":
            model_pretrained = MLPBCModel(
                state_dim=state_dim,
                act_dim=act_dim,
                max_length=K,
                hidden_size=variant["embed_dim"],
                n_layer=variant["n_layer"],
            )
        else:
            raise NotImplementedError

        # To instantiate the randomly initialized model, turn off `pretrained_lm` option.
        variant["pretrained_lm"] = False

    if model_type == "dt":
        model = DecisionTransformer(
            args=variant,
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
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
            state_dict = torch.load(variant["load_checkpoint"])
            model.load_state_dict(state_dict)
            print(f"Loaded from {variant['load_checkpoint']}")
    elif model_type == "bc":
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
        )
    else:
        raise NotImplementedError

    if pretrained_block:
        for i in range(
            12
        ):  # `model` (dt) and `model_pretrained` (gpt2) have 12 Transformer blocks.
            if i == pretrained_block:
                model.transformer.h[i] = model_pretrained.transformer.h[i]
                # model.transformer.h[i] is the i-th Transformer block of `model`.

    model = model.to(device=device)

    warmup_steps = variant["warmup_steps"]
    optimizer = get_optimizer(args=variant, model=model)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    if model_type == "dt":
        trainer = SequenceTrainer(
            args=variant,
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type == "bc":
        trainer = ActTrainer(
            args=variant,
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    if log_to_wandb:
        wandb.init(
            name=exp_name,
            group=group_name,
            project="decision-transformer",
            config=variant,
        )

    for iter in range(variant["max_iters"]):
        print("HI!")
        outputs = trainer.train_iteration(
            num_steps=variant["num_steps_per_iter"], iter_num=iter + 1, print_logs=True
        )
        print("HI2!")
        if log_to_wandb:
            wandb.log(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, default="hopper", help="hopper, halfcheetah, or walker2d"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="medium",
        help="Only medium is used for the experiments.",
    )  # medium, medium-replay, medium-expert, expert
    parser.add_argument(
        "--mode", type=str, default="normal"
    )  # normal for standard setting, delayed for sparse
    parser.add_argument(
        "--K", type=int, default=20, help="--K 1 for no context experiments."
    )
    parser.add_argument("--pct_traj", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--model_type", type=str, default="dt"
    )  # dt for decision transformer, bc for behavior cloning
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument(
        "--lm_learning_rate",
        "-lmlr",
        type=float,
        default=None,
        help="We did not use this option.",
    )
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=5000)  # 10000

    parser.add_argument("--num_eval_episodes", type=int, default=100)
    parser.add_argument("--max_iters", type=int, default=40)
    parser.add_argument("--num_steps_per_iter", type=int, default=2500)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--pretrained_lm",
        "-plm",
        type=str,
        default=None,
        help="gpt2 or openai/imagegpt-small. The model without this option corresponds to randomly initialized model.",
    )
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--log_to_wandb", "-w", action="store_true", default=False)

    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--outdir", type=str, default="checkpoints")
    parser.add_argument(
        "--fp16", action="store_true", default=False, help="We did not use this option."
    )

    parser.add_argument(
        "--frozen",
        action="store_true",
        default=False,
        help="We did not use this option.",
    )
    parser.add_argument(
        "--gpt_kmeans", type=int, default=None, help="We did not use this option."
    )
    parser.add_argument(
        "--extend_positions",
        action="store_true",
        default=False,
        help="We did not use this option.",
    )
    parser.add_argument(
        "--gpt_kmeans_const",
        type=float,
        default=None,
        help="We did not use this option.",
    )
    parser.add_argument(
        "--kmeans_cache", type=str, default=None, help="We did not use this option."
    )

    parser.add_argument("--share_input_output_proj", action="store_true", default=False)
    parser.add_argument(
        "--kmeans_mean",
        action="store_true",
        default=False,
        help="We did not use this option.",
    )

    parser.add_argument(
        "--remove_grad_clip",
        "-rgc",
        action="store_true",
        default=False,
        help="This is only for G.3 Analysis of the Effect of Gradient Clipping.",
    )
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument(
        "--pretrained_block",
        "-pb",
        type=int,
        default=None,
        help="This is for block replacement experiment. The value ranges from 0 to 11. \
                              When using this option, use --pretrained_lm gpt2.",
    )

    args = parser.parse_args()

    experiment("gym-experiment", variant=vars(args))
