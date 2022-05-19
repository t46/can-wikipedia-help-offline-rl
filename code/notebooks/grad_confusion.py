import gym
import numpy as np
import torch
import wandb

import argparse
import pickle
import random
import sys

sys.path.append('/root/projects/can-wikipedia-help-offline-rl/code')

from decision_transformer.evaluation.evaluate_episodes import (
    evaluate_episode,
    evaluate_episode_rtg,
)
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer

from utils import get_optimizer
import os

from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sns.set_style("ticks")
sns.set_context("paper", 1.5, {"lines.linewidth": 2})

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum

def prepare_data(variant):
    env_name, dataset = variant["env"], variant["dataset"]
    model_type = variant["model_type"]
    exp_prefix = 'gym-experiment'
    group_name = f"{exp_prefix}-{env_name}-{dataset}"
    exp_prefix = f"{group_name}-{random.randint(int(1e5), int(1e6) - 1)}"

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
    dataset_path = f"../data/{env_name}-{dataset}-v2.pkl"
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
    
    return trajectories, sorted_inds, state_dim, act_dim, max_ep_len, state_mean, state_std, num_trajectories, p_sample, scale

def get_batch(
    batch_size, 
    max_len,
    trajectories,
    sorted_inds,
    state_dim,
    act_dim,
    max_ep_len,
    state_mean,
    state_std,
    num_trajectories,
    p_sample,
    scale,
    device
    ):
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

def main():
    seed=666
    epoch=1
    env_name='hopper'
    reward_state_action = 'state'

    torch.manual_seed(seed)

    dataset_name = 'medium'

    model_names = ['gpt2', 'igpt', 'dt']  # ['gpt2', 'igpt', 'dt']
    grad_confusions_list = []

    for model_name in model_names:

        if model_name == 'gpt2':
            pretrained_lm1 = 'gpt2'
        elif model_name == 'clip':
            pretrained_lm1 = 'openai/clip-vit-base-patch32'
        elif model_name == 'igpt':
            pretrained_lm1 = 'openai/imagegpt-small'
        elif model_name == 'dt':
            pretrained_lm1 = False

        variant = {
            'embed_dim': 768,
            'n_layer': 12,
            'n_head': 1,
            'activation_function': 'relu',
            'dropout': 0.2, # 0.1
            'load_checkpoint': False if epoch==0 else f'../checkpoints/{model_name}_medium_{env_name}_666/model_{epoch}.pt',
            'seed': seed,
            'outdir': f"checkpoints/{model_name}_{dataset_name}_{env_name}_{seed}",
            'env': env_name,
            'dataset': dataset_name,
            'model_type': 'dt',
            'K': 20, # 2
            'pct_traj': 1.0,
            'batch_size': 100,  # 64
            'num_eval_episodes': 100,
            'max_iters': 40,
            'num_steps_per_iter': 2500,
            'pretrained_lm': pretrained_lm1,
            'gpt_kmeans': None,
            'kmeans_cache': None,
            'frozen': False,
            'extend_positions': False,
            'share_input_output_proj': True
        }

        os.makedirs(variant["outdir"], exist_ok=True)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda:0")

        trajectories, sorted_inds, state_dim, act_dim, max_ep_len, state_mean, state_std, num_trajectories, p_sample, scale = prepare_data(variant)

        K = variant["K"]
        batch_size = variant["batch_size"]

        loss_fn = lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2)

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
        ).to(device)
        if variant["load_checkpoint"]:
            # state_dict = torch.load(variant["load_checkpoint"], map_location=torch.device('cpu'))
            state_dict = torch.load(variant["load_checkpoint"])
            model.load_state_dict(state_dict)
            print(f"Loaded from {variant['load_checkpoint']}")

        # model.eval()

        # grad = {}
        # def get_grad(name):
        #     def hook(model, input, output):
        #         grad[name] = output.detach()
        #     return hook

        # for block_id in range(len(model.transformer.h)):
        #     model.transformer.h[block_id].ln_1.register_backward_hook(get_grad(f'{block_id}.ln_1'))
        #     model.transformer.h[block_id].attn.c_attn.register_backward_hook(get_grad(f'{block_id}.attn.c_attn'))
        #     model.transformer.h[block_id].attn.c_proj.register_backward_hook(get_grad(f'{block_id}.attn.c_proj'))
        #     model.transformer.h[block_id].attn.attn_dropout.register_backward_hook(get_grad(f'{block_id}.attn.attn_dropout'))
        #     model.transformer.h[block_id].attn.resid_dropout.register_backward_hook(get_grad(f'{block_id}.attn.resid_dropout'))
        #     model.transformer.h[block_id].ln_2.register_backward_hook(get_grad(f'{block_id}.ln_2'))
        #     model.transformer.h[block_id].mlp.c_fc.register_backward_hook(get_grad(f'{block_id}.mlp.c_fc'))
        #     model.transformer.h[block_id].mlp.c_proj.register_backward_hook(get_grad(f'{block_id}.mlp.c_proj'))
        #     model.transformer.h[block_id].mlp.act.register_backward_hook(get_grad(f'{block_id}.mlp.act'))
        #     model.transformer.h[block_id].mlp.dropout.register_backward_hook(get_grad(f'{block_id}.mlp.dropout'))
        states, actions, rewards, dones, rtg, timesteps, attention_mask = get_batch(batch_size, 
                                                                                    K,
                                                                                    trajectories,
                                                                                    sorted_inds,
                                                                                    state_dim,
                                                                                    act_dim,
                                                                                    max_ep_len,
                                                                                    state_mean,
                                                                                    state_std,
                                                                                    num_trajectories,
                                                                                    p_sample,
                                                                                    scale,
                                                                                    device
                                                                                    )
        action_target = torch.clone(actions)
        grads_list = []

        for batch_id in tqdm(range(batch_size)):
            ##### 勾配計算 #####
            action_target_batch = action_target[batch_id, :, :].unsqueeze(0).to(device)

            state_preds, action_preds, reward_preds, all_embs = model.forward(
                states[batch_id, :, :].unsqueeze(0),
                actions[batch_id, :, :].unsqueeze(0),
                rewards[batch_id, :, :].unsqueeze(0),
                rtg[batch_id, :-1].unsqueeze(0),
                timesteps[batch_id, :].unsqueeze(0),
                attention_mask=attention_mask[batch_id, :].unsqueeze(0),
            )

            act_dim = action_preds.shape[2]
            action_preds = action_preds.reshape(-1, act_dim)[attention_mask[batch_id, :].unsqueeze(0).reshape(-1) > 0]
            action_target_batch = action_target_batch.reshape(-1, act_dim)[
                attention_mask[batch_id, :].unsqueeze(0).reshape(-1) > 0
            ]

            model.zero_grad()
            loss = loss_fn(
                None,
                action_preds,
                None,
                None,
                action_target_batch,
                None,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), .25)

            grads = []
            for name, param in model.transformer.h.named_parameters():
                grads.append(param.grad.view(-1))
            grads = torch.cat(grads)

            grads_list.append(grads)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)  # grad normの閾値を敷く
            # grad_ordered = {}
            # for block_id in range(len(model.transformer.h)):
            #     for block_name in block_name_list:
            #         grad_ordered[f'{block_id}.{block_name}'] = grad[f'{block_id}.{block_name}']

        # np.save(f'results/grad_{epoch}_{model_name}_{env_name}_{dataset_name}_{seed}_{reward_state_action}.npy', grads_list)

        grad_confusion_list = []
        for grads1 in tqdm(grads_list):
            for grads2 in tqdm(grads_list):
                grad_confusion_list.append(torch.dot(grads1, grads2).numpy())
        grad_confusions = np.array(grad_confusion_list)

        np.save(f'results/gradconfusions_{epoch}_{model_name}_{env_name}_{dataset_name}_{seed}_{reward_state_action}.npy', grad_confusions)
        
        grad_confusions_list.append(grad_confusions)

        del model

    np.save(f'results/gradconfusions_{epoch}_gpt2_igpt_dt_{env_name}_{dataset_name}_{seed}_{reward_state_action}.npy', grad_confusions_list)

    model_name_label = ['GPT2', 'iGPT', 'Random Init']
    colors = [(0.372, 0.537, 0.537), (0.627, 0.352, 0.470), (0.733, 0.737, 0.870)]
    my_palette = sns.color_palette(colors)
    sns.boxplot(data=grad_confusions_list, palette=my_palette)  # "PuBuGn_r"
    plt.xticks(np.arange(3), model_name_label)
    plt.ylabel(r'$\nabla_{\theta}\ell_1 \dot\nabla_{\theta}\ell_2$')
    plt.title('Gradient Confusion')
    plt.savefig(f'figs/gradconfusions_{epoch}_gpt2_igpt_dt_{env_name}_{dataset_name}_{seed}_{reward_state_action}.pdf')
    # plt.show()

if __name__ == "__main__":
    main()