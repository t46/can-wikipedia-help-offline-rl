import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm
from util_cka import cka, gram_linear

sys.path.append('../')
from sample_batch_data import get_batch, get_data_info
from set_config import generate_variant
from signal_propagation import get_activation


def compute_cka(activation_1, activation_2, reward_state_action, timestep=-1):
    """Compute CKA for either return-to-go, state, or action.

    Args:
        activation_1 (np.ndarray[batchsize, represemtation_dim, time_step_in_context]): Neural activation vector.
        activation_2 (np.ndarray[batchsize, represemtation_dim, time_step_in_context]): Neural activation vector.
        reward_state_action (str): "reward" (return-to-go), "state", or "action".
        timestep (int, optional): Activation at this timestep is used for CKA computation. Defaults to -1.

    Returns:
        np.ndarray[]: scalar CKA
    """
    # Input is sequence of [..., return-to-go, state, action]
    if reward_state_action == 'reward':
        idx = timestep * 3
    elif reward_state_action == 'state':
        idx = timestep * 2
    elif reward_state_action == 'action':
        idx = timestep * 1
    else:
        print("Specify either 'reward', 'state', or 'action'.")

    if len(activation_1.shape) == 3:
        activation_1 = activation_1[:, :, idx]
    elif len(activation_1.shape) == 4:
        activation_1 = activation_1[:, :, idx, idx]
    if len(activation_2.shape) == 3:
        activation_2 = activation_2[:, :, idx]
    elif len(activation_2.shape) == 4:
        activation_2 = activation_2[:, :, idx, idx]

    cka_from_examples = cka(gram_linear(activation_1.numpy()), gram_linear(activation_2.numpy()), debiased=True)

    return cka_from_examples

def plot_cka(path_to_save_figure, cka_matrix, reward_state_action, model1, model2, env_name, dataset_name, seed, epoch1, epoch2, block):
    """Plot CKA heatmap.

    Args:
        path_to_save_figure (str): Path to save figure of CKA heatmap.
        cka_matrix (np.ndarray[dim, dim]): CKA heatmap of activation of two models.
        reward_state_action (str): "reward" (return-to-go), "state", or "action".
        model1 (str): 'gpt2', 'igpt', or 'dt'.
        model2 (str): 'gpt2', 'igpt', or 'dt'.
        env_name (str): 'hopper', 'halfcheetah', or 'walker2d'.
        dataset_name (str): 'medium'.
        seed (int): 666.
        epoch1 (int): 0 or 40.
        epoch2 (int): 0 or 40.
        block (bool): If the activation is that of Transformer block, set this True.
    """    
    
    sns.set_style("ticks")
    sns.set_context("paper", 1.5, {"lines.linewidth": 2})

    ax = sns.heatmap(cka_matrix, vmin=0, vmax=1, square=True)
    ax.invert_yaxis()
    if model1 == 'dt':
        label1 = 'random init'
    else:
        label1 = model1
    if model2 == 'dt':
        label2 = 'random init'
    else:
        label2 = model2
    if block:
        plt.xlabel(f'{label2.upper()} Block')
        plt.ylabel(f'{label1.upper()} Block')
    else:
        plt.xlabel(f'{label2.upper()} Layers')
        plt.ylabel(f'{label1.upper()} Layers')
    plt.tight_layout()
    if block:
        plt.savefig(f'{path_to_save_figure}/cka_block_{epoch1}_{epoch2}_{model1}{model2}_{env_name}_{dataset_name}_{seed}_{reward_state_action}.pdf')
    else:
        plt.savefig(f'{path_to_save_figure}/cka_{epoch1}_{epoch2}_{model1}{model2}_{env_name}_{dataset_name}_{seed}_{reward_state_action}.pdf')
    plt.show()


def run_cka(
    path_to_dataset,
    path_to_model_checkpoint,
    path_to_save_cka,
    path_to_save_figure,
    seed=666,
    model1='gpt2',
    model2='gpt2',
    epoch1=40,
    epoch2=40,
    env_name_list=['hopper', 'halfcheetah', 'walker2d'],
    block=False,
    no_context=False,
    device='cpu'
    ):
    """Compute CKA and save it as array and fig.

    Args:
        path_to_model_checkpoint (str): Path to load model checkpoint.
        path_to_save_cka (str): Path to save CKA matrix as np.array.
        path_to_save_figure (str): Path to save figure of CKA heatmap.
        seed (int, optional): Random seed. Defaults to 666.
        model1 (str, optional): 'gpt2', 'igpt', or 'dt'. Defaults to 'gpt2'.
        model2 (str, optional): 'gpt2', 'igpt', or 'dt'. Defaults to 'gpt2'.
        epoch1 (int, optional): 0 or 40. Defaults to 40.
        epoch2 (int, optional): 0 or 40. Defaults to 40.
        env_name_list (list, optional): environment name list. Defaults to ['hopper', 'halfcheetah', 'walker2d'].
        block (bool, optional): If True, compute CKA for transformer block. Defaults to False.
        no_context (bool, optional): If True, compute CKA of K=1. Defaults to False.
        device (str): cuda or cpu
    """    
    
    dataset_name = 'medium'
    device = torch.device(device)
    reward_state_action_list = ['action', 'state', 'reward']

    for env_name in env_name_list:
        
        torch.manual_seed(seed)

        variant = generate_variant(epoch1, path_to_model_checkpoint, model1, env_name, seed, dataset_name)

        if no_context:
            variant['load_checkpoint'] = False if epoch1==0 else f'{path_to_model_checkpoint}/{model1}_{dataset_name}_{env_name}_{seed}_K1/model_{epoch1}.pt'

        state_dim, act_dim, max_ep_len, scale = get_data_info(variant)
        states, actions, rewards, _, rtg, timesteps, attention_mask = get_batch(variant, state_dim, act_dim, max_ep_len, scale, device, path_to_dataset)

        activation_list = []

        for _ in range(2):
    
            activation = get_activation(variant, state_dim, act_dim, max_ep_len, states, actions, rewards, rtg, timesteps, attention_mask, device)
            activation_list.append(activation)

            variant = generate_variant(epoch2, path_to_model_checkpoint, model2, env_name, seed, dataset_name)

            if no_context:
                variant['load_checkpoint'] = False if epoch2==0 else f'{path_to_model_checkpoint}/{model2}_{dataset_name}_{env_name}_{seed}_K1/model_{epoch2}.pt'

        if block:
            for reward_state_action in reward_state_action_list:
                cka_matrix = []
                for key_1, act_1 in tqdm(activation_list[0].items()):
                    # Compute CKA only for output of blocks (e.g. DecisionTransformer.transformer.h[0].mlp.dropout)
                    if ('dropout' in key_1) and ('mlp' in key_1):
                        cka_list = []
                        for key_2, act_2 in activation_list[1].items():
                            if ('dropout' in key_2) and ('mlp' in key_2):
                                cka = compute_cka(act_1, act_2, reward_state_action)
                                cka_list.append(cka)
                        cka_matrix.append(cka_list)
                cka_matrix = np.array(cka_matrix)

                np.save(f'{path_to_save_cka}/cka_block_{epoch1}_{epoch2}_{model1}{model2}_{env_name}_{dataset_name}_{seed}_{reward_state_action}.npy', cka_matrix)
                plot_cka(path_to_save_figure, cka_matrix, reward_state_action, model1, model2, env_name, dataset_name, seed, epoch1, epoch2, block)
        else:
            for reward_state_action in reward_state_action_list:
                cka_matrix = []
                for key_1, act_1 in tqdm(activation_list[0].items()):
                    cka_list = []
                    for key_2, act_2 in activation_list[1].items():
                        cka = compute_cka(act_1, act_2, reward_state_action, timestep=-1)
                        cka_list.append(cka)
                    cka_matrix.append(cka_list)
                cka_matrix = np.array(cka_matrix)

                np.save(f'{path_to_save_cka}/cka_{epoch1}_{epoch2}_{model1}{model2}_{env_name}_{dataset_name}_{seed}_{reward_state_action}.npy', cka_matrix)
                plot_cka(path_to_save_figure, cka_matrix, reward_state_action, model1, model2, env_name, dataset_name, seed, epoch1, epoch2, block)


def main(args):
    _ = run_cka(
        args['path_to_load_data'],
        args['path_to_load_model'],
        args['path_to_save_cka'],
        args['path_to_save_figure'],
        seed=args['seed'],
        model1=args['model1'],
        model2=args['model2'],
        epoch1=args['epoch1'],
        epoch2=args['epoch2'],
        env_name_list=['hopper', 'halfcheetah', 'walker2d'],
        block=False,
        no_context=False,
        device=args['device']
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_load_data", type=str, default='../../data')
    parser.add_argument("--path_to_load_model", type=str, default='../../checkpoints')
    parser.add_argument("--path_to_save_cka", type=str, default='results')
    parser.add_argument("--path_to_save_figure", type=str, default='figs')
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--model1", type=str, default='gpt2')
    parser.add_argument("--model2", type=str, default='gpt2')
    parser.add_argument("--epoch1", type=int, default=40)
    parser.add_argument("--epoch2", type=int, default=40)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    main(vars(args))