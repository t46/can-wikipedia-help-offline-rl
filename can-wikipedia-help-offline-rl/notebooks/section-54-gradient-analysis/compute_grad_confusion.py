import numpy as np
import torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import argparse

sys.path.append('../')
from sample_batch_data import get_data_info, get_batch
from signal_propagation import get_gradients

sns.set_style("ticks")
sns.set_context("paper", 1.5, {"lines.linewidth": 2})

def main(args):
    model_names = ['gpt2', 'igpt', 'dt']
    seed=args['seed']
    epoch=args['epoch']
    dataset_name = args['dataset_name']
    env_name = args['env_name']
    path_to_dataset = args['path_to_load_data']
    path_to_model_checkpoint = args['path_to_load_model']
    path_to_save_grad_cossim = args['path_to_save_gradcossim']
    path_to_save_figure = args['path_to_save_figure']

    gradcossims_list = []
    min_gradcossims_list = []

    for model_name in tqdm(model_names):

        torch.manual_seed(seed)

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
            'load_checkpoint': False if epoch==0 else f'{path_to_model_checkpoint}/{model_name}_medium_{env_name}_{seed}/model_{epoch}.pt',
            'seed': seed,
            'outdir': f"tmp/{model_name}_{dataset_name}_{env_name}_{seed}",
            'env': env_name,
            'dataset': dataset_name,
            'model_type': 'dt',
            'K': 20,
            'pct_traj': 1.0,
            'batch_size': 50,
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

        device = torch.device(args["device"])

        state_dim, act_dim, max_ep_len, scale = get_data_info(variant)
        states, actions, rewards, dones, rtg, timesteps, attention_mask = get_batch(variant, state_dim, act_dim, max_ep_len, scale, device, path_to_dataset)
        grads_list = get_gradients(variant, state_dim, act_dim, max_ep_len, states, actions, rewards, rtg, timesteps, attention_mask, device)
        gradcossim_list = []
        for grads1 in tqdm(grads_list):
            for grads2 in grads_list:
                gradcossim_list.append((torch.dot(grads1, grads2) / (1e-6 + torch.norm(grads1) * torch.norm(grads2))).numpy())
        gradcossim = np.array(gradcossim_list)
        np.save(f'{path_to_save_grad_cossim}/gradcossim_{epoch}_{model_name}_{env_name}_{dataset_name}_{seed}.npy', gradcossim)

        gradcossims_list.append(gradcossim)

    np.save(f'{path_to_save_grad_cossim}/gradcossim_{epoch}_gpt2_igpt_dt_{env_name}_{dataset_name}_{seed}.npy', gradcossims_list)

    # We measure gradient confusion by the minimum gradient cosine similarity.
    min_gradcossims_list = []
    for model_name in model_names:
        min_gradcossim = np.min(np.load(f'{path_to_save_grad_cossim}/gradcossim_{epoch}_{model_name}_{env_name}_{dataset_name}_{seed}.npy'))
        min_gradcossims_list.append([min_gradcossim])

    model_name_label = ['GPT2', 'iGPT', 'Random Init']
    colors = [(0.372, 0.537, 0.537), (0.627, 0.352, 0.470), (0.733, 0.737, 0.870)]
    my_palette = sns.color_palette(colors)
    sns.barplot(data=min_gradcossims_list, palette=my_palette)
    plt.xticks(np.arange(3), model_name_label)
    plt.ylabel(r'Min of $cossim(\nabla_{\theta}\ell_1, \nabla_{\theta}\ell_2)$')
    plt.savefig(f'{path_to_save_figure}/mingradcossim_{epoch}_gpt2_igpt_dt_{env_name}_{dataset_name}_{seed}.pdf')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_load_data", type=str, default='../../data')
    parser.add_argument("--path_to_load_model", type=str, default='../../checkpoints')
    parser.add_argument("--path_to_save_gradcossim", type=str, default='results')
    parser.add_argument("--path_to_save_figure", type=str, default='figs')
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--env_name", type=str, default='hopper')
    parser.add_argument("--dataset_name", type=str, default="medium")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    main(vars(args))