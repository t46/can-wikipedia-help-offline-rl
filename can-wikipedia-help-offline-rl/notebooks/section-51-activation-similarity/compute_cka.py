import numpy as np
import torch
import sys
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

sys.path.append('../')
from sample_batch_data import get_data_info, get_batch
from signal_propagation import get_activation

# The code below is from https://github.com/google-research/google-research/tree/master/representation_similarity

def gram_linear(x):
    """Compute Gram (kernel) matrix for a linear kernel.

    Args:
      x: A num_examples x num_features matrix of features.

    Returns:
      A num_examples x num_examples Gram matrix of examples.
    """
    return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
    """Compute Gram (kernel) matrix for an RBF kernel.

    Args:
      x: A num_examples x num_features matrix of features.
      threshold: Fraction of median Euclidean distance to use as RBF kernel
        bandwidth. (This is the heuristic we use in the paper. There are other
        possible ways to set the bandwidth; we didn't try them.)

    Returns:
      A num_examples x num_examples Gram matrix of examples.
    """
    dot_products = x.dot(x.T)
    sq_norms = np.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = np.median(sq_distances)
    return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
    """Center a symmetric Gram matrix.

    This is equvialent to centering the (possibly infinite-dimensional) features
    induced by the kernel before computing the Gram matrix.

    Args:
      gram: A num_examples x num_examples symmetric matrix.
      unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
        estimate of HSIC. Note that this estimator may be negative.

    Returns:
      A symmetric matrix with centered columns and rows.
    """
    if not np.allclose(gram, gram.T):
      raise ValueError('Input must be a symmetric matrix.')
    gram = gram.copy()

    if unbiased:
      # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
      # L. (2014). Partial distance correlation with methods for dissimilarities.
      # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
      # stable than the alternative from Song et al. (2007).
      n = gram.shape[0]
      np.fill_diagonal(gram, 0)
      means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
      means -= np.sum(means) / (2 * (n - 1))
      gram -= means[:, None]
      gram -= means[None, :]
      np.fill_diagonal(gram, 0)
    else:
      means = np.mean(gram, 0, dtype=np.float64)
      means -= np.mean(means) / 2
      gram -= means[:, None]
      gram -= means[None, :]

    return gram


def cka(gram_x, gram_y, debiased=False):
    """Compute CKA.

    Args:
      gram_x: A num_examples x num_examples Gram matrix.
      gram_y: A num_examples x num_examples Gram matrix.
      debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
      The value of CKA between X and Y.
    """
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y + 1e-6)


def _debiased_dot_product_similarity_helper(
    xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
    n):
  """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
  # This formula can be derived by manipulating the unbiased estimator from
  # Song et al. (2007).
  return (
      xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
      + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))


def feature_space_linear_cka(features_x, features_y, debiased=False):
    """Compute CKA with a linear kernel, in feature space.

    This is typically faster than computing the Gram matrix when there are fewer
    features than examples.

    Args:
      features_x: A num_examples x num_features matrix of features.
      features_y: A num_examples x num_features matrix of features.
      debiased: Use unbiased estimator of dot product similarity. CKA may still be
        biased. Note that this estimator may be negative.

    Returns:
      The value of CKA between X and Y.
    """
    features_x = features_x - np.mean(features_x, 0, keepdims=True)
    features_y = features_y - np.mean(features_y, 0, keepdims=True)

    dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
    normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    normalization_y = np.linalg.norm(features_y.T.dot(features_y))

    if debiased:
      n = features_x.shape[0]
      # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
      sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
      sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
      squared_norm_x = np.sum(sum_squared_rows_x)
      squared_norm_y = np.sum(sum_squared_rows_y)

      dot_product_similarity = _debiased_dot_product_similarity_helper(
          dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
          squared_norm_x, squared_norm_y, n)
      normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
          normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
          squared_norm_x, squared_norm_x, n))
      normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
          normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
          squared_norm_y, squared_norm_y, n))

    return dot_product_similarity / (normalization_x * normalization_y)

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

    ax = sns.heatmap(cka_matrix, vmin=0, vmax=1, square=True)  # , cmap='bone'
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

    for env_name in env_name_list:
        
        torch.manual_seed(seed)

        dataset_name = 'medium'

        if model1 == 'gpt2':
            pretrained_lm1 = 'gpt2'
        elif model1 == 'clip':
            pretrained_lm1 = 'openai/clip-vit-base-patch32'
        elif model1 == 'igpt':
            pretrained_lm1 = 'openai/imagegpt-small'
        elif model1 == 'dt':
            pretrained_lm1 = False

        variant = {
            'embed_dim': 768,
            'n_layer': 12,
            'n_head': 1,
            'activation_function': 'relu',
            'dropout': 0.2, # 0.1
            'load_checkpoint': False if epoch1==0 else f'{path_to_model_checkpoint}/{model1}_medium_{env_name}_{seed}/model_{epoch1}.pt',
            'seed': seed,
            'outdir': f"tmp/{model1}_{dataset_name}_{env_name}_{seed}",
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

        if no_context:
            variant['load_checkpoint'] = False if epoch1==0 else f'{path_to_model_checkpoint}/{model1}_medium_{env_name}_{seed}_K1/model_{epoch1}.pt'

        device = torch.device(device)

        state_dim, act_dim, max_ep_len, scale = get_data_info(variant)
        states, actions, rewards, dones, rtg, timesteps, attention_mask = get_batch(variant, state_dim, act_dim, max_ep_len, scale, device, path_to_dataset)

        activation_list = []

        for _ in range(2):
    
            activation = get_activation(variant, state_dim, act_dim, max_ep_len, states, actions, rewards, rtg, timesteps, attention_mask, device)
            activation_list.append(activation)

            if model2 == 'gpt2':
                pretrained_lm2 = 'gpt2'
            elif model2 == 'clip':
                pretrained_lm2 = 'openai/clip-vit-base-patch32'
            elif model2 == 'igpt':
                pretrained_lm2 = 'openai/imagegpt-small'
            elif model2 == 'dt':
                pretrained_lm2 = False
            
            variant['outdir'] =  f"tmp/{model2}_{dataset_name}_{env_name}_{seed}"
            variant['pretrained_lm'] = pretrained_lm2

            if no_context:
                variant['load_checkpoint'] = False if epoch2==0 else f'{path_to_model_checkpoint}/{model2}_medium_{env_name}_{seed}_K1/model_{epoch2}.pt'
            else:
                variant['load_checkpoint'] = False if epoch2==0 else f'{path_to_model_checkpoint}/{model2}_medium_{env_name}_{seed}/model_{epoch2}.pt'

        reward_state_action_list = ['action', 'state', 'reward']

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
    cka_matrix = run_cka(
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