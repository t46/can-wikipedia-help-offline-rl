def generate_variant(epoch, path_to_model_checkpoint, model_name, env_name, seed, dataset_name, batch_size=100):
    if model_name == 'gpt2':
        pretrained_lm = 'gpt2'
    elif model_name == 'clip':
        pretrained_lm = 'openai/clip-vit-base-patch32'
    elif model_name == 'igpt':
        pretrained_lm = 'openai/imagegpt-small'
    elif model_name == 'dt':
        pretrained_lm = False

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
        'K': 20, # 2
        'pct_traj': 1.0,
        'batch_size': batch_size,  # 64
        'num_eval_episodes': 100,
        'max_iters': 40,
        'num_steps_per_iter': 2500,
        'pretrained_lm': pretrained_lm,
        'gpt_kmeans': None,
        'kmeans_cache': None,
        'frozen': False,
        'extend_positions': False,
        'share_input_output_proj': True
    }

    return variant