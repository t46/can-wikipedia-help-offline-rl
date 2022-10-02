
# On the Effect of Pre-training for Transformer in Different Modality on Offline Reinforcement Learning 
## Ovreall
Repository to replecate the results of *On the Effect of Pre-training for Transformer in Different Modality on Offline Reinforcement Learning*. Our code is based on the followng repositories:
- https://github.com/machelreid/can-wikipedia-help-offline-rl
  - https://github.com/rail-berkeley/d4rl
- https://github.com/gtegner/mine-pytorch
- https://github.com/google-research/google-research/tree/master/representation_similarity

The setup for our experiments are based on that of the following repositories:
- https://github.com/machelreid/can-wikipedia-help-offline-rl
- https://github.com/gtegner/mine-pytorch

Our experiment and analysis are mainly done in `can-wikipedia-help-offline-rl` subdirectory. Thus, if we do not specify which directory we are in below, it assumes that we are in `can-wikipedia-help-offline-rl`. The directory `mine-pytorch` is only used for mutual information estimation.

We provide conda environments `mine` for mutual information calculation  by MINE in `mine-pytorch` and `wikirl-gym` for the others in `can-wikipedia-help-offline-rl`.

## Setup for `can-wikipedia-help-offline-rl`

### Install MuJoCo
D4RL depends on `mujoco-py` which requires `mujoco210`. Thus, install `mujoco210` following the instruction of [the repository](https://github.com/openai/mujoco-py). We excerpted the instruction from the repository below:
> Download the MuJoCo version 2.1 binaries for [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) or [OSX](https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz).
>
> Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`.
>
> If you want to specify a nonstandard location for the package, use the env variable `MUJOCO_PY_MUJOCO_PATH`.

### Create and activate `wikirl-gym` environment

```
cd can-wikipedia-help-offline-rl
conda env create -f conda_env.yml
conda activate wikirl-gym
```

### Install D4RL
```
cd data
git clone https://github.com/rail-berkeley/d4rl.git
cd d4rl
```
Edit `setupy.py` to use D4RL with mujoco 2.1.0 (D4RL currently depends on `dm-control` which requires `mujoco-2.1.1` and `mujoco-py` which requires `mujoco210`. See [reference](https://github.com/Farama-Foundation/D4RL/issues/144).). Specifically, change `'dm_control>=1.0.3'` in `setupy.py` to `'dm_control @ git+https://github.com/deepmind/dm_control@644d9e0047f68b35a6f8b79e5e8493e2910563af'`.

After that, run the following:
```
pip install -e .
```

### Download data of D4RL
```
python download_d4rl_datasets.py
```

### Activate Wandb
```
wandb login
```

## Setup for `mine-pytorch`
### Create and activate `mine` environment
```
cd mine-pytorch
conda env create -f conda_env.yml
conda activate mine
```


## Run Fine-Tuning

We run the following command to fine-tune the *randomly initialized model* used in Section 5.1 - 5.4 (context K=20). 
```{sh}
python experiment.py --env hopper --dataset medium --model_type dt --seed 666 --outdir "checkpoints/dt_medium_hopper_666" --dropout 0.2 --share_input_output_proj --warmup_steps 5000 --embed_dim 768 --n_layer 12 -w
```
For pre-trained models, add `--pretrained_lm gpt2` for *language-pre-trained model (GPT2)* and `--pretrained_lm openai/imagegpt-small` for *image-pre-trained model (iGPT)*. Running command above outputs per epoch i) fine-tuned models under `./checkpoints` directory, i.e. `checkpoints/dt_medium_hopper_666/model_40.pt` and ii) results such as mean return and action error to Weights and Biases. For sanity check in Appendix. A, run `sanity_check_preformance.ipynb` after running the above command.

For output the results of Section 5.5 and 5.6.2 (context K=1), just add `--K 1` option and change `--outdir` to `"checkpoints/dt_medium_hopper_666_K1"`. To get the results of block replacement experiment, run the following command.
```
python block_replacement_experiment.py --env hopper --dataset medium --model_type dt --seed 666 --outdir "checkpoints/block1_dt_medium_hopper_666_K1" --dropout 0.2 --share_input_output_proj --warmup_steps 5000 --embed_dim 768 --n_layer 12 -w --pretrained_lm gpt2 --K 1 --pretrained_block 1 --max_iters 10
```

Although other options are in `experiment.py`, we do not use these options.

## Analysis
### 5.1 Similarity Analysis
The notebooks for activation similarity analisis are in `notebooks/section-51-activation-similarity`.
- `compute-cka.ipynb`
  - Compute CKA between activations of two models, which output the Figure.2 and the CKA values used to plot Figure.1.
- `plot_cka.ipynb`
  - Plot Figure.1 from the CKA values save in `compute-cka.ipynb`.
### 5.2 Mutual Information
Code to run MINE is in `root/mine-pytorch` and notebooks for activation similarity analisis are in `notebooks/section-52-mutual-information`

- `mutual_information.ipynb`
  - Plot Figure 3 and 16 from estimated mutual information by `root/mine-pytorch/run_mi.py` or `root/mine-pytorch/run_mi_no_context.py`
- `save_activation.ipynb`
  - Save hidden representation to estimate mutual information in `root/mine-pytorch/run_mi.py` or `root/mine-pytorch/run_mi_no_context.py`.

The steps for mutual information estimation are following:
1. run `save_activation.ipynb` and save activation into the directory from which `run_mi.py` read activation.
2. run the code below
    ```
    cd ../mine-pytorch
    python run_mi.py
    cd ../can-wikipedia-help-offline-rl
    ```
    For the result of Appendix E.3, run `run_mi_no_context.py` instead of `run_mi.py`.
3. run `mutual_information.ipynb`
### 5.3 Parameter Similarity
The notebook for activation similarity analisis is in `notebooks/section-53-parameter-similarity`.
- `parameter_similarity_analysis.ipynb`
  - Compute parameter similarity and plot Figures 4 and 5.
### 5.4 Gradient Analysis
The notebooks for activation similarity analisis are in `notebooks/section-54-gradient-analysis`.
- `grad_confusion.ipynb`
  - Plot Figure 6.
- `grad_norm.ipynb`
  - Plot Figures 7 and 8.
### 5.5 Fine-Tuning With No Context Information
The notebook for activation plot the results of fine-tuning with no context is in `notebooks/section-55-fine-tuning-no-context`.
- `plot_learning_curve_no_context.ipynb`
  - Plot Figures 9 and 10
    - Note that Figure 10 is created by this notebook though Figure 10 is in Section 5.6.
  - Obtain the result to create Table 1.
### 5.6 More In-Depth Analysis of Context Dependence
The notebook for attention distance analisis is in `notebooks/section-56-dependence-on-context`.
- `attention_distance.ipynb`
  - Plot Figure 11.
## Analysis done during rebuttal period
The following two analyses are done during rebuttal period, the result of which are in Appendix.

### G.3 Analysis of the Effect of Gradient Clipping
Run the experiment with `--remove-grad-clip` option.
```{sh}
python experiment.py --env hopper --dataset medium --model_type igpt --seed 666 --outdir "checkpoints/igpt_medium_hopper_666_no_grd_clip" --dropout 0.2 --share_input_output_proj --warmup_steps 5000 -w
```
After that, plot the figure in the jupyter notebook. The notebook for activation similarity analisis is in `notebooks/section-54-gradient-analysis`.
- `plot_learning_curve_grad_clip.ipynb`
  - Plot Figures 19 and 20.

### H.2 Analysis of Why Randomly Initialized Model Fails for Hopper with No Context
The notebook for comparing mutual information of different data type is in `notebooks/section-52-mutual-information`.

- `mutual_information_data.ipynb`
  - Plot Figure 22.

1. run the code below
    ```
    cd ../mine-pytorch
    conda activate mine
    python run_mi_data.py
    conda deactivate mine
    cd ../can-wikipedia-help-offline-rl
    ```
2. run `mutual_information_data.ipynb`

## License

MIT
