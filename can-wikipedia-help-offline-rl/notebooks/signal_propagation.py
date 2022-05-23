"""
Functions to get activation or gradients.
"""

import torch
from tqdm._tqdm_notebook import tqdm

import sys
sys.path.append('../../')
from decision_transformer.models.decision_transformer import DecisionTransformer

def get_activation(variant, state_dim, act_dim, max_ep_len, states, actions, rewards, rtg, timesteps, attention_mask):
    """Get activation of a model.

    Args:
        variant (dict): arguments.
        state_dim (int): dimension of state.
        act_dim (int): dimension of action.
        max_ep_len (int): context length K.
        states (torch.Tensor): a batch of states.
        actions (torch.Tensor): a batch of actions.
        rewards (torch.Tensor): a batch of rewards.
        rtg (torch.Tensor): a batch of return-to-go.
        timesteps (torch.Tensor): a batch of timesteps.
        attention_mask (torch.Tensor): Mask for causal Transformer.

    Returns:
        dict: {layer_name: activation, ...}
    """    
    torch.manual_seed(0)
    model = DecisionTransformer(
        args=variant,
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=variant["K"],
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
        state_dict = torch.load(variant["load_checkpoint"], map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print(f"Loaded from {variant['load_checkpoint']}")

    model.eval()

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    for block_id in range(len(model.transformer.h)):
        model.transformer.h[block_id].ln_1.register_forward_hook(get_activation(f'{block_id}.ln_1'))
        model.transformer.h[block_id].attn.c_attn.register_forward_hook(get_activation(f'{block_id}.attn.c_attn'))
        model.transformer.h[block_id].attn.c_proj.register_forward_hook(get_activation(f'{block_id}.attn.c_proj'))
        model.transformer.h[block_id].attn.attn_dropout.register_forward_hook(get_activation(f'{block_id}.attn.attn_dropout'))
        model.transformer.h[block_id].attn.resid_dropout.register_forward_hook(get_activation(f'{block_id}.attn.resid_dropout'))
        model.transformer.h[block_id].ln_2.register_forward_hook(get_activation(f'{block_id}.ln_2'))
        model.transformer.h[block_id].mlp.c_fc.register_forward_hook(get_activation(f'{block_id}.mlp.c_fc'))
        model.transformer.h[block_id].mlp.c_proj.register_forward_hook(get_activation(f'{block_id}.mlp.c_proj'))
        try:
            model.transformer.h[block_id].mlp.act.register_forward_hook(get_activation(f'{block_id}.mlp.act'))
        except:
            pass
        model.transformer.h[block_id].mlp.dropout.register_forward_hook(get_activation(f'{block_id}.mlp.dropout'))

    _, _, _, _ = model.forward(
        states,
        actions,
        rewards,
        rtg[:, :-1],
        timesteps,
        attention_mask=attention_mask,
    )

    activation_sorted = {}
    block_name_list = [
        'ln_1',
        'attn.c_attn',
        'attn.c_proj',
        'attn.resid_dropout',
        'ln_2',
        'mlp.c_fc',
        'mlp.c_proj',
        'mlp.act',
        'mlp.dropout'
    ]
    for block_id in range(len(model.transformer.h)):
        for block_name in block_name_list:
            try:
                activation_sorted[f'{block_id}.{block_name}'] = activation[f'{block_id}.{block_name}']
            except:
                pass

    return activation_sorted

def get_gradients(variant, state_dim, act_dim, max_ep_len, states, actions, rewards, rtg, timesteps, attention_mask):
    """Get gradients of a model.

    Args:
        variant (dict): arguments.
        state_dim (int): dimension of state.
        act_dim (int): dimension of action.
        max_ep_len (int): context length K.
        states (torch.Tensor): a batch of states.
        actions (torch.Tensor): a batch of actions.
        rewards (torch.Tensor): a batch of rewards.
        rtg (torch.Tensor): a batch of return-to-go.
        timesteps (torch.Tensor): a batch of timesteps.
        attention_mask (torch.Tensor): Mask for causal Transformer.

    Returns:
        list: gradients of different samples.
    """    

    loss_fn = lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2)

    model = DecisionTransformer(
        args=variant,
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=variant["K"],
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
        state_dict = torch.load(variant["load_checkpoint"], map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print(f"Loaded from {variant['load_checkpoint']}")

    action_target = torch.clone(actions)
    grads_list = []

    for batch_id in tqdm(range(variant["batch_size"])):
        ##### 勾配計算 #####
        action_target_batch = action_target[batch_id, :, :].unsqueeze(0)

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

    return grads_list



def get_gradients_grad_per_norm(variant, state_dim, act_dim, max_ep_len, states, actions, rewards, rtg, timesteps, attention_mask):
    """Get gradients of a model.

    Args:
        variant (dict): arguments.
        state_dim (int): dimension of state.
        act_dim (int): dimension of action.
        max_ep_len (int): context length K.
        states (torch.Tensor): a batch of states.
        actions (torch.Tensor): a batch of actions.
        rewards (torch.Tensor): a batch of rewards.
        rtg (torch.Tensor): a batch of return-to-go.
        timesteps (torch.Tensor): a batch of timesteps.
        attention_mask (torch.Tensor): Mask for causal Transformer.

    Returns:
        tuple(list, dict): (gradients of different samples, gradient norm per parameter({parameter_name: gradient_norm})
    """    

    loss_fn = lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2)

    model = DecisionTransformer(
        args=variant,
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=variant["K"],
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
        state_dict = torch.load(variant["load_checkpoint"], map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print(f"Loaded from {variant['load_checkpoint']}")

    action_target = torch.clone(actions)
    grads_list = []

    for batch_id in tqdm(range(variant["batch_size"])):
        ##### 勾配計算 #####
        action_target_batch = action_target[batch_id, :, :].unsqueeze(0)

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

        model.zero_grad()

        state_preds, action_preds, reward_preds, all_embs = model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target_batch = action_target.reshape(-1, act_dim)[
            attention_mask.reshape(-1) > 0
        ]

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

        grad_norm_per_param = {}
        for name, param in model.transformer.h.named_parameters():
            grad_norm_per_param[name] = torch.norm(param.grad.view(-1)).numpy()

    return grads_list, grad_norm_per_param