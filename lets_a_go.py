#%%
import torch
from torch.utils.data import Dataset, DataLoader
import einops
import numpy as np
import random
import json
import itertools
from argparse import Namespace
import streamlit as st
import matplotlib.pyplot as plt
from plotly import express as px
from typing import Tuple, List, Optional, Literal, Union
from dataclasses import dataclass
import logging
from transformer_lens import ActivationCache
from transformer_lens.utils import get_act_name

from src.environments.registration import register_envs
from src.config import EnvironmentConfig
from src.environments.environments import make_env
from src.decision_transformer.utils import (
    load_decision_transformer,
    get_max_len_from_model_type,
    initialize_padding_inputs
)
from plotly_utils import imshow, line

register_envs()

# device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device = torch.device("cpu") #! FIXME

#%%
TargetObj = Literal['key', 'ball', None]
TargetPos = Literal['top', 'bottom', None]
#%%
# UTILS
action_labels = ['left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done']

# %%
# load the model and create the environment
model_path = 'models/MiniGrid-MemoryS7FixedStart-v0/WorkingModel.pt'
state_dict = torch.load(model_path)

env_config = state_dict["environment_config"]
env_config = EnvironmentConfig(**json.loads(env_config))

env = make_env(env_config, seed=42, idx=0, run_name="dev")()
model = load_decision_transformer(model_path, env, tlens_weight_processing=True).to(device)
if not hasattr(model, "transformer_config"):
    model.transformer_config = Namespace(
        n_ctx=model.n_ctx,
        time_embedding_type=model.time_embedding_type,
    )

max_len = get_max_len_from_model_type(
    model.model_type,
    model.transformer_config.n_ctx,
)
#%%
def render_env(env):
    '''
    Render the environment in streamlit
    '''
    fig = px.imshow(env.render())
    fig.update_layout(
        width=400,
        height=400,
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    fig.show()
    st.write(fig)

#%%
@dataclass
class EnvState:
    '''
        A dataclass to hold the state of the environment
        (agent_pos, all the stuff the model needs, and whether we've terminated)
    '''
    agent_pos: torch.Tensor
    agent_dir: torch.Tensor
    target_pos: torch.Tensor
    obs: torch.Tensor
    actions: torch.Tensor
    rtg: torch.Tensor
    timesteps: torch.Tensor
    terminated: bool
    truncated: bool

    def __repr__(self):
        min_rtg = self.rtg.min().item()
        max_rtg = self.rtg.max().item()
        rtg_str = f'Min RTG={min_rtg:.2f}, Max RTG={max_rtg:.2f}' if min_rtg != max_rtg else f'RTG={min_rtg:.2f}'
        max_timestep = self.timesteps.max().item()
        timestep_str = f'Timesteps to {max_timestep}'
        return f"""EnvState(agent_pos={self.agent_pos}, agent_dir={self.agent_dir}, target_pos={self.target_pos},
        obs: {self.obs.shape}, actions: {self.actions.shape}, {rtg_str}, {timestep_str},
        terminated={self.terminated}, truncated={self.truncated})"""

def init_episode(rtg: float, target_obj: TargetObj = None,
                 target_pos: TargetPos = None) -> EnvState:
    '''
        Initialize an episode
    '''
    obs, _ = env.reset(options={'target_obj': target_obj, 'target_pos': target_pos})
    obs, actions, reward, rtg, timesteps, mask = initialize_padding_inputs(
        max_len=max_len,
        initial_obs=obs,
        initial_rtg=rtg,
        action_pad_token=env.action_space.n,
        batch_size=1,
        device=device,
    )
    return EnvState(env.agent_pos, env.agent_dir, env.target_pos,
                    obs, actions, torch.tensor(rtg), timesteps, False, False)

def step(state: EnvState, new_action: Optional[torch.Tensor] = None) -> EnvState:
    '''
        Take a step in the environment

        If new_action is None, we use the model to predict the next action
    '''
    # all the important stuff is in these 4 lines - model go brr, argmax, take action
    if new_action is None:
        _, action_preds, _ = model.forward(states=state.obs,
                                           actions=state.actions,
                                           rtgs=state.rtg,
                                           timesteps=state.timesteps)
        new_action = torch.argmax(action_preds[:, -1], dim=-1).squeeze(-1)
    new_obs, new_reward, terminated, truncated, info = env.step(new_action)

    # concat init obs to new obs
    obs = torch.cat(
        [state.obs, torch.tensor(new_obs["image"])[None, None, ...].to(device)], dim=1
    )

    # add new reward to init reward
    new_rtg = state.rtg[:, -1:, :] - new_reward
    rtg = torch.cat([state.rtg, new_rtg], dim=1)

    # add new timesteps
    # if we are done, we don't want to increment the timestep,
    # so we use the not operator to flip the done bit
    new_timestep = state.timesteps[:, -1:, :] + 1
    timesteps = torch.cat([state.timesteps, new_timestep], dim=1)

    actions = torch.cat(
        [state.actions, new_action[None, None, None]], dim=1
    )
    # truncations:
    obs = obs[:, -max_len:]
    actions = actions[:, -(max_len - 1) :]
    timesteps = timesteps[:, -max_len:]
    rtg = rtg[:, -max_len:]

    return EnvState(env.agent_pos, env.agent_dir, env.target_pos,
                    obs, actions, rtg, timesteps, terminated, truncated)

# %%
def run_episode(rtg: float):
    '''
        Run an episode and log/render some stats throughout
    '''
    state = init_episode(rtg)

    # logging and stuff
    print(f'{env.target_obj=:<10} {env.target_pos=}')
    print(f'INITIAL:             agent pos: {env.agent_pos}      agent dir: {env.agent_dir}')
    render_env(env)

    while not state.terminated and not state.truncated:
        state = step(state)

        render_env(env)

        print(f'action: {action_labels[state.actions[0, -1, 0]]:<12} agent pos: {env.agent_pos}      agent dir: {env.agent_dir}')
        if env.agent_pos == (5, 3) and env.agent_dir == 0:
            print('DECISION POINT')
        if state.terminated or state.truncated:
            if (env.agent_pos == (5, 2) and env.target_pos == 'top') or (
                env.agent_pos == (5, 4) and env.target_pos == 'bottom'):
                print('SUCCESS')
            else:
                print('FAILURE')
# %%
# Running the episode in streamlit
st.write("# Let's a go!")
rtg = st.slider('Reward to go', min_value=0.0, max_value=1.0, value=0.8, step=0.01)
if st.button('Run episode'):
    run_episode(rtg)
# %%
def change_dir(current_dir: int, new_dir: int) -> List[int]:
    '''
    Returns a list of actions to change direction from current_dir to new_dir
    '''
    # TODO if we make this stochastic (eg turn left 3 times sometimes instead of right once) we can get more data
    if current_dir == new_dir:
        return []
    if (current_dir + 1) % 4 == new_dir: # turn right
        return [1]
    if current_dir == (new_dir + 1) % 4: # turn left
        return [0]
    if (current_dir + 2) % 4 == new_dir: # turn around
        return [0, 0]# if random.random() < 0.5 else [1, 1]
    raise ValueError(f"Invalid direction change: {current_dir} -> {new_dir}. Congratulations, you broke math!")

def create_trajectory_to(final_pos: Tuple[int, int], final_dir: int,
                         starting_pos: Tuple[int, int] = (1, 3), starting_dir: int = 0,
                         supress_start_pos_warning: bool = False,
                         ) -> List[int]:
    '''
    Creates a trajectory (ie actions) to a given position and direction

    Note: this function cannot see walls or any other obstacles, but as long as
    the starting position is the default and the final position is valid it'll work

    Both positions are given as (y, x) tuples (following Minigrid's idiocy)
    '''
    # TODO can be more stochastic (as long as we ensure it doesn't bumpt into walls)
    if starting_pos != (1, 3) and not supress_start_pos_warning:
        logging.warning('WARNING: using create_trajectory_to with a non-default starting position can result in trajectories that teleport through walls')
        
    x1, y1 = starting_pos
    x2, y2 = final_pos
    dx = x2 - x1
    dy = y2 - y1
    current_dir = starting_dir
    moves = []
    if dx > 0: # move right
        if current_dir != 0:
            moves.extend(change_dir(current_dir, 0))
            current_dir = 0
        moves.extend([2] * dx)
    elif dx < 0: # move left
        if current_dir != 2:
            moves.extend(change_dir(current_dir, 2))
            current_dir = 2
        moves.extend([2] * abs(dx))
    if dy > 0: # move down
        if current_dir != 1:
            moves.extend(change_dir(current_dir, 1))
            current_dir = 1
        moves.extend([2] * dy)
    elif dy < 0: # move up
        if current_dir != 3:
            moves.extend(change_dir(current_dir, 3))
            current_dir = 3
        moves.extend([2] * abs(dy))
    if current_dir != final_dir: # change direction
        moves.extend(change_dir(current_dir, final_dir))
        current_dir = final_dir
    return moves

# %%
create_trajectory_to((3, 4), 3)
# %%
def create_full_trajectory_to(final_pos: Tuple[int, int], final_dir: int, rtg: float,
                              starting_pos: Tuple[int, int] = (1, 3), starting_dir: int = 0,
                              target_obj: TargetObj = None, target_pos: TargetPos = None,
                              ) -> EnvState:
    '''
        Returns the observations, actions, rewards, and timesteps for a trajectory to a given position and direction
    '''
    state = init_episode(rtg, target_obj, target_pos)
    moves = create_trajectory_to(final_pos, final_dir, starting_pos, starting_dir)
    for move in moves:
        state = step(state, torch.tensor(move).to(device))
        if state.terminated or state.truncated:
            break
    return state

# %%
# get all the valid positions (ie empty spaces that aren't right next to the target)
env.reset()
valid_positions = []
for x, y in itertools.product(range(7), range(7)):
    if env.grid.grid[7*y+x] is None and (y != 5 or x == 3):
        valid_positions.append((y, x))
# %%
def get_all_trajectories(rtg: float, target_obj: TargetObj = None,
                         target_pos: TargetPos = None) -> List[EnvState]:
    '''
    Returns a list of trajectories to all valid positions
    '''
    trajectories_to_all_positions = []
    for pos in valid_positions:
        for dir in range(4):
            trajectories_to_all_positions.append(
                create_full_trajectory_to(pos, dir, rtg,
                                          target_obj=target_obj, target_pos=target_pos)
            )
    return trajectories_to_all_positions
#%%
def optimal_policy(state: EnvState) -> int:
    '''
    Returns the optimal action to take from any given position

    Acts as the ground truth for the optimal policy
    '''
    pos = state.agent_pos
    dir = state.agent_dir
    if state.target_pos not in ['top', 'bottom']:
        raise ValueError(f'Invalid target position: {state.target_pos}')
    target_pos = (5, 1) if state.target_pos == 'top' else (5, 5)
    # TODO make this not be stupidly inefficient - we're calculating the whole trajectory for no reason
    # if we're at column=5, move towards the target
    if pos[0] == 5:
        return create_trajectory_to(target_pos, 0, pos, dir, supress_start_pos_warning=True)[0]
    # if we're at (3, 3) or (4, 3), move to (5, 3)
    if pos == (3, 3) or pos == (4, 3):
        return create_trajectory_to((5, 3), 0, pos, dir, supress_start_pos_warning=True)[0]
    # if we're anywhere else, move to (3, 3)
    return create_trajectory_to((3, 3), 0, pos, dir, supress_start_pos_warning=True)[0]
#%%
# create datasets
class TrajectoriesDataset(Dataset):
    def __init__(self, high_rtg: Optional[bool] = None, target_obj: TargetObj = None,
                 target_pos: TargetPos = None):
        '''
        Creates a dataset of trajectories

        If high_rtg is None, returns both high-rtg and low-rtg trajectories
        If target_obj is None, returns trajectories with both the ball and the key as the target object
        If target_pos is None, returns trajectories with both the target object at the top and the bottom
        '''
        high_rtg_options = [0.95, 0.05] if high_rtg is None else ([0.95] if high_rtg else [0.05])
        target_obj_options = [None, 'ball', 'key'] if target_obj is None else [target_obj]
        target_pos_options = [None, 'top', 'bottom'] if target_pos is None else [target_pos]

        data = []
        # TODO this stuff can be made more efficient, we can prob just change RTG in the trajectories directly
        for high_rtg_opt, target_obj_opt, target_pos_opt in itertools.product(high_rtg_options,
                                                                         target_obj_options,
                                                                         target_pos_options):
            data.extend(get_all_trajectories(high_rtg_opt, target_obj_opt, target_pos_opt))
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
# %%
full_dataset = TrajectoriesDataset()
print(len(full_dataset))
high_rtg_dataset = TrajectoriesDataset(high_rtg=True)
# %%
def embed_state(state: EnvState) -> torch.Tensor:
    return model.to_tokens(state.obs, state.actions, state.rtg, state.timesteps)
first_row_embeddings = embed_state(full_dataset[0])
transformer_outs, cache = model.transformer.run_with_cache(first_row_embeddings)
def next_action_logits(transformer_outs) -> torch.Tensor:
    _, action_logits, _ = model.get_logits(transformer_outs, batch_size=transformer_outs.shape[0],
                                           seq_length=max_len, no_actions=False)
    return action_logits[:, -1]
print(next_action_logits(transformer_outs))
# %%
# print out the logits and correct answers for a couple of random trajectories (high_rtg only)
for i in range(5):
    random_index = random.randint(0, len(high_rtg_dataset))
    state = high_rtg_dataset[random_index]
    embeddings = embed_state(state)
    outs, cache = model.transformer.run_with_cache(embeddings)
    print(next_action_logits(outs), optimal_policy(state))
#%%
# how often does the model get the correct action? (high_rtg only)
correct = 0
incorrect = 0
for state in high_rtg_dataset:
    embeddings = embed_state(state)
    outs, cache = model.transformer.run_with_cache(embeddings)
    if torch.argmax(next_action_logits(outs)) == optimal_policy(state):
        correct += 1
    else:
        incorrect += 1
print(f"{correct=}, {incorrect=} (IMPORTANT NOTE: This is comparing the model's actions to that of an optimal policy, but the optimal policy isn't unique so it's possible that an optimal model will still be classified as incorrect here")
# %%
def trajectory_to_decision_square(high_rtg: bool, target_obj: TargetObj,
                                  target_pos: TargetPos) -> EnvState:
    '''
    Returns a trajectory to the decision square with the given parameters
    '''
    return create_full_trajectory_to(final_pos=(5, 3), final_dir=0,
                                     rtg=(0.95 if high_rtg else 0.05),
                                     target_obj=target_obj, target_pos=target_pos)
traj_pos_top = trajectory_to_decision_square(high_rtg=True, target_obj='ball', target_pos='top')
traj_pos_bottom = trajectory_to_decision_square(high_rtg=True, target_obj='ball', target_pos='bottom')
top_embed = embed_state(traj_pos_top)
bottom_embed = embed_state(traj_pos_bottom)

top_outs, top_cache = model.transformer.run_with_cache(top_embed)
bottom_outs, bottom_cache = model.transformer.run_with_cache(bottom_embed)
# %%
def plot_act_diff(hook_point: list, timestep: int = -1, put_in_square: bool = True):
    full_hook_point = get_act_name(*hook_point)
    top_act = top_cache[full_hook_point][0]
    bottom_act = bottom_cache[full_hook_point][0]
    diff_act = (top_act - bottom_act)[timestep]
    title = f"Top-Bottom in {hook_point} at timestep {timestep}"
    if put_in_square:
        diff_act = diff_act.reshape((16, -1))
        title += " (arbitrarily put into a square shape for readability)"
    imshow(diff_act, title=title)
plot_act_diff(['post', 1, 'mlp'], -1)
# %%
logit_diff_dir = model.action_predictor.weight[0] - model.action_predictor.weight[1]
def residual_stack_to_logit_diff(
    residual_stack: torch.Tensor, 
    cache: ActivationCache,
    logit_diff_directions: torch.Tensor = logit_diff_dir,
) -> torch.Tensor:
    '''
    Gets the avg logit difference between the correct and incorrect answer for a given 
    stack of components in the residual stream.
    '''
    # SOLUTION
    batch_size = residual_stack.size(-2)
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
    return einops.einsum(
        scaled_residual_stack, logit_diff_directions,
        "... batch d_model, d_model -> ..."
    ) / batch_size
#%%
def plot_attribution(should_use_bottom: bool = False, accumulated: bool = False):
    cache = bottom_cache if should_use_bottom else top_cache
    if accumulated:
        residual_stack, labels = cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)
    else:
        residual_stack, labels = cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)
    diff = residual_stack_to_logit_diff(residual_stack, cache)
    if should_use_bottom: diff = -diff
    line(
        diff, 
        hovermode="x unified",
        title=f"Logit Difference From {'Accumulated ' if accumulated else ''}Residual Stream {'(Bottom)' if should_use_bottom else '(Top)'}",
        labels={"x": "Layer", "y": "Logit Diff"},
        xaxis_tickvals=labels,
        width=800
    )
def plot_attribution_all():
    plot_attribution()
    plot_attribution(accumulated=True)
    plot_attribution(should_use_bottom=True)
    plot_attribution(should_use_bottom=True, accumulated=True)
#%%
plot_attribution_all()
#%%
imshow(top_cache['post', 2, 'mlp'][0,-1].reshape(16,-1))
# %%
imshow(bottom_cache['post', 2, 'mlp'][0,-1].reshape(16,-1))
# %%
def plot_neuron_diff(should_use_bottom: bool = False, layer: int = -1, thresh: float = 0.4):
    if layer < -1: raise ValueError("layer must be -1 or greater")
    cache = bottom_cache if should_use_bottom else top_cache
    neuron_stack, labels = cache.stack_neuron_results(layer=layer if layer == -1 else layer+1,
                                                      pos_slice=-1, return_labels=True)
    if layer != 0:
        neuron_stack = neuron_stack[-256:]
        labels = labels[-256:]
    neuron_diffs = residual_stack_to_logit_diff(neuron_stack, cache)
    if should_use_bottom: neuron_diffs = -neuron_diffs

    imshow(neuron_diffs.reshape(16, -1),
        title=f"Logit Difference of Neurons in MLP{layer if layer != -1 else '2'} on target_pos_{'bottom' if should_use_bottom else 'top'} (squarified for readability)")
    # print the labels of neurons with an absolute logit difference of > thresh
    print(f"Neurons with an absolute logit difference of > {thresh}")
    for i, label in enumerate(labels):
        if abs(neuron_diffs[i]) > thresh:
            print(f"{label:>6}: {neuron_diffs[i]}")
# %%
plot_neuron_diff()
plot_neuron_diff(should_use_bottom=True)
# %%
# changes some of the global vars, if you then re-run the cells above it'll use these values
# (yes this is extremely cursed, yes I'm sorry)

traj_pos_top = trajectory_to_decision_square(high_rtg=True, target_obj='key', target_pos='top')
traj_pos_bottom = trajectory_to_decision_square(high_rtg=True, target_obj='key', target_pos='bottom')
top_embed = embed_state(traj_pos_top)
bottom_embed = embed_state(traj_pos_bottom)

top_outs, top_cache = model.transformer.run_with_cache(top_embed)
bottom_outs, bottom_cache = model.transformer.run_with_cache(bottom_embed)

plot_attribution_all()
plot_neuron_diff()
plot_neuron_diff(should_use_bottom=True)
#%%
plot_neuron_diff(layer=1, thresh=0.2)
# %%
