import os
import pickle
from typing import Optional
from jarvis.steveI.steveI_lib.data.EpisodeStorage import EpisodeStorage
import numpy as np
import cv2
from jarvis.steveI.steveI_lib.VPT.lib.tree_util import tree_map
from jarvis.steveI.steveI_lib.helpers import object_to_numpy, batch_recursive_objects
from torch.utils.data import Dataset
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from jarvis.steveI.steveI_lib.VPT.agent import AGENT_RESOLUTION, resize_image, ActionTransformer, \
    ACTION_TRANSFORMER_KWARGS, CameraHierarchicalMapping

NONE_EMBED_OFFSET = 15

def env_obs_to_agent(frame, embed):
    """
    Turn observation from MineRL environment into model's observation

    Returns torch tensors.
    """
    agent_input = resize_image(frame, AGENT_RESOLUTION)[None]
    return {
        "img": agent_input,
        "mineclip_embed": embed,
    }


action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)
action_mapper = CameraHierarchicalMapping(n_camera_bins=11)

def load_sampling(sample_dir, sample_name):
    train_path = os.path.join(sample_dir, sample_name + "_train.txt")
    val_path = os.path.join(sample_dir, sample_name + "_val.txt")

    train_episodes = []
    with open(train_path, "r") as f:
        for line in f:
            train_episodes.append(line.strip())

    val_episodes = []
    with open(val_path, "r") as f:
        for line in f:
            val_episodes.append(line.strip())

    return train_episodes, val_episodes

def env_action_to_agent(minerl_action_transformed):
    minerl_action = action_transformer.env2policy(minerl_action_transformed)
    if minerl_action["camera"].ndim == 1:
        minerl_action = {k: v[None] for k, v in minerl_action.items()}
    action = action_mapper.from_factored(minerl_action)
    return action


def get_episode_chunk(episode_chunk, min_btwn_goals, max_btwn_goals, p_uncond):
    """Get a chunk of an episode as described by episode_chunk.

    Args:
        episode_chunk (tuple): (episode_dirpath, start_timestep, end_timestep)
        min_btwn_goals (int): Minimum number of timesteps between goals.
        max_btwn_goals (int): Maximum number of timesteps between goals.
        p_uncond (float): Probability of setting the goal embed to zero.
        rate_augmented_embeds (float): The probability of using an augmented embed instead of the original embed.
        text_embeddings (dict[list[np.ndarray]]): Mapping from episode_dirpath to the list of text embeddings for that episode.
    """
    episode_dirpath, start_timestep, end_timestep = episode_chunk
    T = end_timestep - start_timestep
    episode = EpisodeStorage(episode_dirpath)

    # Get the goal embeddings
    embeds = episode.load_embeds_attn()

    frames = episode.load_frames(only_range=(start_timestep, end_timestep))
    total_timesteps = len(episode)

    # Choose goal timesteps
    goal_timesteps = []
    curr_timestep = 0
    while curr_timestep < total_timesteps - 1:
        curr_timestep += np.random.randint(min_btwn_goals, max_btwn_goals)
        if (total_timesteps - curr_timestep) < min_btwn_goals:
            curr_timestep = total_timesteps - 1
        goal_timesteps.append(curr_timestep)

    embeds_per_timestep = []
    # Tranlate into embeds per timestep
    cur_goal_timestep_idx = 0
    for t in range(total_timesteps):
        goal_timestep = goal_timesteps[cur_goal_timestep_idx]
        embed = embeds[goal_timestep]

        embeds_per_timestep.append(embed)
        if t == goal_timesteps[cur_goal_timestep_idx] + 1:
            # We've reached the timestep after the goal timestep, so move to the next goal
            cur_goal_timestep_idx += 1

    # With probability p_uncond, set the embeds to zero
    if np.random.rand() < p_uncond:
        embeds_per_timestep = [np.zeros_like(embed) for embed in embeds_per_timestep]

    # Load the actions
    all_actions = episode.load_actions()

    obs_list = []
    actions_list = []
    firsts_list = [True] + [False] * (T - 1)

    for t in range(start_timestep, end_timestep):
        frame = frames[t]

        obs = env_obs_to_agent(frame, embeds_per_timestep[t].reshape(1, -1))
        obs_list.append(obs)

        action = env_action_to_agent(all_actions[t])
        actions_list.append(action)

    obs_np = batch_recursive_objects(obs_list)
    actions_np = batch_recursive_objects(actions_list)
    firsts_np = np.array(firsts_list, dtype=bool).reshape(T, 1)

    return obs_np, actions_np, firsts_np


def batch_if_numpy(xs):
    if isinstance(xs, np.ndarray):
        return np.array(xs)
    else:
        return xs


class MinecraftDataset(Dataset):
    def __init__(self, episode_dirnames, T, min_btwn_goals, max_btwn_goals,
                 p_uncond=0.1, limit=None, every_nth=None):
        """A dataset of Minecraft episodes.

        Args:
            dir (string): Directory with all the episodes.
            T (int): Number of timesteps per chunk (returned by __getitem__).
            min_btwn_goals (int): Minimum number of timesteps between goals.
            max_btwn_goals (int): Maximum number of timesteps between goals.
            p_uncond (float): Probability of setting the goal embed to zero.
            limit (int): Limit the number of episodes to this number.
            every_nth (int): Only use every nth episode.
        """
        assert min_btwn_goals >= 15, "min_btwn_goals must be >= 15 since the first 15 embeds are None (min_btwn_goals" \
                                     "is for an index, which is why = is allowed - it's really the 16th embed)"
        assert min_btwn_goals < max_btwn_goals, "min_btwn_goals must be < max_btwn_goals"
        if limit is not None:
            print(f"Limiting dataset to {limit} episodes")
        # Print whether we use geometric or uniform sampling
        print(f"Using uniform sampling with min_btwn_goals={min_btwn_goals} and max_btwn_goals={max_btwn_goals}")
        self.episode_chunks = create_episode_chunks(episode_dirnames, T, min_btwn_goals, limit=limit)
        if every_nth is not None:
            self.episode_chunks = self.episode_chunks[::every_nth]
        self.min_btwn_goals = min_btwn_goals
        self.max_btwn_goals = max_btwn_goals
        self.T = T
        self.p_uncond = p_uncond

    def __len__(self):
        return len(self.episode_chunks)

    def __getitem__(self, idx):
        obs_np, actions_np, firsts_np = \
            get_episode_chunk(self.episode_chunks[idx], 
                              self.min_btwn_goals, 
                              self.max_btwn_goals,
                              self.p_uncond)
        return obs_np, actions_np, firsts_np

    def collate_fn(self, batch):
        obs_np, actions_np, firsts_np = zip(*batch)
        obs = batch_recursive_objects(obs_np)
        actions = batch_recursive_objects(actions_np)
        firsts = np.array(firsts_np)

        return obs, actions, firsts

    def get_total_frames(self):
        total_frames = 0
        for episode_chunk in self.episode_chunks:
            episode_dirpath, start_timestep, end_timestep = episode_chunk
            total_frames += end_timestep - start_timestep
        return total_frames


def validate_episodes(episode_dirnames, T, min_btwn_goals, limit=None):
    """Return list of tuples (episode_dirpath, episode_len) for valid episodes.
    Same as get_valid_episodes, but you pass in the list of episode_dirnames instead of the directory."""
    print(f'Getting valid episodes from the provided episode list...')
    if limit is not None:
        episode_dirnames = episode_dirnames[:limit]

    # Filter valid episodes and append tuple (episode_dirpath, episode_len)
    min_len = max(min_btwn_goals + NONE_EMBED_OFFSET, T + NONE_EMBED_OFFSET)
    valid_episodes = []
    total_frames = 0
    for episode_dirname in tqdm(episode_dirnames):
        episode_storage = EpisodeStorage(episode_dirname)
        if episode_storage.is_valid(for_training=True, min_frames_training=min_len, expensive=True)[0]:
            episode_len = len(episode_storage)
            total_frames += episode_len
            valid_episodes.append((episode_dirname, episode_len))
    print(f'Found {len(valid_episodes)} valid episodes.')
    print(f'Total frames: {total_frames:,}')
    return valid_episodes

def get_valid_episodes(dir, T, min_btwn_goals, limit=None):
    """Return list of tuples (episode_dirpath, episode_len) for valid episodes."""
    print(f'Getting valid episodes from {dir}...')
    episode_dirnames = os.listdir(dir)
    if limit is not None:
        episode_dirnames = episode_dirnames[:limit]

    # Filter valid episodes and append tuple (episode_dirpath, episode_len)
    min_len = max(min_btwn_goals + NONE_EMBED_OFFSET, T + NONE_EMBED_OFFSET)
    valid_episodes = []
    for episode_dirnames in tqdm(episode_dirnames):
        episode_storage = EpisodeStorage(os.path.join(dir, episode_dirnames))
        if episode_storage.is_valid(for_training=True, min_frames_training=min_len, expensive=True)[0]:
            episode_len = len(episode_storage)
            episode_dirpath = os.path.join(dir, episode_dirnames)
            valid_episodes.append((episode_dirpath, episode_len))
    print(f'Found {len(valid_episodes)} valid episodes.')
    return valid_episodes


def create_episode_chunks(episode_dirnames, T, min_btwn_goals, limit=None):
    """For all valid episodes, create a list of (episode_dirpath, start_timestep, end_timestep) tuples.
    Each tuple represents a chunk of T timesteps in the episode. The list covers all chunks in all valid episodes.
    """
    valid_episodes = validate_episodes(episode_dirnames, T, min_btwn_goals, limit=limit)
    episode_chunks = []
    for episode_dirpath, episode_len in valid_episodes:
        # Create chunks of size T. Adjust the last chunk if it is too small.
        for chunk_idx in range(NONE_EMBED_OFFSET, episode_len, T):
            start_timestep = chunk_idx
            end_timestep = start_timestep + T
            if end_timestep > episode_len:
                # The last chunk is too small, so we adjust it to be the last T timesteps.
                # There will be some overlap with the previous chunk, but that's okay.
                start_timestep = episode_len - T
                end_timestep = episode_len
            episode_chunks.append((episode_dirpath, start_timestep, end_timestep))
    return episode_chunks
