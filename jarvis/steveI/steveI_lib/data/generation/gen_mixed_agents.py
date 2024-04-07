import argparse
import random
import time
from typing import Optional
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

import pickle
import gym
import numpy as np
import cv2
from jarvis.mineclip import MineCLIP
import torch
import os
import uuid

from tqdm import tqdm

from jarvis.steveI.steveI_lib.config import MINECLIP_CONFIG
from jarvis.steveI.steveI_lib.data.EpisodeStorage import EpisodeStorage
from jarvis.steveI.steveI_lib.VPT.agent import MineRLAgent, ENV_KWARGS
from jarvis.steveI.steveI_lib.data.generation.FrameBuffer import FrameBuffer, QueueFrameBuffer
from jarvis.steveI.steveI_lib.data.generation.vpt_agents import VPT_AGENT_PAIRS
from jarvis.steveI.steveI_lib.mineclip_code.load_mineclip import load
from jarvis.steveI.steveI_lib.helpers import timeit_context


def label_episode_dirpath(episode_dirpath: str,
                          max_frames: int,
                          agent1_weights_name: str,
                          agent2_weights_name: str,
                          num_frames: int,
                          term_reason: str,
                          seed: Optional[int]):
    """Add labels to episode dirpath to indicate what weights were run and what was saved."""
    if seed is None:
        raise ValueError('numerical seed must be provided')

    episode_dirpath += f'_models[{agent1_weights_name},{agent2_weights_name}]'
    episode_dirpath += f'_mxFrm{max_frames}'

    episode_dirpath += f'_nmFrm{num_frames}'
    episode_dirpath += f'_seed{seed}'
    episode_dirpath += f'_term{term_reason.capitalize()}'

    return episode_dirpath


def process_frame_mineclip(frame: np.ndarray, height: int = 160, width: int = 256):
    """Processes frame to format that mineclip expects (160x256) and (C, H, W)."""
    assert frame.shape[2] == 3, f'Expected channel dim to be at axis 2, got shape {frame.shape}'

    if frame.shape != (160, 256, 3):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

    return np.moveaxis(frame, -1, 0)


@torch.no_grad()
def embed_videos_mineclip_batched(frame_buffer: QueueFrameBuffer, mineclip, device, batch_size=32):
    """Compute mineclip_code video embedding for an entire QueueFrameBuffer. Returns a listr of 512 vectors
    with shape (1, 512).
    """
    print(f'Embedding {len(frame_buffer)} frames with batch size {batch_size}...')

    frame_iter = iter(frame_buffer)
    prog = tqdm(total=len(frame_buffer))
    video_embeds_all = []
    done = False
    while not done:
        # Get batch of videos
        videos = []
        for _ in range(batch_size):
            try:
                frame = next(frame_iter)
                prog.update(1)
            except StopIteration:
                done = True
                break
            videos.append(frame)
        if len(videos) == 0:
            break

        # Compute embeddings in batched form
        video_batch = torch.cat(videos).to(device)
        bsz = video_batch.shape[0]
        # Autocast so that we can use fp16
        with torch.cuda.amp.autocast():
            video_embeds = mineclip.encode_video(video_batch)
        video_embeds = video_embeds.detach().cpu().numpy()
        assert video_embeds.shape == (bsz, 512)  # batch of 512-vectors

        # Add to list (each embed is its own element)
        for video_embed in video_embeds:
            video_embed = video_embed.reshape(1, 512)
            assert video_embed.shape == (1, 512)
            video_embeds_all.append(video_embed)
    return video_embeds_all


def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


def make_vpt_agent(in_model, in_weights):
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    env = gym.make("MineRLBasaltFindCave-v0")
    agent = MineRLAgent(env, device='cuda', policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(in_weights)
    env.close()
    return agent


def pick_vpt_agent_pair(metadata):
    agent1_info, agent2_info = random.choice(VPT_AGENT_PAIRS)

    # Update metadata
    metadata['agent1_name'] = agent1_info['name']
    metadata['agent1_in_model'] = agent1_info['in_model']
    metadata['agent1_in_weights'] = agent1_info['in_weights']
    metadata['agent2_name'] = agent2_info['name']
    metadata['agent2_in_model'] = agent2_info['in_model']
    metadata['agent2_in_weights'] = agent2_info['in_weights']

    print(f'Loading agent pair {agent1_info["name"]} and {agent2_info["name"]}...')
    agent1 = make_vpt_agent(agent1_info['in_model'], agent1_info['in_weights'])
    agent2 = make_vpt_agent(agent2_info['in_model'], agent2_info['in_weights'])
    return agent1, agent2, metadata


def get_spin_actions(env, total_degrees=360, degrees_per_action=5):
    """Get actions that will perform a spin in the MineRL environment.

    Args:
        env: MineRL environment
        total_degrees: Total number of degrees to turn (how much we to spin)
        degrees_per_action: Number of degrees to turn per action (speed of the spin)
    """
    num_actions = total_degrees // degrees_per_action
    spin_action_wrong_format = env.action_space.noop()
    if random.random() < 0.5:
        # 50% chance of turning left instead of right
        degrees_per_action *= -1
    spin_action_wrong_format['camera'] = [0, degrees_per_action]  # TODO: Add up-down spinning in future?

    # Fix format of spin action to be same as agent action
    spin_action = {}
    for key in spin_action_wrong_format.keys():
        if key in ['ESC', 'pickItem', 'swapHands']:
            continue
        spin_action[key] = np.array([spin_action_wrong_format[key]])

    return [spin_action] * num_actions


def set_switch_agent_probs(agent_names, switch_agent_prob):
    """If one of the models is RL, set the probability of switching to RL to 1/3000 and the probability of switching
    to non-RL to 1/1000. Otherwise, set the probability of switching to either model to switch_agent_prob."""
    assert len(agent_names) == 2, f'Expected 2 agent names, got {len(agent_names)}'
    agent_is_rl = ['rl_from' in agent_name for agent_name in agent_names]
    if sum(agent_is_rl) == 1:
        # One of the agents is RL
        switch_to_rl_prob = 1 / 3000
        switch_to_non_rl_prob = 1 / 1000
        if agent_is_rl[0]:
            # Agent 1 is RL
            switch_to_agent1_prob = switch_to_rl_prob
            switch_to_agent2_prob = switch_to_non_rl_prob
        else:
            # Agent 2 is RL
            switch_to_agent1_prob = switch_to_non_rl_prob
            switch_to_agent2_prob = switch_to_rl_prob
    elif sum(agent_is_rl) == 0:
        switch_to_agent1_prob = switch_to_agent2_prob = switch_agent_prob
    else:
        raise ValueError(f'Expected 0 or 1 RL agents, got {sum(agent_is_rl)}')
    return switch_to_agent1_prob, switch_to_agent2_prob


def generate_episode(episode_dirpath,
                     metadata,
                     agent1,
                     agent2,
                     env,
                     mineclip,
                     max_timesteps,
                     min_timesteps,
                     seed,
                     switch_agent_prob,
                     perform_spin_prob,
                     batch_size):
    """Run an episode of the environment with each command for t timesteps each.
    First, the environment is warmed up for warmup timesteps."""
    tik_episode = time.time()
    assert 0 <= switch_agent_prob <= 1, f'switch_agent_prob must be in [0, 1], got {switch_agent_prob}'
    assert 0 <= perform_spin_prob <= 1, f'perform_spin_prob must be in [0, 1], got {perform_spin_prob}'

    if seed is None:
        seed = np.random.randint(0, 2 ** 25)
    metadata['seed'] = seed

    print(f'Starting new episode with seed {seed}...')
    episode_storage = EpisodeStorage(episode_dirpath)
    frame_buffer = QueueFrameBuffer()

    # Random switching between
    agents = [agent1, agent2]
    agent_names = [metadata['agent1_name'], metadata['agent2_name']]
    # Set the probability of switching to the other agent
    switch_to_agent1_prob, switch_to_agent2_prob = set_switch_agent_probs(agent_names, switch_agent_prob)
    metadata['switch_to_agent1_prob'] = switch_to_agent1_prob
    metadata['switch_to_agent2_prob'] = switch_to_agent2_prob
    # active_agent_idx keeps track of which agent is currently active (False is agent1, True is agent2)
    active_agent_idx = random.choice([False, True])
    # spin_actions is either None if not spinning right now or a list of the spin actions are left
    spin_actions = None

    # Save the switch info when the agent switches and when the agent starts spinning
    switches = [(0, active_agent_idx)]
    spins = []

    env.seed(seed)
    minerl_obs, _ = env.reset()
    num_timesteps = 0
    agent_input_povs = []
    actions = []
    try:
        while num_timesteps < max_timesteps:
            tik_timestep = time.time()

            if active_agent_idx and random.random() < switch_to_agent1_prob:
                # agent2 is active and we want to switch to agent1
                active_agent_idx = False
                agents[active_agent_idx].reset()
                switches.append((num_timesteps, active_agent_idx))
            elif not active_agent_idx and random.random() < switch_to_agent2_prob:
                # agent1 is active and we want to switch to agent2
                active_agent_idx = True
                agents[active_agent_idx].reset()
                switches.append((num_timesteps, active_agent_idx))

            # Get actions for a spin with probably perform_spin_prob
            if spin_actions is None and random.random() < perform_spin_prob:
                spin_actions = get_spin_actions(env, random.randint(90, 360), random.randint(20, 30))
                spins.append(num_timesteps)

            # Process frame for mineclip and add to frame buffer
            frame = minerl_obs['img']
            frame_mineclip = process_frame_mineclip(frame)
            frame_buffer.add_frame(frame_mineclip)

            # Get action
            with timeit_context('agent action'):
                agent_input_pov = agents[active_agent_idx].get_agent_input_pov(frame)
                if spin_actions is None:
                    minerl_action = agents[active_agent_idx].take_action_on_frame(agent_input_pov)
                else:
                    minerl_action = spin_actions.pop(0)
                    if len(spin_actions) == 0:
                        # Reset spin_actions and reset agent
                        spin_actions = None
                        agents[active_agent_idx].reset()
            agent_input_pov = agent_input_pov[0]  # unsqueeze

            # Take env step
            with timeit_context('env step'):
                minerl_obs, _, _, _, _ = env.step(minerl_action)

            # Store input to VPT agent and taken action
            agent_input_povs.append(agent_input_pov)
            actions.append(minerl_action)

            num_timesteps += 1
            tok_timestep = time.time()
            fps = 1 / (tok_timestep - tik_timestep)
            print(f'Timestep {num_timesteps}/{max_timesteps} | {tok_timestep - tik_timestep:.2f}s'
                  f' | {fps:.2f} FPS | ETA: {(max_timesteps - num_timesteps) / fps / 60:.2f}min')
        term_reason = 'maxTimesteps'
    except KeyboardInterrupt as e:
        term_reason = 'keyboard'
        metadata['error_msg'] = str(e)
        print(e)
    except Exception as e:
        term_reason = 'exception'
        metadata['error_msg'] = str(e)
        print(e)

    if num_timesteps < min_timesteps:
        print(f'Episode terminated and not saved due to min_timesteps. {num_timesteps} < {min_timesteps}')
        return

    # Compute mineclip embeddings for frames and add 15 None elems to the beginning of the list
    with timeit_context('mineclip_batched'):
        video_embeds = embed_videos_mineclip_batched(frame_buffer, mineclip, 'cuda', batch_size)
        video_embeds = [None] * 15 + video_embeds
    # Add episode data to episode_storage, TODO: can be more efficient by passing whole lists
    with timeit_context('episode_storage'):
        for agent_input_pov, action, video_embed in zip(agent_input_povs, actions, video_embeds):
            episode_storage.append(agent_input_pov=agent_input_pov,
                                   action=action,
                                   video_embed_attn=video_embed)

    agent1_weights_name = metadata['agent1_in_weights'].split('/')[-1].split('.')[0]
    agent2_weights_name = metadata['agent2_in_weights'].split('/')[-1].split('.')[0]
    labeled_episode_dirpath = label_episode_dirpath(episode_dirpath,
                                                    max_timesteps,
                                                    agent1_weights_name,
                                                    agent2_weights_name,
                                                    num_timesteps,
                                                    term_reason,
                                                    seed)

    episode_storage.update_episode_dirpath(labeled_episode_dirpath)

    metadata['total_minutes'] = round((time.time() - tik_episode) / 60, 4)
    metadata['switches'] = switches
    metadata['spins'] = spins
    episode_storage.save_episode()
    episode_storage.save_metadata(metadata)
    print(f'Episode saved to {labeled_episode_dirpath}')


def main(args):
    print('Loading MineClip...')
    mineclip = load(MINECLIP_CONFIG, device='cuda')

    print('Loading MineRL...')
    env = HumanSurvival(**ENV_KWARGS).make()
    env.reset()  # necessary for seed to register from the start
    print('MineRL loaded.')

    metadata_base = {
        'output_dir': args.output_dir,
        'max_timesteps': args.max_timesteps,
        'min_timesteps': args.min_timesteps,
        'switch_agent_prob': args.switch_agent_prob,
        'perform_spin_prob': args.perform_spin_prob,
    }

    for i in range(args.num_episodes):
        metadata = metadata_base.copy()
        agent1, agent2, metadata = pick_vpt_agent_pair(metadata)
        episode_dirname = str(uuid.uuid4())
        episode_dirpath = os.path.join(args.output_dir, episode_dirname)
        generate_episode(episode_dirpath,
                         metadata,
                         agent1,
                         agent2,
                         env,
                         mineclip,
                         args.max_timesteps,
                         args.min_timesteps,
                         args.env_seed,
                         args.switch_agent_prob,
                         args.perform_spin_prob,
                         args.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='data/dataset_mixed_agents/')
    parser.add_argument('--max_timesteps', type=int, default=7200)
    parser.add_argument('--min_timesteps', type=int, default=1000)
    parser.add_argument('--env_seed', type=int, default=None)
    parser.add_argument('--switch_agent_prob', type=float, default=1 / 1000)
    parser.add_argument('--perform_spin_prob', type=float, default=1 / 750)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_episodes', type=int, default=5)
    args = parser.parse_args()
    main(args)
