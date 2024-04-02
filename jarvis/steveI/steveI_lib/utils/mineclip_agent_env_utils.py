import pickle

import gym
import torch

from jarvis.steveI.steveI_lib.MineRLConditionalAgent import MineRLConditionalAgent
from jarvis.steveI.steveI_lib.config import MINECLIP_CONFIG, DEVICE
from jarvis.steveI.steveI_lib.mineclip_code.load_mineclip import load


def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


def load_mineclip_wconfig():
    print('Loading MineClip...')
    return load(MINECLIP_CONFIG, device=DEVICE)


def make_env(env_config):
    from jarvis.stark_tech.ray_bridge import MinecraftWrapper
    print('Loading MineRL...')
    env = MinecraftWrapper(env_config)
    print('Starting new env...')
    env.reset()
    return env


def make_agent(in_model, in_weights, cond_scale, device):
    print(f'Loading agent with cond_scale {cond_scale}...')
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    # Make conditional agent
    agent = MineRLConditionalAgent(device=device, policy_kwargs=agent_policy_kwargs,
                                   pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(in_weights)
    agent.reset(cond_scale=cond_scale)
    return agent


def load_mineclip_agent_env(in_model, in_weights, cond_scale, env_config, device='cuda'):
    mineclip = load_mineclip_wconfig()
    agent = make_agent(in_model, in_weights, cond_scale=cond_scale, device=device)
    env = make_env(env_config)
    return agent, mineclip, env
