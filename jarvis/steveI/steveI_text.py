from typing import Callable, Dict, Optional, Tuple, Union
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig
import torch
from pathlib import Path
import numpy as np

from jarvis.steveI.steveI_lib.utils.mineclip_agent_env_utils import make_agent
from jarvis.steveI.steveI_lib.utils.embed_utils import get_prior_embed
from jarvis.steveI.steveI_lib.config import MINECLIP_CONFIG, PRIOR_INFO
import jarvis.steveI.steveI_lib.mineclip_code.load_mineclip as load_mineclip
from jarvis.steveI.steveI_lib.data.text_alignment.vae import load_vae_model
from jarvis.assembly.base import MarkBase
from jarvis.stark_tech.env_interface import MinecraftWrapper
from jarvis.steveI.path import VPT_MODEL_PATH, VPT_WEIGHT_PATH, PRIOR_WEIGHT_PATH, MINECLIP_WEIGHT_PATH

class SteveIText(MarkBase):
    
    def __init__(self, env, cond_scale: float = 6.0,  device: Union[str, torch.device] = "cuda", **unused_kwargs):
        self.env = env
        self.cond_scale = cond_scale
        mineclip_config = MINECLIP_CONFIG
        mineclip_config['ckpt']['path'] = str(MINECLIP_WEIGHT_PATH)

        self.device = device
        self.mineclip = load_mineclip.load(mineclip_config, device=self.device)
        prior_info = PRIOR_INFO
        prior_info['model_path'] = str(PRIOR_WEIGHT_PATH)
        self.prior = load_vae_model(prior_info, device=self.device)
        self.agent = make_agent(str(VPT_MODEL_PATH), str(VPT_WEIGHT_PATH), cond_scale=self.cond_scale, device=self.device)
        self.agent.reset(cond_scale=self.cond_scale)

    def reset(self):
        super().reset()
        self.agent.reset(cond_scale=self.cond_scale)
    
    def do(
        self, 
        condition: str = '', 
        timeout: int = 500, 
        target_reward: float = 1., 
        monitor_fn: Optional[Callable] = None,
        **kwargs, 
    ) -> Tuple[bool, Dict]:
        prompt_embed = get_prior_embed(condition, self.mineclip, self.prior, device=self.device)
        
        self.reset()

        self.obs, reward, terminated, truncated, self.info = self.env.step(self.env.noop_action())
        time_step = 0
        episode_reward = 0

        while (
            not terminated 
            and not truncated
            and time_step < timeout
        ):
            with torch.cuda.amp.autocast():
                minerl_action = self.agent.get_action(self.obs, prompt_embed)
            
            masked_minerl_actions = minerl_action
            masked_minerl_actions['hotbar.1'] = np.array([0])
            masked_minerl_actions['hotbar.2'] = np.array([0])
            masked_minerl_actions['hotbar.3'] = np.array([0])
            masked_minerl_actions['hotbar.4'] = np.array([0])
            masked_minerl_actions['hotbar.5'] = np.array([0])
            masked_minerl_actions['hotbar.6'] = np.array([0])
            masked_minerl_actions['hotbar.7'] = np.array([0])
            masked_minerl_actions['hotbar.8'] = np.array([0])
            masked_minerl_actions['hotbar.9'] = np.array([0])
            if self.info['isGuiOpen']:
                masked_minerl_actions['inventory'] = np.array([1])
            else:
                masked_minerl_actions['inventory'] = np.array([0])
                
            self.obs, self.reward, terminated, truncated, self.info = self.env.step(masked_minerl_actions)
            self.record_step()

            if monitor_fn is not None:
                monitor_result = monitor_fn(self.info)
                if monitor_result[0]:
                    return monitor_result
                
            episode_reward += self.reward
            time_step += 1

            if episode_reward >= target_reward:
                return True, {'success': True, 'terminated': False}
        if terminated:
            return False, {'reason': "environment reset.", "terminated": True}
        else:
            return False, {'reason': "reach goal maximum steps.", "terminated": False}

if __name__ == '__main__':
    from jarvis.assembly.marks import RenderWrapper
    
    env = MinecraftWrapper('collect_grass')
    env = RenderWrapper(env)
    env.reset()
    
    steveI = SteveIText(env=env)
    steveI.reset()

    result, error_message = steveI.do('collect grass')
    print(result, error_message)