import os
from typing import Callable, Dict, Optional, Tuple, Union
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig
import torch
from pathlib import Path
import numpy as np
import cv2
from jarvis.steveI.steveI_lib.data.generation.FrameBuffer import QueueFrameBuffer

from jarvis.steveI.steveI_lib.utils.mineclip_agent_env_utils import make_agent
from jarvis.steveI.steveI_lib.config import MINECLIP_CONFIG
import jarvis.steveI.steveI_lib.mineclip_code.load_mineclip as load_mineclip
from jarvis.assembly.marks import MarkBase
from jarvis.stark_tech.env_interface import MinecraftWrapper
from jarvis.steveI.path import VPT_MODEL_PATH, VPT_WEIGHT_PATH, PRIOR_WEIGHT_PATH, MINECLIP_WEIGHT_PATH

def process_frame_mineclip(frame: np.ndarray, height: int = 160, width: int = 256):
    """Processes frame to format that mineclip expects (160x256) and (C, H, W)."""
    assert frame.shape[2] == 3, f'Expected channel dim to be at axis 2, got shape {frame.shape}'

    if frame.shape != (160, 256, 3):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

    return np.moveaxis(frame, -1, 0)

def load_video(video_path):
    video = cv2.VideoCapture(video_path)

    frames_mineclip = []

    while True:
        ret, frame = video.read()
        if ret:
            cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
            frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
            mineclip_frame = process_frame_mineclip(frame)
            frames_mineclip.append(mineclip_frame)
        else:
            break

    video.release()
    return frames_mineclip

# Copied from steveI_lib/data/generation/gen_mixed_agents.py
@torch.no_grad()
def embed_videos_mineclip_batched(frame_buffer: QueueFrameBuffer, mineclip, device, batch_size=32):
    """Compute mineclip_code video embedding for an entire QueueFrameBuffer. Returns a listr of 512 vectors
    with shape (1, 512).
    """
    print(f'Embedding {len(frame_buffer)} frames with batch size {batch_size}...')

    frame_iter = iter(frame_buffer)
    video_embeds_all = []
    done = False
    while not done:
        # Get batch of videos
        videos = []
        for _ in range(batch_size):
            try:
                frame = next(frame_iter)
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

def get_visual_embed(video_path, mineclip):
    assert os.path.exists(video_path), f"Trajectory file {video_path} does not exist"
    frames_mineclip = load_video(video_path)[-16:]
    frame_buffer = QueueFrameBuffer()
    for frame in frames_mineclip:
        frame_buffer.add_frame(frame)
    mineclip_embeds = embed_videos_mineclip_batched(frame_buffer, mineclip, 'cuda', 1)
    assert len(mineclip_embeds) == 1
    mineclip_embeds = mineclip_embeds[0]
    assert mineclip_embeds.shape == (1, 512)
    return mineclip_embeds

class SteveIVisual(MarkBase):
    
    def __init__(self, env, cond_scale: float = 7.0, device: Union[str, torch.device] = "cuda", **kwargs):
        self.env = env
        self.kwargs = kwargs

        self.cond_scale = cond_scale
        mineclip_config = MINECLIP_CONFIG
        mineclip_config['ckpt']['path'] = MINECLIP_WEIGHT_PATH

        self.device = device
        self.mineclip = load_mineclip.load(mineclip_config, device=self.device)
        self.agent = make_agent(VPT_MODEL_PATH, VPT_WEIGHT_PATH, cond_scale=self.cond_scale, device=self.device)
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
        prompt_embed = get_visual_embed(condition, self.mineclip)
        
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

    env = MinecraftWrapper('explore_mine')
    env = RenderWrapper(env)
    env.reset()
    
    steveI = SteveIVisual(env=env)
    steveI.reset()

    result, error_message = steveI.do('/scratch/zhangbowei/diverses/explore_mine/human/0.mp4')
    print(result, error_message)