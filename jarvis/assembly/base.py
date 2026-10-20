from jarvis.utils import write_video

import os
from openai import OpenAI

# MiniMax exposes an OpenAI-compatible API. The global and mainland-China
# endpoints share the same request/response schema but live on different hosts,
# so the client is configured by region instead of being hardcoded.
MINIMAX_BASE_URLS = {
    "global_en": "https://api.minimax.io/v1",
    "cn_zh": "https://api.minimax.cn/v1",
}
DEFAULT_REGION = "global_en"

# Current MiniMax text models; explicit legacy/custom model IDs remain valid.
MINIMAX_MODELS = ["MiniMax-M3.1-Flash", "MiniMax-M3"]
DEFAULT_MODEL = MINIMAX_MODELS[0]


def uses_minimax():
    """Select MiniMax only when its configuration is explicitly supplied."""
    return any(os.environ.get(name) for name in (
        "MINIMAX_API_KEY", "MINIMAX_BASE_URL", "MINIMAX_REGION", "MINIMAX_MODEL",
    ))


def get_base_url():
    """Resolve the MiniMax OpenAI-compatible base URL.

    ``MINIMAX_BASE_URL`` takes precedence when set; otherwise ``MINIMAX_REGION``
    selects between the global (``global_en``) and mainland-China (``cn_zh``)
    endpoints, defaulting to the global endpoint.
    """
    base_url = os.environ.get("MINIMAX_BASE_URL")
    if base_url:
        return base_url
    region = os.environ.get("MINIMAX_REGION", DEFAULT_REGION)
    return MINIMAX_BASE_URLS.get(region, MINIMAX_BASE_URLS[DEFAULT_REGION])


def get_model():
    """Resolve the selected provider's model without rewriting explicit IDs."""
    if uses_minimax():
        return os.environ.get("MINIMAX_MODEL", DEFAULT_MODEL)
    return os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")


def build_client():
    """Create the selected client without sharing credentials across providers."""
    if not uses_minimax():
        return OpenAI()
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise ValueError("MINIMAX_API_KEY is required for MiniMax configuration")
    return OpenAI(api_key=api_key, base_url=get_base_url())


client = build_client()

import json
from jarvis.assets import TASKS_FILE, TAG_ITEMS_FILE, SPAWN_FILE, SKILL_FILE
with open(TASKS_FILE, 'r') as f:
    jarvis_tasks = json.load(f)

def get_task_config(task_name):
    task_dict = {}
    for task in jarvis_tasks:
        if task['task'] == task_name:
            task_dict = task
            return task_dict
    return {}

with open(TAG_ITEMS_FILE, 'r') as f:
    tag_items = json.load(f)

with open(SKILL_FILE, 'r') as f:
    skills = json.load(f)

from jarvis.assets import MEMORY_FILE
with open(MEMORY_FILE, 'r') as f:
    memory = json.load(f)

class MarkBase:
    
    def __init__(self, **kwargs):
        pass
        
    def reset(self):
        self.record_frames = []
        self.record_infos = []
    
    def do(self):
        raise NotImplementedError

    def record_step(self):
        record_frames = getattr(self, 'record_frames', [])
        record_infos = getattr(self, 'record_infos', [])
        self.record_frames = record_frames + [self.info['pov']]
        self.record_infos = record_infos + [self.info]

    def make_traj_video(self, output_path: str):
        if getattr(self, 'record_frames', None) is None:
            return
        write_video(self.record_frames, output_path)



