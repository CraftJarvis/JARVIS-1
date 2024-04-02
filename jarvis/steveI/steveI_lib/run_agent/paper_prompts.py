import os

from jarvis.steveI.steveI_lib.config import DEVICE, PRIOR_INFO
from jarvis.steveI.steveI_lib.data.text_alignment.vae import load_vae_model
from jarvis.steveI.steveI_lib.utils.embed_utils import get_prior_embed
from jarvis.steveI.steveI_lib.utils.file_utils import load_pickle
from jarvis.steveI.steveI_lib.utils.mineclip_agent_env_utils import load_mineclip_wconfig

VISUAL_PROMPTS_PKL_DIRPATH = 'data/visual_prompt_embeds/'
TEXT_PROMPTS = {
    'dig': 'dig as far as possible',
    'dirt': 'get dirt, dig hole, dig dirt, gather a ton of dirt, collect dirt',
    'sky': 'look at the sky',
    'leaves': 'break leaves',
    'wood': 'chop down the tree, gather wood, pick up wood, chop it down, break tree',
    'seeds': 'break tall grass, break grass, collect seeds, punch the ground, run around in circles getting seeds from bushes',
    'flower': 'break a flower',
    'explore': 'go explore',
    'swim': 'go swimming',
    'underwater': 'go underwater',
    'inventory': 'open inventory',
}


def load_text_prompt_embeds():
    """Load the text prompt embeds. Returns a dict mapping prompt name to prompt embed.
    Prompt names are: dig, dirt, sky, leaves, wood, seeds, flower, explore, swim, underwater, inventory.
    """
    mineclip = load_mineclip_wconfig()
    prior = load_vae_model(PRIOR_INFO)
    prompts = {}
    for prompt_name in TEXT_PROMPTS:
        text_prompt_embed = get_prior_embed(TEXT_PROMPTS[prompt_name], mineclip, prior, DEVICE)
        prompts[prompt_name] = text_prompt_embed
    return prompts


def load_visual_prompt_embeds():
    """Load the visual prompt embeds. Returns a dict mapping prompt name to prompt embed.
    Prompt names are: dig, dirt, sky, leaves, wood, seeds, flower, explore, swim, underwater, inventory.
    """
    prompts = {}
    for prompt_name in TEXT_PROMPTS:
        visual_prompt_embed = load_pickle(os.path.join(VISUAL_PROMPTS_PKL_DIRPATH, prompt_name + '.pkl'))
        prompts[prompt_name] = visual_prompt_embed
    return prompts
