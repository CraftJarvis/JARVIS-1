import torch

from jarvis.steveI.steveI_lib.data.EpisodeStorage import EpisodeStorage


def get_prior_embed(text, mineclip, prior, device):
    """Get the embed processed by the prior."""
    with torch.cuda.amp.autocast():
        text_embed = mineclip.encode_text(text).detach().cpu().numpy()
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_prompt_embed = prior(torch.tensor(text_embed).float().to(device)).cpu().detach().numpy()
    return text_prompt_embed


def get_visual_embed_from_episode(episode_dirpath, timestep):
    """Get the visual embed at the given timestep from the given episode in the dataset. Episode must have been
    saved with EpisodeStorage format (this is how the dataset generation code saves episodes).
    """
    episode = EpisodeStorage(episode_dirpath)
    visual_embeds = episode.load_embeds_attn()
    visual_embed = visual_embeds[timestep]
    return visual_embed
