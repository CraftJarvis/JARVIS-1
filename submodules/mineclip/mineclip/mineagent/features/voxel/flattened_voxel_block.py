import torch
import torch.nn as nn

from mineclip.utils import build_mlp
import mineclip.utils as U


class FlattenedVoxelBlockEncoder(nn.Module):
    def __init__(
        self,
        *,
        n_blocks: int = 27,
        embed_dim: int,
        hidden_dim: int,
        output_dim: int,
        hidden_depth: int,
        device: torch.device,
        learn_padding: bool = False,
    ):
        super().__init__()
        self._embed = nn.Embedding(
            NUM_TYPES, embed_dim, padding_idx=None if learn_padding else 0
        )
        self._mlp = build_mlp(
            input_dim=n_blocks * embed_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            hidden_depth=hidden_depth,
        )
        self._output_dim = output_dim
        self._device = device

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, x, **kwargs):
        x = U.any_to_torch_tensor(x, device=self._device)
        x = self._embed(x)
        x = x.view(x.shape[:-2] + (-1,))
        x = self._mlp(x)
        return x, None


VOXEL_BLOCK_NAME_MAP = {
    "obsidian": 1,
    "portal": 1,
    "tnt": 2,
    "melon": 3,
    "sugar cane": 4,
    "crops": 5,
    "carrot": 6,
    "fence": 7,
    "water": 8,
    "air": 9,
    # "grass block": 10,
    # "farmland": 11,
    "oak fence": 12,
    "lava": 13,
    "wood": 14,
    "pumpkin": 15,
    "oak sapling": 16,
    "carpet": 17,
    # ------ below are placeholders, so we can have an embedding layer with consistent size ------
    "placeholder_7": 18,
    "placeholder_8": 19,
    "placeholder_9": 20,
    "placeholder_10": 21,
    "placeholder_11": 22,
    "placeholder_12": 23,
    "placeholder_13": 24,
    "placeholder_14": 25,
    "placeholder_15": 26,
    "placeholder_16": 27,
    "placeholder_17": 28,
    "placeholder_18": 29,
    "placeholder_19": 30,
    "placeholder_20": 31,
}
MAX_IDX = max(list(VOXEL_BLOCK_NAME_MAP.values()))
NUM_TYPES = MAX_IDX + 1
