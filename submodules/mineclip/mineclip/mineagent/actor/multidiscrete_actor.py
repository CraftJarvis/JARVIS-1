from __future__ import annotations

import torch
import torch.nn as nn

from mineclip.utils import build_mlp
from .distribution import MultiCategorical


class MultiCategoricalActor(nn.Module):
    def __init__(
        self,
        preprocess_net: nn.Module,
        *,
        action_dim: list[int],
        hidden_dim: int,
        hidden_depth: int,
        activation: str = "relu",
        device,
    ):
        super().__init__()
        self.mlps = nn.ModuleList()
        self.preprocess = preprocess_net
        for action in action_dim:
            net = build_mlp(
                input_dim=preprocess_net.output_dim,
                output_dim=action,
                hidden_dim=hidden_dim,
                hidden_depth=hidden_depth,
                activation=activation,
                norm_type=None,
            )
            self.mlps.append(net)
        self._action_dim = action_dim
        self._device = device

    def forward(self, x, state=None, info=None):
        hidden = None
        x, _ = self.preprocess(x)
        return torch.cat([mlp(x) for mlp in self.mlps], dim=1), hidden

    @property
    def dist_fn(self):
        return lambda x: MultiCategorical(logits=x, action_dims=self._action_dim)
