from __future__ import annotations

import torch
import torch.nn as nn

from ..batch import Batch
from mineclip.utils import build_mlp, call_once


class SimpleFeatureFusion(nn.Module):
    def __init__(
        self,
        extractors: dict[str, nn.Module],
        hidden_depth: int,
        output_dim: int,
        device: torch.device,
    ):
        super().__init__()
        self._extractors = nn.ModuleDict(extractors)
        extractors_output_dim = sum(e.output_dim for e in extractors.values())
        self._output_dim = output_dim
        self._head = build_mlp(
            input_dim=extractors_output_dim,
            hidden_dim=output_dim,
            output_dim=output_dim,
            hidden_depth=hidden_depth,
            activation="relu",
            weight_init="orthogonal",
            bias_init="zeros",
            norm_type=None,
            # add input activation because we assume upstream extractors do not have activation at the end
            add_input_activation=True,
            add_input_norm=False,
            add_output_activation=True,
            add_output_norm=False,
        )
        self._device = device

    @property
    def output_dim(self):
        return self._output_dim

    @call_once
    def _check_obs_key_match(self, obs: dict):
        assert set(self._extractors.keys()).issubset(set(obs.keys()))

    def forward(self, x, **kwargs):
        self._check_obs_key_match(x)
        if isinstance(x, Batch):
            x.to_torch(device=self._device)
        x = {k: v.forward(x[k], **kwargs)[0] for k, v in self._extractors.items()}
        x = torch.cat([x[k] for k in sorted(x.keys())], dim=-1)
        x = self._head(x)
        return x, None
