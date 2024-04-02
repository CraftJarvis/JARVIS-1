from __future__ import annotations

import torch.nn as nn

from .batch import Batch


class MineAgent(nn.Module):
    def __init__(
        self,
        actor: nn.Module,
        deterministic_eval: bool = True,
    ):
        super().__init__()
        self.actor = actor
        self._deterministic_eval = deterministic_eval
        self.dist_fn = actor.dist_fn

    def forward(
        self,
        batch: Batch,
        state=None,
        **kwargs,
    ) -> Batch:
        logits, hidden = self.actor(batch.obs, state=state)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        if self._deterministic_eval and not self.training:
            act = dist.mode()
        else:
            act = dist.sample()
        return Batch(logits=logits, act=act, state=hidden, dist=dist)
