import torch
import torch.nn as nn

from mineclip.utils import build_mlp
import mineclip.utils as U


class CompassMLP(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 8,
        hidden_dim: int,
        output_dim: int,
        hidden_depth: int,
        device: torch.device
    ):
        super().__init__()
        self._mlp = build_mlp(
            input_dim=input_dim,
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
        return self._mlp(x), None
