import torch
import torch.nn as nn

from mineclip.utils import build_mlp
import mineclip.utils as U


class GPSMLP(nn.Module):
    def __init__(
        self, hidden_dim: int, output_dim: int, hidden_depth: int, device: torch.device
    ):
        super().__init__()
        self._mlp = build_mlp(
            input_dim=3,
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
        # divide by [1000, 100, 1000] because x and z tend to have larger values (several hundreds)
        # but y tends to be smaller (e.g., around 60)
        x = U.any_to_torch_tensor(x, device=self._device) / torch.tensor(
            [1000, 100, 1000], device=self._device, dtype=torch.float32
        )
        return self._mlp(x), None
