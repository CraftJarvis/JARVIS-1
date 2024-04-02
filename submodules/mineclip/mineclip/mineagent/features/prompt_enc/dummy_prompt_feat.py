"""
Note that prompt feature is provided by MineCLIP.
"""
import torch.nn as nn


class PromptEmbFeat(nn.Module):
    def __init__(self, output_dim: int, device):
        super().__init__()
        self._output_dim = output_dim

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, x, **kwargs):
        return x, None
