import torch
import torch.nn as nn
import torch.nn.functional as F

from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('model')
class IdentityEmbeddingHead(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 sample_rate=4):
        super().__init__()
        self.sample_rate = sample_rate
        if in_channels != out_channels:
            self.project = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.project = None
    
    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        pass
    
    def forward(self, x, masks):
        if self.project is not None:
            x = self.project(x)
        x = x * masks[:, 0:1, ::self.sample_rate]
        return x