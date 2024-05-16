from svtas.utils import AbstractBuildFactory
from .base_optim import TorchOptimizer
import torch

@AbstractBuildFactory.register('optimizer')
class AdamOptimizer(TorchOptimizer, torch.optim.Adam):
    def __init__(self,
                 model,
                 learning_rate=0.01,
                 betas=(0.9, 0.999),
                 weight_decay=1e-4,
                 finetuning_scale_factor=0.1,
                 no_decay_key = [],
                 finetuning_key = [],
                 freeze_key = [],
                 **kwargs) -> None:
        params = self.get_optim_policies(model, finetuning_key, finetuning_scale_factor, no_decay_key, freeze_key, learning_rate, weight_decay)
        super().__init__(params=params, lr=learning_rate, betas=betas, weight_decay=weight_decay)