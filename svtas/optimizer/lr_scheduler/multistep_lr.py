from svtas.utils import AbstractBuildFactory
import torch
from .base_lr_scheduler import TorchLRScheduler

@AbstractBuildFactory.register('lr_scheduler')
class MultiStepLR(TorchLRScheduler, torch.optim.lr_scheduler.MultiStepLR):
    def __init__(self,
                 optimizer,
                 step_size=[10, 30],
                 gamma=0.1) -> None:
        super().__init__(optimizer=optimizer, milestones=step_size, gamma=gamma)