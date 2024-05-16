from svtas.utils import AbstractBuildFactory
import torch
from .base_lr_scheduler import TorchLRScheduler

@AbstractBuildFactory.register('lr_scheduler')
class CosineAnnealingLR(TorchLRScheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
    def __init__(self,
                 optimizer,
                 T_max,
                 eta_min=0,
                 last_epoch=- 1,
                 verbose=False) -> None:
        super().__init__(optimizer=optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch, verbose=verbose)