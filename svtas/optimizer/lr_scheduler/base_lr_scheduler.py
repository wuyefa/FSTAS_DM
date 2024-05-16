import abc
from torch.optim.lr_scheduler import _LRScheduler

class BaseLRScheduler(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_lr(self):
        pass
    
    @abc.abstractmethod
    def step(self):
        pass

class TorchLRScheduler(_LRScheduler, BaseLRScheduler):
    pass