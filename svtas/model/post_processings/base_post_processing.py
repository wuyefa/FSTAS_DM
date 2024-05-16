import abc
from typing import List

class BasePostProcessing(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        self.init_flag = False

    @abc.abstractmethod
    def init_scores(self, sliding_num, batch_size):
        raise NotImplementedError("You must implement init_scores function!")
    
    @abc.abstractmethod
    def update(self, seg_scores, gt, idx):
        raise NotImplementedError("You must implement update function!")
    
    @abc.abstractmethod
    def output(self) -> List:
        raise NotImplementedError("You must implement output function!")