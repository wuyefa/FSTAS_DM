import abc
from typing import Any, Dict

class BaseModel(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        self.train()

    @property
    def training(self):
        return self._training
    
    @training.setter
    def training(self, val: bool):
        self._training = val
    
    def train(self, val: bool = True):
        self._training = val

    def eval(self):
        self.model
        self._training = False

    @abc.abstractmethod
    def _clear_memory_buffer(self):
        pass
    
    @abc.abstractmethod
    def init_weights(self, init_cfg: Dict = {}):
        pass
    
    @abc.abstractmethod
    def run_train(*args: Any, **kwds: Any):
        raise NotImplementedError("You must implement train function!")
    
    @abc.abstractmethod
    def run_test(*args: Any, **kwds: Any):
        raise NotImplementedError("You must implement infer function!")
    
    @abc.abstractmethod
    def forward(self, *args: Any, **kwds: Any) -> Any:
        pass
    
    def reset_state(self):
        pass