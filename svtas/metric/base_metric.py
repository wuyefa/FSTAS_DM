import abc

class BaseMetric(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def update(self, outputs):
        """update metrics during each iter
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        raise NotImplementedError