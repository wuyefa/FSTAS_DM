from .standalone_engine import StandaloneEngine
from svtas.loader.dataloader import BaseDataloader
from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('engine')
class TorchDistributedDataParallelEngine(StandaloneEngine):
    def run(self):
        for epoch in self.iter_method.run():
            if self.model_pipline.local_rank <= 0:
                if self.running_mode in ['train']:
                    self.save(file_name = self.model_name + f"_epoch_{epoch + 1:05d}")
                elif self.running_mode in ['validation']:
                    self.save(file_name = self.model_name + "_best")