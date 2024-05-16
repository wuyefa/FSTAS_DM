from typing import Dict
from svtas.model_pipline.wrapper import TorchBaseModel

class BaseLoss(TorchBaseModel):
    def __init__(self) -> None:
        super().__init__()
        
    def init_weights(self, init_cfg: Dict = ...):
        pass