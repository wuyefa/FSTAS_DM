import torch
import torch.nn as nn

class ConditionUnet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def init_weights(self, init_cfg: dict = {}):
        pass
    
    def _clear_memory_buffer(self):
       pass