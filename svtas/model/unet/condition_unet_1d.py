import torch
import torch.nn as nn
from .condition_unet import ConditionUnet

from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('model')
class ConditionUnet1D(ConditionUnet):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)