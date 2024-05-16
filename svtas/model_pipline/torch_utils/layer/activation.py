import torch
import torch.nn as nn
import torch.nn.functional as F
from svtas.utils import AbstractBuildFactory
from svtas.utils.package_utils import digit_version

for module in [
        nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.RReLU, nn.ReLU6, nn.ELU,
        nn.Sigmoid, nn.Tanh, nn.GELU
]:
    AbstractBuildFactory.register_obj(obj=module, registory_name='model')

if digit_version(torch.__version__) >= digit_version('1.7.0'):
    AbstractBuildFactory.register_obj(obj=nn.SiLU, registory_name='model', obj_name='SiLU')
else:

    class SiLU(nn.Module):
        """Sigmoid Weighted Liner Unit."""

        def __init__(self, inplace=False):
            super().__init__()
            self.inplace = inplace

        def forward(self, inputs) -> torch.Tensor:
            if self.inplace:
                return inputs.mul_(torch.sigmoid(inputs))
            else:
                return inputs * torch.sigmoid(inputs)

    AbstractBuildFactory.register_obj(obj=SiLU, registory_name='model', obj_name='SiLU')
