# Copyright (c) OpenMMLab. All rights reserved.
from svtas.utils import AbstractBuildFactory
from .resnet_3d import ResNet3d, BasicBlock3d, Bottleneck3d
import torch.nn as nn

from svtas.model_pipline.torch_utils import load_checkpoint
from svtas.utils.logger import get_logger
from svtas.model_pipline.torch_utils import constant_init, kaiming_init


@AbstractBuildFactory.register('model')
class ResNet2Plus1d(ResNet3d):
    """ResNet (2+1)d backbone.
    This model is proposed in `A Closer Look at Spatiotemporal Convolutions for
    Action Recognition <https://arxiv.org/abs/1711.11248>`_
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.pretrained2d is False
        assert self.conv_cfg['type'] == 'Conv2plus1d'

    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
    
    def _clear_memory_buffer(self):
        pass

    def init_weights(self, child_model=False, revise_keys=[(r'backbone.', r'')]):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
            
        if child_model is False:
            if isinstance(self.pretrained, str):
                logger = get_logger("SVTAS")
                if self.pretrained2d:
                    # Inflate 2D model into 3D model.
                    self.inflate_weights(logger)

                else:
                    # Directly load 3D model.
                    load_checkpoint(
                        self, self.pretrained, strict=False, logger=logger, revise_keys=revise_keys)

            elif self.pretrained is None:
                for m in self.modules():
                    if isinstance(m, nn.Conv3d):
                        kaiming_init(m)
                    elif isinstance(m, nn.BatchNorm3d):
                        constant_init(m, 1)

                if self.zero_init_residual:
                    for m in self.modules():
                        if isinstance(m, Bottleneck3d):
                            constant_init(m.conv3.bn, 0)
                        elif isinstance(m, BasicBlock3d):
                            constant_init(m.conv2.bn, 0)
            else:
                raise TypeError('pretrained must be a str or None')
        else:
            for m in self.modules():
                    if isinstance(m, nn.Conv3d):
                        kaiming_init(m)
                    elif isinstance(m, nn.BatchNorm3d):
                        constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck3d):
                        constant_init(m.conv3.bn, 0)
                    elif isinstance(m, BasicBlock3d):
                        constant_init(m.conv2.bn, 0)

    def forward(self, x, masks):
        """Defines the computation performed at every call.
        Args:
            x (torch.Tensor): The input data.
        Returns:
            torch.Tensor: The feature of the input
            samples extracted by the backbone.
        """
        x = self.conv1(x)
        x = self.maxpool(x)
        for layer_name in self.res_layers:
            res_layer = getattr(self, layer_name)
            # no pool2 in R(2+1)d
            x = res_layer(x)

        return x * masks