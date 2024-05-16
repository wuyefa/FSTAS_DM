import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from svtas.model_pipline.torch_utils import constant_init, kaiming_init
import numpy as np

from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('model')
class PhaseChangeModel(nn.Module):
    def __init__(self,
                 num_stages,
                 num_layers,
                 num_f_maps,
                 dim,
                 num_classes,
                 kernel_size=3,
                 sample_rate=1,
                 exponential_boundary=100,
                 deta_dilation=0,
                 out_feature=False,
                 out_dict=False):
        super(PhaseChangeModel, self).__init__()
        self.sample_rate = sample_rate
        self.out_feature = out_feature
        self.out_dict = out_dict
        self.stage1 = SingleStageModel(exponential_boundary, deta_dilation, kernel_size, num_layers, num_f_maps, dim, num_classes, out_feature=out_feature)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(exponential_boundary, deta_dilation, kernel_size, num_layers, num_f_maps, num_classes, num_classes, out_feature)) for s in range(num_stages-1)])

    def init_weights(self):
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         kaiming_init(m)
        #     elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
        #         constant_init(m, 1)
        pass

    def _clear_memory_buffer(self):
        pass

    def forward(self, x, mask):
        mask = mask[:, :, ::self.sample_rate]
        
        output = self.stage1(x, mask)

        if self.out_feature is True:
            feature, out = output
        else:
            out = output

        outputs = out.unsqueeze(0)
        for s in self.stages:
            if self.out_feature is True:
                feature, out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            else:
                out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        
        outputs = F.interpolate(
            input=outputs,
            scale_factor=[1, self.sample_rate],
            mode="nearest")
        
        if self.out_feature is True:
            if not self.out_dict:
                return feature, outputs
            else:
                return {'output':outputs, 'output_feature':feature}
        if not self.out_dict:
            return outputs
        else:
            return {'output':outputs}

@AbstractBuildFactory.register('model')
class SingleStageModel(nn.Module):
    def __init__(self, exponential_boundary, deta_dilation, kernel_size, num_layers, num_f_maps, dim, num_classes, out_feature=False):
        super(SingleStageModel, self).__init__()
        self.out_feature = out_feature
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i <= exponential_boundary:
                self.layers.append(copy.deepcopy(DilatedResidualLayer(kernel_size, 2 ** i, num_f_maps, num_f_maps)))
            else:
                self.layers.append(copy.deepcopy(DilatedResidualLayer(kernel_size, (2 ** exponential_boundary + i * deta_dilation), num_f_maps, num_f_maps)))
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        feature_embedding = self.conv_1x1(x)
        feature = feature_embedding
        for layer in self.layers:
            feature = layer(feature, mask)
        out = self.conv_out(feature) * mask[:, 0:1, :]
        if self.out_feature is True:
            return feature_embedding * mask[:, 0:1, :], out

        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, kernel_size, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        # self.norm = nn.BatchNorm1d(out_channels)
        self.norm = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.norm(out)
        return (x + out) * mask[:, 0:1, :]