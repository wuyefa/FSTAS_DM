import torch
import copy
import torch.nn as nn
import torch.nn.functional as F

from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('model')
class MultiFlowMultiStageModel(nn.Module):
    def __init__(self,
                 kernel_size1,
                 kernel_size2,
                 num_stages,
                 num_layers,
                 num_f_maps,
                 dim,
                 num_classes,
                 sample_rate=1,
                 out_feature=False):
        super(MultiFlowMultiStageModel, self).__init__()
        self.sample_rate = sample_rate
        self.out_feature = out_feature
        self.stage1 = SingleStageModel(kernel_size1, kernel_size2, num_layers, num_f_maps, dim, num_classes, out_feature=out_feature)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(kernel_size1, kernel_size2, num_layers, num_f_maps, num_classes, num_classes, out_feature)) for s in range(num_stages-1)])

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
            # return feature, outputs
            return {'output':outputs, 'output_feature':feature}
        # return outputs
        return {'output':outputs}


class SingleStageModel(nn.Module):
    def __init__(self, kernel_size1, kernel_size2, num_layers, num_f_maps, dim, num_classes, out_feature=False):
        super(SingleStageModel, self).__init__()
        self.out_feature = out_feature
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        # self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(kernel_size, 2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.layers_1 = nn.ModuleList()
        self.layers_2 = nn.ModuleList()
        for i in range(num_layers):
            if i <= 5:
                self.layers_1.append(copy.deepcopy(DilatedResidualLayer(kernel_size1, 2 ** i, num_f_maps, num_f_maps)))
                self.layers_2.append(copy.deepcopy(DilatedResidualLayer(kernel_size2, 2 ** i, num_f_maps, num_f_maps)))
            else:
                self.layers_1.append(copy.deepcopy(DilatedResidualLayer(kernel_size1, (2 ** 5 + i * 10), num_f_maps, num_f_maps)))
                self.layers_2.append(copy.deepcopy(DilatedResidualLayer(kernel_size2, (2 ** 5 + i * 10), num_f_maps, num_f_maps)))
        
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        feature_embedding = self.conv_1x1(x)
        feature = feature_embedding
        for layer1, layer2 in zip(self.layers_1, self.layers_2):
            feature1 = layer1(feature, mask)
            feature2 = layer2(feature, mask)
            feature = 0.5 * feature1 + 0.5* feature2
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
        self.norm = nn.Dropout() # 空的

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.norm(out)
        return (x + out) * mask[:, 0:1, :]


if __name__ == "__main__":
    model = MultiFlowMultiStageModel(
                            kernel_size1 = 5,
                            kernel_size2 = 3,
                            num_stages = 4, # 4
                            num_layers = 10,
                            num_f_maps = 128, # 128
                            dim = 1024, # 512
                            num_classes = 19,
                            sample_rate = 16,)
    data = torch.randn(1, 1024, 128)
    mask = torch.randn(1, 1, 2048)

    output = model(data, mask)
    print(output.shape)