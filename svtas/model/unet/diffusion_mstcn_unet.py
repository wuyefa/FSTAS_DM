import copy
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .diffact_unet import get_timestep_embedding
from .condition_unet_1d import ConditionUnet1D
from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('model')
class TASDiffusionConditionUnet(ConditionUnet1D):
    def __init__(self,
                 num_layers,
                 num_f_maps,
                 dim,
                 num_classes,
                 condition_dim,
                 time_embedding_dim,
                 condtion_res_layer_idx = [3, 4, 5, 6],
                 sample_rate = 1,
                 condition_types = ['full', 'zero', 'boundary05-', 'random'],
                 out_feature=False) -> None:
        super().__init__()
        self.out_feature = out_feature
        self.single_condition_stage = SingleStageConditionModel(num_layers,
                                                                num_f_maps,
                                                                dim,
                                                                num_classes,
                                                                condition_dim,
                                                                time_embedding_dim,
                                                                condtion_res_layer_idx,
                                                                out_feature)
        self.sample_rate = sample_rate
        self.time_embedding_dim = time_embedding_dim
        self.condition_types = condition_types

        self.time_in = nn.Sequential(
            torch.nn.Linear(time_embedding_dim, time_embedding_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(time_embedding_dim, time_embedding_dim)
        )

    def generate_binary_matrix_with_ratio(self, input_matrix, ratio_of_ones):
        """
        Generate a binary matrix with the same shape as input_matrix.
        :param input_matrix: torch.Tensor, the input matrix.
        :param ratio_of_ones: float, the ratio of ones in the new matrix.
        :return: torch.Tensor, the generated binary matrix.
        """
        # Check if ratio is valid
        if not (0 <= ratio_of_ones <= 1):
            raise ValueError("ratio_of_ones should be between 0 and 1.")
        
        # Get the shape of the input matrix
        shape = input_matrix.shape
        
        # Calculate the number of ones
        num_elements = torch.prod(torch.tensor(shape)).item()
        num_ones = int(num_elements * ratio_of_ones)
        
        # Create a flat tensor with a specific number of ones
        flat_tensor = torch.cat([torch.ones(num_ones), torch.zeros(num_elements - num_ones)])
        
        # Shuffle the tensor
        flat_tensor = flat_tensor[torch.randperm(num_elements)]
        
        # Reshape to the desired shape
        binary_matrix = flat_tensor.reshape(shape)
        
        # Move the binary matrix to the same device as input_matrix
        binary_matrix = binary_matrix.to(input_matrix.device)
        
        return binary_matrix

    def get_random_label_index(self, labels):
        y = torch.zeros_like(labels)
        refine_labels = torch.where(labels != self.ignore_index, labels, y)
        events = torch.unique(refine_labels, dim=-1)
        random_index = torch.argsort(torch.rand_like(events, dtype=torch.float), dim=-1)
        random_event = torch.gather(events, dim=-1, index=random_index)[:, :1]
        return random_event

    def get_condition_latent_mask(self, condition_latens, boundary_prob, labels):
        cond_type = random.choice(self.condition_types)

        if cond_type == 'full':
            feature_mask = torch.ones_like(condition_latens)
        
        elif cond_type == 'zero':
            feature_mask = torch.zeros_like(condition_latens)
        
        elif cond_type == 'random':
            feature_mask = self.generate_binary_matrix_with_ratio(condition_latens, 0.6)
        
        elif cond_type == 'boundary05-':
            feature_mask = (boundary_prob < 0.5)
            feature_mask = feature_mask.unsqueeze(1).float()[:, :, ::self.sample_rate]
            
        elif cond_type == 'boundary03-':
            feature_mask = (boundary_prob < 0.3)
            feature_mask = feature_mask.unsqueeze(1).float()[:, :, ::self.sample_rate]

        elif cond_type == 'segment=1':
            random_event = self.get_random_label_index(labels=labels)
            feature_mask = (labels != random_event) * (labels != self.ignore_index)
            feature_mask = feature_mask.unsqueeze(1).float()[:, :, ::self.sample_rate]

        elif cond_type == 'segment=2':
            random_event_1 = self.get_random_label_index(labels=labels)
            random_event_2 = self.get_random_label_index(labels=labels)
            while random_event_1 == random_event_2:
                random_event_2 = self.get_random_label_index(labels=labels)
            feature_mask = (labels != random_event_1) * (labels != random_event_2) * (labels != self.ignore_index)
            feature_mask = feature_mask.unsqueeze(1).float()[:, :, ::self.sample_rate]
        else:
            raise Exception('Invalid Cond Type')
        return feature_mask
    
    def forward(self, data_dict):
        # latent
        # timestep
        timestep = data_dict['timestep']
        noise_label = data_dict['noise_label']
        condition_latens = data_dict['condition_latens']
        mask = data_dict['masks_m']

        mask = mask[:, :, ::self.sample_rate]
        time_emb = get_timestep_embedding(timestep, self.time_embedding_dim)
        time_emb = self.time_in(time_emb).unsqueeze(0).permute(0, 2, 1)
        
        if self.out_feature:
            feature, output = self.single_condition_stage(noise_label, condition_latens, time_emb, mask)
        else:
            output = self.single_condition_stage(noise_label, condition_latens, time_emb, mask)
        outputs = output.unsqueeze(0)

        if self.training:
            outputs = F.interpolate(
                input=outputs,
                scale_factor=[1, self.sample_rate],
                mode="nearest")
        
        
        if self.out_feature:
            return dict(output=outputs, feature=feature)
        else:
            return dict(output=outputs)

class SingleStageConditionModel(nn.Module):
    def __init__(self,
                 num_layers,
                 num_f_maps,
                 dim,
                 num_classes,
                 condition_dim,
                 time_embedding_dim,
                 condtion_res_layer_idx = [3, 4, 5, 6],
                 out_feature=False):
        super(SingleStageConditionModel, self).__init__()
        self.out_feature = out_feature
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) if i not in condtion_res_layer_idx else
                                     copy.deepcopy(ConditionDilatedResidualLayer(2 ** i, num_f_maps, num_f_maps, time_embedding_dim, condition_dim))
                                     for i in range(num_layers)])
        self.condtion_res_layer_idx = condtion_res_layer_idx
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, condition_latents, time_embedding, mask):
        feature_embedding = self.conv_1x1(x)
        feature = feature_embedding
        for i, layer in enumerate(self.layers):
            if i in self.condtion_res_layer_idx:
                feature = layer(feature, condition_latents, time_embedding, mask)
            else:
                feature = layer(feature, mask)
        out = self.conv_out(feature) * mask[:, 0:1, :]
        if self.out_feature is True:
            return feature_embedding * mask[:, 0:1, :], out

        return out

class ConditionDilatedResidualLayer(nn.Module):
    def __init__(self,
                 dilation,
                 in_channels,
                 out_channels,
                 time_embedding_dim,
                 condition_dim,
                 ):
        super(ConditionDilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        
        self.condition_embedding = nn.Conv1d(condition_dim, out_channels, 1)
        
        self.time_embedding = nn.Conv1d(time_embedding_dim, out_channels, 1)
        
        self.swish = nn.SiLU()
        self.norm = nn.Dropout() # 空的
        self.condition_types = ['full', 'random', 'boundary03-']

    def forward(self, x, condition_latents, time_embedding, mask):
        x = x + self.norm(self.condition_embedding(condition_latents)) + self.time_embedding(self.swish(time_embedding)).permute(2, 1, 0)
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.norm(out)
        return (x + out) * mask[:, 0:1, :]

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.norm = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.norm(out)
        return (x + out) * mask[:, 0:1, :]


class MixedConvAttentionLayerV2(nn.Module):
    
    def __init__(self, d_model, d_cross, kernel_size, dilation, dropout_rate):
        super(MixedConvAttentionLayerV2, self).__init__()
        
        self.d_model = d_model
        self.d_cross = d_cross
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout_rate = dropout_rate
        self.padding = (self.kernel_size // 2) * self.dilation 
        
        assert(self.kernel_size % 2 == 1)

        self.conv_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=self.padding, dilation=dilation),
            # nn.ReLU(),
        )

        self.att_linear_q = nn.Conv1d(d_model + d_cross, d_model, 1)
        self.att_linear_k = nn.Conv1d(d_model + d_cross, d_model, 1)
        self.att_linear_v = nn.Conv1d(d_model, d_model, 1)

        self.ffn_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 1),
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.InstanceNorm1d(d_model, track_running_stats=False)

        self.attn_indices = None


    def get_attn_indices(self, l, device):
            
        attn_indices = []
                
        for q in range(l):
            s = q - self.padding
            e = q + self.padding + 1
            step = max(self.dilation // 1, 1)  
            # 1  2  4   8  16  32  64  128  256  512  # self.dilation
            # 1  1  1   2  4   8   16   32   64  128  # max(self.dilation // 4, 1)  
            # 3  3  3 ...                             (k=3, //1)          
            # 3  5  5  ....                           (k=3, //2)
            # 3  5  9   9 ...                         (k=3, //4)
                        
            indices = [i + self.padding for i in range(s,e,step)]

            attn_indices.append(indices)
        
        attn_indices = np.array(attn_indices)
            
        self.attn_indices = torch.from_numpy(attn_indices).long()
        self.attn_indices = self.attn_indices.to(device)
        
        
    def attention(self, x, x_cross):
        
        if self.attn_indices is None:
            self.get_attn_indices(x.shape[2], x.device)
        else:
            if self.attn_indices.shape[0] < x.shape[2]:
                self.get_attn_indices(x.shape[2], x.device)
                                
        flat_indicies = torch.reshape(self.attn_indices[:x.shape[2],:], (-1,))
        
        x_q = self.att_linear_q(torch.cat([x, x_cross], 1))
        x_k = self.att_linear_k(torch.cat([x, x_cross], 1))
        x_v = self.att_linear_v(x)

        x_k = torch.index_select(
            F.pad(x_k, (self.padding, self.padding), 'constant', 0),
            2, flat_indicies)  
        x_v = torch.index_select(
            F.pad(x_v, (self.padding, self.padding), 'constant', 0), 
            2, flat_indicies)  
                        
        x_k = torch.reshape(x_k, (x_k.shape[0], x_k.shape[1], x_q.shape[2], self.attn_indices.shape[1]))
        x_v = torch.reshape(x_v, (x_v.shape[0], x_v.shape[1], x_q.shape[2], self.attn_indices.shape[1])) 
        
        att = torch.einsum('n c l, n c l k -> n l k', x_q, x_k)
        
        padding_mask = torch.logical_and(
            self.attn_indices[:x.shape[2],:] >= self.padding,
            self.attn_indices[:x.shape[2],:] < att.shape[1] + self.padding
        ) # 1 keep, 0 mask
        
        att = att / np.sqrt(self.d_model)
        att = att + torch.log(padding_mask + 1e-6)
        att = F.softmax(att, 2)
        att = att * padding_mask

        r = torch.einsum('n l k, n c l k -> n c l', att, x_v)
        
        return r
    
                
    def forward(self, x, x_cross):
        
        x_drop = self.dropout(x)
        x_cross_drop = self.dropout(x_cross)

        # out1 = self.conv_block(x_drop)
        out2 = self.attention(x_drop, x_cross_drop)
        # out2 = self.attention(x, x_cross)
                
        # out = self.ffn_block(self.norm(out1 + out2))
        out = self.ffn_block(self.norm(out2))
        # return x + out
        return out