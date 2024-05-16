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

class Prediction_Generation(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Prediction_Generation, self).__init__()

        self.num_layers = num_layers

        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        self.conv_dilated_1 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**(num_layers-1-i), dilation=2**(num_layers-1-i))
            for i in range(num_layers)
        ))

        self.conv_dilated_2 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**i, dilation=2**i)
            for i in range(num_layers)
        ))

        self.conv_fusion = nn.ModuleList((
             nn.Conv1d(2*num_f_maps, num_f_maps, 1)
             for i in range(num_layers)

            ))


        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        f = self.conv_1x1_in(x)

        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in

        # out = self.conv_out(f)
        out = f

        return out

@AbstractBuildFactory.register('model')
class TASDiffusionConditionUnetV2(ConditionUnet1D):
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
        self.PG = Prediction_Generation(11, num_f_maps, 128, num_classes)
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
        # labels_onehot = data_dict['labels_onehot'] # 把标签取出来
        # labels = data_dict['labels'] # 把标签取出来
        
        # if self.training:
        #     gt_labels = data_dict['labels']
        #     boundary_prob = data_dict['boundary_prob']
        #     feature_mask = self.get_condition_latent_mask(condition_latens=condition_latens,
        #                                                   labels=gt_labels,
        #                                                   boundary_prob=boundary_prob)
        #     condition_latens = condition_latens * feature_mask

        mask = mask[:, :, ::self.sample_rate]
        # noise_label = noise_label[:, :, ::self.sample_rate]
        time_emb = get_timestep_embedding(timestep, self.time_embedding_dim)
        time_emb = self.time_in(time_emb).unsqueeze(0).permute(0, 2, 1)
        
        condition_latens = self.PG(condition_latens)
        
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
        
        # outputs_index = torch.argmax(outputs.squeeze(0), dim=1).squeeze(1)
        # error_label = (outputs_index == labels).to(torch.int)
        
        
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
        # self.norm = nn.BatchNorm1d(out_channels)
        self.norm = nn.Dropout()
        self.condition_types = ['full', 'random', 'boundary03-']

    def forward(self, x, condition_latents, time_embedding, mask):
        x = x + self.norm(self.condition_embedding(condition_latents)) + self.time_embedding(self.swish(time_embedding))
        # x = x + self.condition_embedding(condition_latents) + self.time_embedding(self.swish(time_embedding)) # without dropout
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.norm(out)
        return (x + out) * mask[:, 0:1, :]

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        # self.norm = nn.BatchNorm1d(out_channels)
        self.norm = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.norm(out)
        return (x + out) * mask[:, 0:1, :]