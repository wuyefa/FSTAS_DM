import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from svtas.utils.logger import get_logger
from svtas.utils import AbstractBuildFactory
from ..general import SeriousModel

@AbstractBuildFactory.register('model')
class FeatureSegmentation(SeriousModel):
    backbone: nn.Module
    neck: nn.Module
    head: nn.Module
    def __init__(self,
                 architecture_type='1d',
                 backbone=None,
                 neck=None,
                 head=None,
                 weight_init_cfg=dict(
                    backbone=dict(
                    child_model=True)),
                 location_embedding=False):
        assert architecture_type in ['1d', '3d'], f'Unsupport architecture_type: {architecture_type}!'
        super().__init__(weight_init_cfg=weight_init_cfg, backbone=backbone, neck=neck, head=head)
        self.sample_rate = head.sample_rate
        self.architecture_type = architecture_type
        self.location_embedding = location_embedding
        time_emb_dim = 2048

    def preprocessing(self, input_data):
        masks = input_data['masks'].unsqueeze(1)
        input_data['masks_m'] = masks
        
        if self.backbone is not None:
            input_data['backbone_masks'] = torch.reshape(masks[:, :, ::self.sample_rate], [-1]).unsqueeze(-1)
            
        if self.architecture_type == '1d':
            feature = torch.permute(input_data['feature'], dims=[0, 2, 1]).contiguous()
            input_data['feature_m'] = feature
        elif self.architecture_type == '3d':
            pass
        return input_data
    
    def get_timestep_embedding(self, timesteps, embedding_dim): # for diffusion model
        # timesteps: batch,
        # out:       batch, embedding_dim
        """
        This matches the implementation in Denoising DiffusionModel Probabilistic Models:
        From Fairseq.
        Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=timesteps.device)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0,1,0,0))
        return emb
    
    def forward(self, input_data):
        input_data = self.preprocessing(input_data)
        masks = input_data['masks_m']
        feature = input_data['feature_m']
        if self.location_embedding:
            start_idx = input_data['start_idx']
            end_idx = input_data['end_idx']
            start_embedding = self.get_timestep_embedding(start_idx, 2048)
            end_embedding = self.get_timestep_embedding(end_idx, 2048)
            start_embedding = self.time_in_start(start_embedding)[:, :, None]
            end_embedding = self.time_in_end(end_embedding)[:, :, None]
            feature = feature + start_embedding + end_embedding
            

        if self.backbone is not None:
             # masks.shape [N C T]
            backbone_masks = input_data['backbone_masks']
            feature = self.backbone(feature, backbone_masks)
        else:
            feature = feature

        # feature [N, F_dim, T]
        # step 3 extract memory feature
        if self.neck is not None:
            seg_feature = self.neck(
                feature, masks)
            
        else:
            seg_feature = feature

        # step 5 segmentation
        # seg_feature [N, H_dim, T]
        # cls_feature [N, F_dim, T]
        if self.head is not None:
            head_score = self.head(seg_feature, masks)
        else:
            head_score = None
        # seg_score [stage_num, N, C, T]
        return {"output":head_score}