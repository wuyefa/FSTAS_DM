import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from svtas.utils import AbstractBuildFactory
from .base_loss import BaseLoss

@AbstractBuildFactory.register('loss')
class TASDiffusionStreamSegmentationLoss(BaseLoss):
    def __init__(self,
                 unet_loss_cfg,
                 vae_loss_cfg: Dict = None,
                 prompt_net_loss_cfg: Dict = None,
                 control_net_loss_cfg: Dict = None,
                 prompt_backbone_loss_cfg = None):
        super().__init__()
        self.elps = 1e-10

        self.unet_loss = AbstractBuildFactory.create_factory('loss').create(unet_loss_cfg)
        self.vae_loss = AbstractBuildFactory.create_factory('loss').create(vae_loss_cfg)
        self.prompt_net_loss= AbstractBuildFactory.create_factory('loss').create(prompt_net_loss_cfg)
        self.control_net_loss = AbstractBuildFactory.create_factory('loss').create(control_net_loss_cfg)
        self.prompt_backbone_loss= AbstractBuildFactory.create_factory('loss').create(prompt_backbone_loss_cfg)

    def forward(self, model_output, input_data):
        noise, pred_labels, backbone_score, prompt_backbone_score = model_output["noise"], model_output["output"], model_output["backbone_score"], model_output["prompt_backbone_score"] # 增加prompt的backbone的loss
        masks, labels, precise_sliding_num = input_data["masks"], input_data["labels"], input_data['precise_sliding_num']
        # seg_score [stage_num, N, C, T]
        # masks [N, T]
        # labels [N, T]

        loss_dict={}
        loss = 0.
        
        # unet loss
        unet_loss_info = {"masks": masks, "labels": labels, "precise_sliding_num": precise_sliding_num}
        unet_loss = self.unet_loss({"output":pred_labels, "noise": noise}, unet_loss_info)['loss']
        loss += unet_loss

        # vae backbone label learning
        if self.control_net_loss is not None:
            pass

        # prompt net label learning
        if self.prompt_net_loss is not None:
            prompt_net_loss_info = {"masks": masks, "labels": labels, "precise_sliding_num": precise_sliding_num}
            # prompt_net_loss = self.prompt_net_loss({"output":backbone_score.unsqueeze(0)}, prompt_net_loss_info)['loss']
            prompt_net_loss = self.prompt_net_loss({"output":backbone_score}, prompt_net_loss_info)['loss']
            loss += prompt_net_loss
            loss_dict["prompt_net_loss"] = prompt_net_loss

        # segmentation branch loss
        if self.vae_loss is not None:
            vae_score = model_output["vae_score"]
            vae_loss_info = {"masks": masks, "labels": labels, "precise_sliding_num": precise_sliding_num}
            vae_loss = self.vae_loss({"output":vae_score}, vae_loss_info)['loss']
            loss += vae_loss
            loss_dict["vae_loss"] = vae_loss
        
        # prompt backbone loss
        if self.prompt_backbone_loss is not None:
            prompt_backbone_loss_info = {"masks": masks, "labels": labels, "precise_sliding_num": precise_sliding_num}
            if len(prompt_backbone_score.shape) == 2:
                prompt_backbone_loss = self.prompt_backbone_loss({"output":prompt_backbone_score.unsqueeze(0). unsqueeze(0)}, prompt_backbone_loss_info)['loss'] # 多batch的时候需要加一个维，不然不需要
            elif len(prompt_backbone_score.shape) == 3:
                prompt_backbone_loss = self.prompt_backbone_loss({"output":prompt_backbone_score.unsqueeze(0)}, prompt_backbone_loss_info)['loss']
            loss += prompt_backbone_loss
            loss_dict["prompt_backbone_loss"] = prompt_backbone_loss
        
        # if loss > 100:
        #     print("!")
        
        loss_dict["loss"] = loss
        loss_dict["unet_loss"] = unet_loss
        return loss_dict