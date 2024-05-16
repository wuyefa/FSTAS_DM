import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from svtas.utils import AbstractBuildFactory
from .base_loss import BaseLoss

@AbstractBuildFactory.register('loss')
class SegmentationLoss(BaseLoss):
    """
    Args:
        addtion_loss: Dict[str, Dict[str, Any]]
    example:
    ```
        addition_loss = dict(
            boundary_loss = dict(
                name = "BoundaryRegressionLoss",
                pos_weight = [1, 1]
            )
        )
    ```
    """
    def __init__(self,
                 num_classes,
                 loss_weight=1.0,
                 sample_rate=1,
                 smooth_weight=0.5,
                 ignore_index=-100,
                 class_weight=None,
                 addtion_loss={}):
        super().__init__()
        self.smooth_weight = smooth_weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.elps = 1e-10
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        if self.class_weight is not None:
            self.ce = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weight), ignore_index=self.ignore_index, reduction='none')
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
        self.sm_loss = nn.MSELoss(reduction='none')
        self.addition_loss_dict = {}
        for key, cfg in addtion_loss.items():
            self.addition_loss_dict[key] = AbstractBuildFactory.create_factory('loss').create(cfg=cfg['loss'])
    
    def _compute_smooth_loss(self, p, labels, masks, b, precise_sliding_num):
        return torch.mean(
            (torch.mean(torch.reshape(torch.clamp(
                self.sm_loss(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1))
                , min=0, max=16) * masks[:, 1:].unsqueeze(1), [b, -1]), dim=-1) / (precise_sliding_num + self.elps)))

    def forward(self, model_output, input_data):
        # score shape [stage_num N C T]
        # masks shape [N T]
        head_score = model_output["output"]
        masks, labels, precise_sliding_num = input_data["masks"], input_data["labels"], input_data['precise_sliding_num']
        
        # if precise_sliding_num == 0:
        #     precise_sliding_num += 1
        
        # print("head_score.shape", head_score.shape)
        _, b, _, t = head_score.shape

        loss = 0.
        for p in head_score:
            seg_cls_loss = self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), labels.view(-1))
            loss += torch.sum(torch.sum(torch.reshape(seg_cls_loss, shape=[b, t]), dim=-1) / (precise_sliding_num + self.elps)) / (torch.sum(labels != -100) + self.elps)
            if(self.smooth_weight> 0.0):
                loss += self.smooth_weight * self._compute_smooth_loss(p, labels, masks, b, precise_sliding_num)
        loss = loss * self.loss_weight

        for key, loss_instance in self.addition_loss_dict.items():
            loss += loss_instance(model_output)

        loss_dict={}
        loss_dict["loss"] = loss
        return loss_dict

@AbstractBuildFactory.register('loss')
class ActionCLIPSegmentationLoss(SegmentationLoss):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, model_output, input_data):
        # score shape [stage_num N C T]
        # masks shape [N T]
        img_feature, text_feature, head_score = model_output["image_feature"], model_output["text_feature"], model_output["output"]
        masks, labels, precise_sliding_num = input_data["masks"], input_data["labels"], input_data['precise_sliding_num']
        
        _, b, _, t = head_score.shape

        loss = 0.
        for p in head_score:
            seg_cls_loss = self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), labels.view(-1))
            loss += torch.sum(torch.sum(torch.reshape(seg_cls_loss, shape=[b, t]), dim=-1) / (precise_sliding_num + self.elps)) / (torch.sum(labels != -100) + self.elps)
            loss += self.smooth_weight * torch.mean(
                (torch.mean(torch.reshape(torch.clamp(
                    self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1))
                    , min=0, max=16) * masks[:, 1:].unsqueeze(1), [b, -1]), dim=-1) / (precise_sliding_num + self.elps)))
        
        loss_dict={}
        loss_dict["loss"] = loss * self.loss_weight
        return loss_dict

@AbstractBuildFactory.register('loss')
class LSTRSegmentationLoss(BaseLoss):
    def __init__(self,
                 num_classes,
                 loss_weight=1.0,
                 ignore_index=-100):
            super().__init__()
            self.num_classes = num_classes
            self.ignore_index = ignore_index
            self.loss_weight = loss_weight
            self.logsoftmax = nn.LogSoftmax(dim=-1)
            self.elps = 1e-10
    
    def forward(self, model_output, input_data):
        # score shape [stage_num N C T]
        # masks shape [N T]
        head_score = model_output["output"]
        masks, labels, precise_sliding_num = input_data["masks"], input_data["labels"], input_data['precise_sliding_num']
        
        _, b, _, t = head_score.shape

        loss = 0.
        for p in head_score:
            # smooth label learning
            with torch.no_grad():
                device = head_score.device
                # [N T]
                raw_labels = labels[:, (-t):]
                # deal label over num_classes
                # [N, 1]
                y = torch.zeros(raw_labels.shape, dtype=raw_labels.dtype, device=device)
                refine_label = torch.where(raw_labels != self.ignore_index, raw_labels, y)
                # [N T C]
                ce_y = F.one_hot(refine_label, num_classes=self.num_classes)

                raw_labels_repeat = torch.tile(raw_labels.unsqueeze(2), dims=[1, 1, self.num_classes])
                ce_y = torch.where(raw_labels_repeat != self.ignore_index, ce_y, torch.zeros(ce_y.shape, device=device, dtype=ce_y.dtype)).float()
            seg_cls_loss = torch.sum(-ce_y.view(-1, self.num_classes) * self.logsoftmax(p.transpose(2, 1).contiguous().view(-1, self.num_classes)),dim=1)
            loss += torch.sum(torch.sum(torch.reshape(seg_cls_loss, shape=[b, t]), dim=-1) / (precise_sliding_num + self.elps)) / (torch.sum(labels != -100) + self.elps)
        
        loss_dict={}
        loss_dict["loss"] = loss * self.loss_weight
        return loss_dict