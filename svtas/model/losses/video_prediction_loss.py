import torch
import torch.nn as nn
import torch.nn.functional as F
from .segmentation_loss import SegmentationLoss

from svtas.utils import AbstractBuildFactory
from .base_loss import BaseLoss

@AbstractBuildFactory.register('loss')
class VideoPredictionLoss(BaseLoss):
    def __init__(self,
                 num_classes,
                 sample_rate=1,
                 smooth_weight=0.5,
                 pred_loss_weight=1.0,
                 segment_loss_weight=1.0,
                 ignore_index=-100):
        super().__init__()
        self.smooth_weight = smooth_weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.pred_loss_weight = pred_loss_weight
        self.segment_loss_weight = segment_loss_weight
        self.elps = 1e-10
        self.segmentation_loss = SegmentationLoss(num_classes=self.num_classes, loss_weight=self.segment_loss_weight, 
                sample_rate=self.sample_rate, smooth_weight=self.smooth_weight, ignore_index=self.ignore_index)
        self.pred_loss = SegmentationLoss(num_classes=self.num_classes, loss_weight=self.pred_loss_weight, 
                sample_rate=self.sample_rate, smooth_weight=self.smooth_weight, ignore_index=self.ignore_index)
    
    def forward(self, model_output, input_data):
        pred_frames_score, frames_score = model_output["output"]
        masks, labels, pred_labels, precise_sliding_num = input_data["masks"], input_data["labels"], input_data["pred_labels"], input_data['precise_sliding_num']
        # frames_score [stage_num, N, C, T]
        # pred_frames_score [stage_num， N, C, P_T]
        # masks [N, T]
        # labels [N, T]
        # pred labels [N, T]

        # segmentation branch loss
        seg_loss_info = {"masks": masks, "labels": labels, "precise_sliding_num": precise_sliding_num}
        seg_loss = self.segmentation_loss(frames_score, seg_loss_info)['loss']

        # pred branch loss
        pred_masks = torch.where(pred_labels != self.ignore_index, torch.ones_like(pred_labels), torch.zeros_like(pred_labels))
        pred_loss_info = {"masks": pred_masks, "labels": pred_labels, "precise_sliding_num": precise_sliding_num}
        pred_loss = self.pred_loss(pred_frames_score, pred_loss_info)['loss']

        loss = seg_loss + pred_loss

        loss_dict={}
        loss_dict["loss"] = loss
        return loss_dict