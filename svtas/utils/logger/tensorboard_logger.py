import os
from .meter import AverageMeter
from ..build import AbstractBuildFactory
from .base_logger import BaseLogger, LoggerLevel

import cv2
import torch
from torch.utils.tensorboard import SummaryWriter

@AbstractBuildFactory.register('logger')
class TensboardLogger(BaseLogger):
    def __init__(self, name: str, root_path: str = None, level=LoggerLevel.INFO) -> None:
        super().__init__(name, root_path, level)
        self.logger = SummaryWriter(os.path.join(self.path, "tensorboard"), comment=name)
        self.step = 0
        self.batch = 0
        self.epoch = 0
    
    def log_epoch(self, metric_list, epoch=None, mode='train', ips=None):
        if epoch is None or mode != 'train':
            epoch = self.epoch
            self.epoch + 1
        else:
            self.epoch = epoch

        for m in metric_list:
            if not (m == 'batch_time' or m == 'reader_time'):
                self.logger.add_scalar(mode + "/" + m, metric_list[m].avg, epoch)
    
    def log_batch(self, metric_list, batch_id, mode, ips, epoch_id=None, total_epoch=None):
        if batch_id is None:
            batch_id = self.batch
            self.batch + 1
        else:
            self.batch = batch_id
    
    def log_step(self, metric_list, step_id, mode, ips, total_step=None):
        if step_id is None:
            step_id = self.step
            self.step += 1
        else:
            self.step =  step_id
    
    def log_metric(self, metric_dict, epoch, mode):
        if epoch is None or mode != 'train':
            epoch = self.epoch
        else:
            self.epoch = epoch
        for k, v in metric_dict.items():
            if not (k == 'batch_time' or k == 'reader_time'):
                self.logger.add_scalar(mode + "/" + k, v, epoch)
    
    def log_feature_image(self, feature: torch.Tensor, tag: str, step: int = None):
        if step is None:
            step = self.step
        assert len(list(feature.shape)) == 2, f"Only support fearure shape is 2-D, now is {list(feature.shape)}-D!"
        feature_data = feature.detach().clone().cpu().numpy()
        heatmapshow = None
        heatmapshow = cv2.normalize(feature_data, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        self.logger.add_image(tag, heatmapshow, step, dataformats = "HWC")
    
    def log_tensor(self, tensor, name, step=None):
        if step is None:
            step = self.step
        self.logger.add_histogram(tag=name, values=tensor, global_step=self.step)

    def log(self, msg, value=None, step=None):
        # ignore string
        if isinstance(msg, str) and value is None:
            return
        
        if step is None:
            step = self.step
        self.logger.add_scalar(msg, value, step)
    
    def log_model_parameters_grad_histogram(self, model, step=None):
        if step is None:
            step = self.step
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.logger.add_histogram(tag=name+'_grad', values=param.grad.detach().cpu(), global_step=self.step)
    
    def log_model_parameters_histogram(self, model, step=None):
        if step is None:
            step = self.step
        for name, param in model.named_parameters():
            self.logger.add_histogram(tag=name, values=param.detach().cpu(), global_step=self.step)
    
    def tensorboard_add_graph(self, model, fake_input = torch.randn(16,2)):
        self.logger.add_graph(model=model, input_to_model=fake_input)
        
    def close(self):
        self.logger.close()
    
    def info(self,
             msg: object,
             *args: object,
             **kwargs):
        pass

    def debug(self,
             msg: object,
             *args: object,
             **kwargs):
        pass
    
    def warn(self,
             msg: object,
             *args: object,
             **kwargs):
        pass
    
    def error(self,
             msg: object,
             *args: object,
             **kwargs):
        pass
    
    def critical(self,
             msg: object,
             *args: object,
             **kwargs):
        pass