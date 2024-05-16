from .base_pipline import BaseModelPipline, FakeModelPipline
from .torch_model_pipline import TorchModelPipline
from .deepspeed_model_pipline import DeepspeedModelPipline
from .torch_model_ddp_pipline import TorchDistributedDataParallelModelPipline
from .torch_cam_model_pipline import TorchCAMModelPipline
__all__ = [
    'BaseModelPipline', 'TorchModelPipline', 'DeepspeedModelPipline',
    'TorchDistributedDataParallelModelPipline', 'FakeModelPipline',
    'TorchCAMModelPipline'
]