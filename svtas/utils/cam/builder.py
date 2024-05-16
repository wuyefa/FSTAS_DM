from . import target_utils as custom_model_targets
from ..package_utils import is_pytorch_grad_cam_available

if is_pytorch_grad_cam_available():
    from pytorch_grad_cam.utils import model_targets

from . import match_fn

def get_model_target_class(target_name, cfg):
    target = getattr(model_targets, target_name, False)
    if target is False:
        target = getattr(custom_model_targets, target_name)(**cfg)
    else:
        target = target(**cfg)
    return target

def get_match_fn_class(fn_name):
    fn = getattr(match_fn, fn_name, False)
    return fn