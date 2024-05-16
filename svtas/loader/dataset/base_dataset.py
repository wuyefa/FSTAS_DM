import abc
import os.path as osp
from typing import Dict

from svtas.dist import get_world_size_from_os, get_rank_from_os
class BaseDataset(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        pass
    
    @abc.abstractmethod
    def shuffle_dataset(self):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass



class BaseTorchDataset(BaseDataset):
    def __init__(self,
                 file_path,
                 gt_path,
                 pipeline,
                 actions_map_file_path,
                 temporal_clip_batch_size,
                 video_batch_size,
                 train_mode=False,
                 suffix='',
                 dataset_type='gtea',
                 data_prefix=None,
                 drop_last=False,
                 local_rank=-1,
                 nprocs=1,
                 data_path=None) -> None:
        super().__init__()
        self.suffix = suffix
        self.data_path = data_path
        self.gt_path = gt_path
        self.actions_map_file_path = actions_map_file_path
        self.dataset_type = dataset_type
        self.train_mode = train_mode
        
        self.file_path = file_path
        self.data_prefix = osp.realpath(data_prefix) if \
            data_prefix is not None and osp.isdir(data_prefix) else data_prefix
        self.pipeline = pipeline

        # distribute
        self.local_rank = get_rank_from_os()
        self.nprocs = get_world_size_from_os()
        self.drop_last = drop_last
        self.video_batch_size = video_batch_size
        self.temporal_clip_batch_size = temporal_clip_batch_size
    
    @abc.abstractmethod
    def shuffle_dataset(self):
        pass
    
    @abc.abstractmethod
    def __len__(self):
        pass

    def save(self) -> Dict:
        return {}

    def load(self, load_dict) -> None:
        pass