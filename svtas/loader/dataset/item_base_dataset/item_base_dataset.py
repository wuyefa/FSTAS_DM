from abc import abstractmethod

import torch.utils.data as data
from ..base_dataset import BaseTorchDataset


class ItemDataset(data.Dataset, BaseTorchDataset):
    """
    ItemDataset For Temporal Video Segmentation
    Other TVS ItemDataset should inherite it.
    """
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
        super().__init__(file_path, gt_path, pipeline, actions_map_file_path,
                         temporal_clip_batch_size, video_batch_size, train_mode,
                         suffix, dataset_type, data_prefix, drop_last, local_rank,
                         nprocs, data_path)

        # actions dict generate
        file_ptr = open(self.actions_map_file_path, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        self.actions_dict = dict()
        for a in actions:
            self.actions_dict[a.split()[1]] = int(a.split()[0])
        
        self.info = self.load_file()
    
    def shuffle_dataset(self):
        return self._viodeo_sample_shuffle()

    def _viodeo_sample_shuffle(self):
        pass

    @abstractmethod
    def load_file(self):
        raise NotImplementedError("You should Implement it!")

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError("You should Implement it!")
    
    def __len__(self):
        return len(self.info)
    
