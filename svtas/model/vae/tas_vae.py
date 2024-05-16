from typing import Dict
import torch
from ..architectures.general import VariationalAutoEncoder
from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('model')
class TemporalActionSegmentationVariationalAutoEncoder(VariationalAutoEncoder):
    def __init__(self,
                 encoder,
                 decoder,
                 weight_init_cfg=dict(
                    encoder = dict(
                        child_model=False,
                        revise_keys=[(r'backbone.', r'')]),
                    decoder = dict(
                        child_model=False,
                        revise_keys=[(r'backbone.', r'')]),
                    )):
        super().__init__(encoder, decoder, weight_init_cfg)
    
    def encode(self, input_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output_dict = self.encoder(input_data)
        if isinstance(output_dict['output'], dict):
            output_dict.update(output_dict['output'])
        return output_dict