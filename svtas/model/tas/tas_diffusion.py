import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List
from ..architectures import DiffusionModel
from svtas.utils import AbstractBuildFactory
import numpy as np

@AbstractBuildFactory.register('model')
class TemporalActionSegmentationDiffusionModel(DiffusionModel):
    def __init__(self,
                 vae: Dict,
                 scheduler: Dict,
                 unet: Dict = None,
                 weight_init_cfg: dict | List[dict] | None = dict(
                     vae=dict(
                        child_model=False,
                        revise_keys=[(r'backbone.', r'')]
                    ))) -> None:
        super().__init__(vae, scheduler, unet, weight_init_cfg)
        self.epoch_num = 0
    
    def run_train(self, data_dict: Dict[str, torch.FloatTensor]):
        if self.vae is not None:
            latents_dict = self.vae.encode(data_dict)
        else:
            latents_dict = data_dict

        labels = torch.permute(data_dict['labels_onehot'], dims=[0, 2, 1]).contiguous()
        condition_latents = latents_dict['output_feature']

        # Sample noise to add to the images
        noise = torch.randn(labels.shape).to(labels.device)
        bs = labels.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (bs,), device=labels.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        
        noisy_labels = self.scheduler.add_noise(labels, noise, timesteps)

        pred_noise_data_dict = dict(
            timestep = timesteps,
            noise_label = noisy_labels,
            condition_latens = condition_latents,
            labels = data_dict['labels'],
            boundary_prob = data_dict['boundary_prob'],
            masks_m = data_dict['masks_m']
        )
        noise_labels = self.unet(pred_noise_data_dict)['output']
        output_dict = dict(
            output = noise_labels,
            backbone_score = latents_dict['output'].squeeze(0)
        )
        if 'backbone_score' in latents_dict:
            output_dict['vae_backbone_score'] = latents_dict['backbone_score']
        return output_dict
    
    def run_test(self, data_dict: Dict[str, torch.FloatTensor]):
        # get latents
        if self.vae is not None:
            latents_dict = self.vae.encode(data_dict)
        else:
            latents_dict = data_dict
        
        # prepare for backward denoise
        
        condition_latents = latents_dict['output_feature']
        gt_labels = data_dict['labels_onehot']
        
        # Sample noise to add to the images
        # pred_labels = torch.permute(torch.randn_like(gt_labels), dims=[0, 2, 1]).contiguous()
        generator = torch.Generator(condition_latents.device).manual_seed(self.scheduler.get_random_seed_from_generator())
        pred_labels = torch.permute(torch.randn(list(gt_labels.shape), device=gt_labels.device, dtype=gt_labels.dtype, generator=generator), dims=[0, 2, 1]).contiguous()
        self.scheduler.set_timesteps(device=pred_labels.device)
        # generator = torch.Generator(pred_labels.device).manual_seed(self.scheduler.infer_region_seed)
        
        # denoise process
        for i, t in enumerate(self.scheduler.timesteps):
            t_now = torch.full((1,), t[0], device=t[0].device, dtype=torch.long)
            t_next = torch.full((1,), t[1], device=t[1].device, dtype=torch.long)

            pred_labels_scale = self.scheduler.scale_model_output(pred_labels)
            denoise_data_dict = dict(
                timestep = t_now,
                noise_label = pred_labels_scale,
                condition_latens = condition_latents,
                masks_m = data_dict['masks_m']
            )
            pred_labels_dict = self.unet(denoise_data_dict)
            pred_noise_labels = pred_labels_dict['output'].squeeze(0)
            
            step_dict = self.scheduler.step(model_output=pred_noise_labels,
                                            sample=pred_labels,
                                            timestep=t_now,
                                            next_timestep=t_next,
                                            generator=generator)
            pred_labels = step_dict['denoise_labels']
        pred_labels = self.scheduler.scale_model_output(pred_labels)
        decode_latents_dict = dict(
            output = pred_labels.unsqueeze(0),
            backbone_score = latents_dict['output'].squeeze(0)
        )
        if 'backbone_score' in latents_dict:
            decode_latents_dict['vae_backbone_score'] = latents_dict['backbone_score']
        # latent to output
        output_dict = self.vae.decode(decode_latents_dict)
        return output_dict

from diffusers import DDIMParallelScheduler
try_scheduler = DDIMParallelScheduler()
try_scheduler.set_timesteps(1000, torch.device("cuda:0"))

@AbstractBuildFactory.register('model')
class TemporalActionSegmentationDDIMModel(DiffusionModel):
    def __init__(self,
                 scheduler: Dict,
                 unet: Dict = None,
                 vae: Dict = None,
                 control_net: Dict = None,
                 prompt_net: Dict = None,
                 direct_pred: bool = True,
                 weight_init_cfg: dict | List[dict] | None = dict(
                     vae=dict(
                        child_model=False,
                        revise_keys=[(r'backbone.', r'')]
                    ))) -> None:
        super().__init__(vae, scheduler, unet, weight_init_cfg)
        if control_net is not None:
            self.control_net = AbstractBuildFactory.create_factory('model').create(control_net)
            self.component_list.append('control_net')
        else:
            self.control_net = None
        
        if prompt_net is not None:
            self.prompt_net = AbstractBuildFactory.create_factory('model').create(prompt_net)
            self.component_list.append('prompt_net')
        else:
            self.prompt_net = None
        self.direct_pred = direct_pred

    def run_train(self, data_dict: Dict[str, torch.FloatTensor]):
        labels = torch.permute(data_dict['labels_onehot'], dims=[0, 2, 1]).contiguous()

        if self.vae is not None:
            latents = self.vae.encode(dict(input_data = labels))['output']
        else:
            latents = labels
        
         # Sample noise to add to the images
        noise = torch.randn(latents.shape).to(latents.device)

        bs = latents.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (bs,), device=latents.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        # noisy_latents = try_scheduler.add_noise(original_samples=latents, noise=noise, timesteps=timesteps)

        # get condition feature
        condition_latents_dict = {}
        if self.control_net is not None:
            control_info = self.control_net(data_dict)
            condition_latents_dict['control_info'] = control_info
        if self.prompt_net is not None:
            prompt_info = self.prompt_net(data_dict)
            condition_latents_dict['prompt_info'] = prompt_info
        
        fill_features = F.interpolate(
            input=condition_latents_dict['prompt_info']['output']['output_feature'],
            scale_factor=[2],
            mode="nearest",
        )
        
        pred_noise_data_dict = dict(
            timestep = timesteps,
            noise_label = noisy_latents,
            condition_latens = fill_features
        )
        pred_noise_data_dict.update(data_dict)
        noise_labels = self.unet(pred_noise_data_dict)['output']
        
        prompt_backbone_score = F.interpolate(
            input=condition_latents_dict['prompt_info']['backbone_score'],
            scale_factor=[self.unet.sample_rate * 2],
            mode="nearest"
        ).squeeze(0)

        if self.direct_pred:
            output_dict = dict(
                output = noise_labels,
                noise = noise_labels,
                backbone_score = condition_latents_dict['prompt_info']['output']['output'].squeeze(0),
                prompt_backbone_score = prompt_backbone_score
            )
        else:
            output_dict = dict(
                output = noise_labels,
                noise = noise.unsqueeze(0),
                backbone_score = condition_latents_dict['prompt_info']['output']['output'].squeeze(0),
                prompt_backbone_score = prompt_backbone_score
            )
        if 'backbone_score' in condition_latents_dict['prompt_info']['output']:
            output_dict['condition_backbone_score'] = condition_latents_dict['prompt_info']['output']['backbone_score']
        return output_dict
    
    def run_test(self, data_dict: Dict[str, torch.FloatTensor]):
        gt_labels = data_dict['labels_onehot']
        # Sample noise to add to the images
        generator = torch.Generator(gt_labels.device).manual_seed(self.scheduler.get_random_seed_from_generator())
        pred_labels = torch.permute(torch.randn(list(gt_labels.shape), device=gt_labels.device, dtype=gt_labels.dtype, generator=generator), dims=[0, 2, 1]).contiguous()
        self.scheduler.set_timesteps(device=pred_labels.device)

        # prepare for backward denoise
        # get latents
        if self.vae is not None:
            pred_labels = self.vae.encode(dict(input_data = pred_labels))['output']

        # get condition feature
        condition_latents_dict = {}
        if self.control_net is not None:
            control_info = self.control_net(data_dict)
            condition_latents_dict['control_info'] = control_info
        if self.prompt_net is not None:
            prompt_info = self.prompt_net(data_dict)
            condition_latents_dict['prompt_info'] = prompt_info
        
        fill_features = F.interpolate(
            input=condition_latents_dict['prompt_info']['output']['output_feature'],
            scale_factor=[2],
            mode="nearest",
        )
        
        # denoise process
        for i, t in enumerate(self.scheduler.timesteps):
            t_now = torch.full((1,), t[0], device=t[0].device, dtype=torch.long)
            t_next = torch.full((1,), t[1], device=t[1].device, dtype=torch.long)

            pred_labels_scale = self.scheduler.scale_model_output(pred_labels)
            denoise_data_dict = dict(
                timestep = t_now,
                noise_label = pred_labels_scale,
                condition_latens = fill_features
            )
            denoise_data_dict.update(data_dict)
            pred_labels_dict = self.unet(denoise_data_dict)
            pred_noise_labels = pred_labels_dict['output'].squeeze(0)
            
            step_dict = self.scheduler.step(model_output=pred_noise_labels,
                                            sample=pred_labels,
                                            timestep=t_now,
                                            next_timestep=t_next,
                                            generator=generator)
            pred_labels = step_dict['denoise_labels']
        pred_labels = self.scheduler.scale_model_output(pred_labels)

        # latent to output
        if self.vae is not None:
            pred_labels = self.vae.decode(dict(input_data = pred_labels))['output']

        pred_labels = F.interpolate(
                input=pred_labels.unsqueeze(0),
                scale_factor=[1, self.unet.sample_rate],
                mode="nearest")
        
        prompt_backbone_score = F.interpolate(
            input=condition_latents_dict['prompt_info']['backbone_score'],
            scale_factor=[self.unet.sample_rate * 2],
            mode="nearest"
        ).squeeze(0)
        
        output_dict = dict(
            output = pred_labels,
            noise = pred_labels,
            backbone_score = condition_latents_dict['prompt_info']['output']['output'].squeeze(0),
            prompt_backbone_score = prompt_backbone_score
        )

        if 'backbone_score' in condition_latents_dict['prompt_info']['output']:
            output_dict['condition_backbone_score'] = condition_latents_dict['prompt_info']['output']['backbone_score']
        return output_dict


@AbstractBuildFactory.register('model')
class FeatureTemporalActionSegmentationDDIMModel(DiffusionModel):
    def __init__(self,
                 scheduler: Dict,
                 unet: Dict = None,
                 vae: Dict = None,
                 control_net: Dict = None,
                 prompt_net: Dict = None,
                 direct_pred: bool = True,
                 weight_init_cfg: dict | List[dict] | None = dict(
                     vae=dict(
                        child_model=False,
                        revise_keys=[(r'backbone.', r'')]
                    ))) -> None:
        super().__init__(vae, scheduler, unet, weight_init_cfg)
        if control_net is not None:
            self.control_net = AbstractBuildFactory.create_factory('model').create(control_net)
            self.component_list.append('control_net')
        else:
            self.control_net = None
        
        if prompt_net is not None:
            self.prompt_net = AbstractBuildFactory.create_factory('model').create(prompt_net)
            self.component_list.append('prompt_net')
        else:
            self.prompt_net = None
        self.direct_pred = direct_pred

    def run_train(self, data_dict: Dict[str, torch.FloatTensor]):
        labels = torch.permute(data_dict['labels_onehot'], dims=[0, 2, 1]).contiguous()

        if self.vae is not None:
            latents = self.vae.encode(dict(input_data = labels))['output']
        else:
            latents = labels
        
         # Sample noise to add to the images
        noise = torch.randn(latents.shape).to(latents.device)
        bs = latents.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (bs,), device=latents.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # get condition feature
        condition_latents_dict = {}
        if self.control_net is not None:
            control_info = self.control_net(data_dict)
            condition_latents_dict['control_info'] = control_info
        if self.prompt_net is not None:
            prompt_info = self.prompt_net(data_dict)
            condition_latents_dict['prompt_info'] = prompt_info
        
        pred_noise_data_dict = dict(
            timestep = timesteps,
            noise_label = noisy_latents,
            condition_latens = condition_latents_dict['prompt_info']['output']['output_feature']
        )
        pred_noise_data_dict.update(data_dict)
        noise_labels = self.unet(pred_noise_data_dict)['output']

        if self.direct_pred:
            output_dict = dict(
                output = noise_labels,
                noise = noise_labels,
                backbone_score = condition_latents_dict['prompt_info']['output']['output'].squeeze(0)
            )
        else:
            noise = F.interpolate(
                input=noise.unsqueeze(0),
                scale_factor=[1, noise_labels.shape[-1] // noise.shape[-1]],
                mode="nearest")
            
            output_dict = dict(
                output = noise_labels,
                noise = noise,
                backbone_score = condition_latents_dict['prompt_info']['output']['output'].squeeze(0)
            )
        if 'backbone_score' in condition_latents_dict['prompt_info']['output']:
            output_dict['condition_backbone_score'] = condition_latents_dict['prompt_info']['output']['backbone_score']
        return output_dict
    
    def run_test(self, data_dict: Dict[str, torch.FloatTensor]):
        gt_labels = data_dict['labels_onehot']
        # Sample noise to add to the images
        generator = torch.Generator(gt_labels.device).manual_seed(self.scheduler.get_random_seed_from_generator())
        pred_labels = torch.permute(torch.randn(list(gt_labels.shape), device=gt_labels.device, dtype=gt_labels.dtype, generator=generator), dims=[0, 2, 1]).contiguous()
        self.scheduler.set_timesteps(device=pred_labels.device)

        # prepare for backward denoise
        # get latents
        if self.vae is not None:
            pred_labels = self.vae.encode(dict(input_data = pred_labels))['output']

        # get condition feature
        condition_latents_dict = {}
        if self.control_net is not None:
            control_info = self.control_net(data_dict)
            condition_latents_dict['control_info'] = control_info
        if self.prompt_net is not None:
            prompt_info = self.prompt_net(data_dict)
            condition_latents_dict['prompt_info'] = prompt_info
        
        # denoise process
        for i, t in enumerate(self.scheduler.timesteps):
            t_now = torch.full((1,), t[0], device=t[0].device, dtype=torch.long)
            t_next = torch.full((1,), t[1], device=t[1].device, dtype=torch.long)

            pred_labels_scale = self.scheduler.scale_model_output(pred_labels)
            denoise_data_dict = dict(
                timestep = t_now,
                noise_label = pred_labels_scale,
                condition_latens = condition_latents_dict['prompt_info']['output']['output_feature']
            )
            denoise_data_dict.update(data_dict)
            pred_labels_dict = self.unet(denoise_data_dict)
            pred_noise_labels = pred_labels_dict['output'].squeeze(0)
            
            step_dict = self.scheduler.step(model_output=pred_noise_labels,
                                            sample=pred_labels,
                                            timestep=t_now,
                                            next_timestep=t_next,
                                            generator=generator)
            pred_labels = step_dict['denoise_labels']
        pred_labels = self.scheduler.scale_model_output(pred_labels)

        # latent to output
        if self.vae is not None:
            pred_labels = self.vae.decode(dict(input_data = pred_labels))['output']

        pred_labels = F.interpolate(
                input=pred_labels.unsqueeze(0),
                scale_factor=[1, self.unet.sample_rate],
                mode="nearest")
        
        output_dict = dict(
            output = pred_labels,
            noise = pred_labels,
            backbone_score = condition_latents_dict['prompt_info']['output']['output'].squeeze(0)
        )

        if 'backbone_score' in condition_latents_dict['prompt_info']['output']:
            output_dict['condition_backbone_score'] = condition_latents_dict['prompt_info']['output']['backbone_score']
        return output_dict