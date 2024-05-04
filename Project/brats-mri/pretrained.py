from typing import Optional

import torch
from generative.networks.nets import DiffusionModelUNet

from utils import model_config


def load_autoencoder(bundle_target,
                     override_model_cfg_json: Optional[str] = None,
                     override_weights_load_path: Optional[str] = None):
    model_json = (
        override_model_cfg_json
        if override_model_cfg_json is not None
        else 'inference.json'
    )
    config = model_config(bundle_target, model_json)
    device = config.get_parsed_content('device')

    autoencoder = config.get_parsed_content('autoencoder')
    weights_load_path = (
        override_weights_load_path
        if override_weights_load_path is not None
        else config.get_parsed_content('load_autoencoder_path')
    )

    autoencoder.load_state_dict(torch.load(weights_load_path, map_location=device), strict=False)
    return autoencoder


def load_unet(bundle_target,
              context_conditioning: bool,
              n_classes: int=6,
              override_model_cfg_json: Optional[str] = None,
              override_weights_load_path: Optional[str] = None):
    model_json = (
        override_model_cfg_json
        if override_model_cfg_json is not None
        else 'inference.json'
    )
    config = model_config(bundle_target, model_json)
    device = config.get_parsed_content('device')
    
    # Build U-Net manually with class conditioning
    net_params = config['network_def']  # Not all params are pre-evaluated

    if context_conditioning:
        unet = DiffusionModelUNet(spatial_dims=config.get_parsed_content('spatial_dims'),
                                in_channels=config['latent_channels'],
                                out_channels=config['latent_channels'],
                                num_channels=net_params['num_channels'],
                                attention_levels=net_params['attention_levels'],
                                num_head_channels=net_params['num_head_channels'],
                                num_res_blocks=net_params['num_res_blocks'],
                                with_conditioning=True, cross_attention_dim=1).to(device)
    else:
        unet = DiffusionModelUNet(spatial_dims=config.get_parsed_content('spatial_dims'),
                                in_channels=config['latent_channels'],
                                out_channels=config['latent_channels'],
                                num_channels=net_params['num_channels'],
                                attention_levels=net_params['attention_levels'],
                                num_head_channels=net_params['num_head_channels'],
                                num_res_blocks=net_params['num_res_blocks'],
                                num_class_embeds=n_classes).to(device)
    
    weights_load_path = (
        override_weights_load_path
        if override_weights_load_path is not None
        else config.get_parsed_content('load_diffusion_path')
    )
    pretrained = torch.load(weights_load_path, map_location=device)


    """
    Reminder
    Missing: missing from src but expected in dst
    Unexpected: present in src but not in dst

    We successfully loaded (len(pretrained) - len(unexpected)) / len(pretrained) keys
    Missing keys don't matter- they are parameters that didn't exist in the src model
    """
    missing, unexpected = unet.load_state_dict(pretrained, strict=False)

    if unexpected:
        success_rate = 100. * (len(pretrained) - len(unexpected)) / len(pretrained)
        print(f"Unable to load complete model: successfully loaded {success_rate:.1f}% of weights")

    return unet