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
    autoencoder = config.get_parsed_content('autoencoder')
    weights_load_path = (
        override_weights_load_path
        if override_weights_load_path is not None
        else config.get_parsed_content('load_autoencoder_path')
    )

    autoencoder.load_state_dict(torch.load(weights_load_path), strict=False)
    return autoencoder


def load_unet(bundle_target,
              n_classes=6,
              override_model_cfg_json: Optional[str] = None,
              override_weights_load_path: Optional[str] = None
              ):
    model_json = (
        override_model_cfg_json
        if override_model_cfg_json is not None
        else 'inference.json'
    )
    config = model_config(bundle_target, model_json)
    device = config.get_parsed_content('device')
    
    # Build U-Net manually with class conditioning
    net_params = config['network_def']  # Not all params are pre-evaluated
    # TODO - notice that this unet does not use context conditioning, it uses class embedding.
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
    unet.load_state_dict(torch.load(weights_load_path), strict=False)
    return unet