import torch
from generative.networks.nets import DiffusionModelUNet

from utils import model_config

def load_autoencoder(bundle_target):  
    config = model_config(bundle_target, 'inference.json')
    autoencoder = config.get_parsed_content('autoencoder')
    autoencoder.load_state_dict(torch.load(config.get_parsed_content('load_autoencoder_path')), strict=False)
    return autoencoder

def load_unet(bundle_target, n_classes=6):
    config = model_config(bundle_target, 'inference.json')
    device = config.get_parsed_content('device')
    
    # Build U-Net manually with class conditioning
    net_params = config['network_def']  # Not all params are pre-evaluated
    unet = DiffusionModelUNet(spatial_dims=config.get_parsed_content('spatial_dims'),
                              in_channels=config['latent_channels'],
                              out_channels=config['latent_channels'],
                              num_channels=net_params['num_channels'],
                              attention_levels=net_params['attention_levels'],
                              num_head_channels=net_params['num_head_channels'],
                              num_res_blocks=net_params['num_res_blocks'],
                              num_class_embeds=n_classes).to(device)
    
    unet.load_state_dict(torch.load(config.get_parsed_content('load_diffusion_path')), strict=False)
    return unet