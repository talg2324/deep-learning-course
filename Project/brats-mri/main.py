import os
import sys
import torch
import monai
from matplotlib import pyplot as plt
from monai.bundle import ConfigParser
from generative.networks.nets import DiffusionModelUNet

MODEL_NAME = 'brats_mri_axial_slices_generative_diffusion'

sys.path.append(MODEL_NAME)


if __name__ == "__main__":
    
    monai.utils.set_determinism(seed=7)

    # Download, unzip, load state dict
    model_weights = monai.bundle.load(name=MODEL_NAME, bundle_dir="./")

    # Load the config file containing function definitions
    model_config_file = os.path.join(MODEL_NAME, "configs", "inference.json")
    model_config = ConfigParser()
    model_config.read_config(model_config_file)

    model_config['bundle_root'] = MODEL_NAME
    model_config['model_dir'] = os.path.join(MODEL_NAME, 'models')

    # Initialize models
    device = model_config.get_parsed_content('device')
    autoencoder = model_config.get_parsed_content('autoencoder')

    # Build U-Net manually with class conditioning
    net_params = model_config['network_def']  # Not all params are pre-evaluated
    unet = DiffusionModelUNet(spatial_dims=model_config.get_parsed_content('spatial_dims'),
                              in_channels=model_config['latent_channels'],
                              out_channels=model_config['latent_channels'],
                              num_channels=net_params['num_channels'],
                              attention_levels=net_params['attention_levels'],
                              num_head_channels=net_params['num_head_channels'],
                              num_res_blocks=net_params['num_res_blocks'],
                              with_conditioning=True,
                              cross_attention_dim=1).to(device)
    
    # unet.load_state_dict(model_weights, strict=False)

    # Test
    inferer = model_config.get_parsed_content('inferer')
    scheduler = model_config.get_parsed_content('noise_scheduler')
    latent_shape = model_config.get_parsed_content('latent_shape')

    # TODO:
    # See if this helps:
    # https://github.com/Project-MONAI/GenerativeModels/blob/main/tutorials/generative/classifier_free_guidance/2d_ddpm_classifier_free_guidance_tutorial.ipynb

    class_conditioning = torch.zeros((1, 1), dtype=torch.float32, device=device)

    for i in range(1, 4):
        noise = torch.randn([1] + latent_shape).to(device)
        sample = inferer.sampling_fn(noise, autoencoder, unet, scheduler, conditioning=class_conditioning)

        plt.subplot(1, 3, i)
        plt.imshow(sample.squeeze().cpu(), cmap='gray')
        plt.axis('off')
    plt.show()