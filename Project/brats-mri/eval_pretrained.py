import os
import sys
import torch
import monai
from matplotlib import pyplot as plt
from monai.bundle import ConfigParser


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
    unet = model_config.get_parsed_content("diffusion")
    autoencoder = model_config.get_parsed_content('autoencoder')

    # Load pre-trained
    unet.load_state_dict(model_weights)
    autoencoder.load_state_dict(torch.load(model_config.get_parsed_content('load_autoencoder_path')))

    # Test
    inferer = model_config.get_parsed_content('inferer')
    scheduler = model_config.get_parsed_content('noise_scheduler')
    latent_shape = model_config.get_parsed_content('latent_shape')

    for i in range(1, 4):
        noise = torch.randn([1] + latent_shape).to(device)
        sample = inferer.sampling_fn(noise, autoencoder, unet, scheduler, conditioning=None)

        plt.subplot(1, 3, i)
        plt.imshow(sample.squeeze().cpu(), cmap='gray')
        plt.axis('off')
    plt.show()