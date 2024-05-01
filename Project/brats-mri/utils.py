import os
import monai
import shutil
from monai.bundle import ConfigParser


PRETRAINED_MODEL_NAME = 'brats_mri_axial_slices_generative_diffusion'

def model_config(bundle_target, file_name):
    model_config_file = os.path.join(bundle_target, "configs", file_name)
    
    model_config = ConfigParser()
    model_config.read_config(model_config_file)

    model_config['bundle_root'] = bundle_target
    model_config['model_dir'] = os.path.join(bundle_target, 'models')
    return model_config


def download_weights_if_not_already(bundle_target):

    model_config = ConfigParser()
    config_file = os.path.join(bundle_target, "configs", "inference.json")

    model_config.read_config(config_file)

    autoencoder_weights = model_config.get_parsed_content('load_autoencoder_path').lstrip('./')
    unet_weights = model_config.get_parsed_content('load_diffusion_path').lstrip('./')

    if os.path.exists(os.path.join(bundle_target, autoencoder_weights)) and os.path.exists(os.path.join(bundle_target, unet_weights)):
        return
    else:
        if not os.path.exists(PRETRAINED_MODEL_NAME):
            monai.bundle.load(name=PRETRAINED_MODEL_NAME, bundle_dir="")

        copy_weights_file(autoencoder_weights, PRETRAINED_MODEL_NAME, bundle_target)
        copy_weights_file(unet_weights, PRETRAINED_MODEL_NAME, bundle_target)


def copy_weights_file(file_name, src_dir, dst_dir):
    shutil.copyfile(os.path.join(src_dir, file_name), os.path.join(dst_dir, file_name))
