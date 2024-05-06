import os
import monai
import torch
import shutil
from monai.bundle import ConfigParser
import argparse

PRETRAINED_MODEL_NAME = 'brats_mri_axial_slices_generative_diffusion'

def rescale_outputs(im):
    """
    Model outputs are in [0, 1]
    Rescale them to [0, 255] and convert to uint8 so they can be displayed
    """
    im = im.clamp(0, 1) * 255
    return im.to(torch.uint8).cpu().numpy()


def rescale_inputs(im):
    """
    Models expect inputs in [-1, 1] but return [0, 1]
    Need to rescale inputs to view them
    """
    return 0.5 * im + 0.5


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

        if not os.path.exists(os.path.join(bundle_target, 'models')):
            os.mkdir(os.path.join(bundle_target, 'models'))

        copy_weights_file(autoencoder_weights, PRETRAINED_MODEL_NAME, bundle_target)
        copy_weights_file(unet_weights, PRETRAINED_MODEL_NAME, bundle_target)


def copy_weights_file(file_name, src_dir, dst_dir):
    shutil.copyfile(os.path.join(src_dir, file_name), os.path.join(dst_dir, file_name))


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Neural Network Training Script')

    # Argument for number of epochs
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs for training')

    # Argument for setting epochs interval for validation
    parser.add_argument('--val_every_n_epochs', type=int, default=1,
                        help='epochs interval between validation loops, default is 1')

    # Argument for resuming training from a checkpoint
    parser.add_argument('--resume_from_ckpt', type=str, default=None,
                        help='Path to a checkpoint file to resume training from')

    # Argument for name of the model
    parser.add_argument('--name', type=str, default='model',
                        help='Name of the model')

    # Argument for path to the config file
    parser.add_argument('--config', type=str, default='inference.json',
                        help='Path to the configuration file')

    # Argument for seed
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed setting')

    # Argument for periodic checkpoint saving
    parser.add_argument('--save_ckpt_every_n', type=int, default=None,
                        help='periodic ckpt saving')
    
    parser.add_argument('--conditioning', type=str, choices=['class', 'context', 'none'],
                        required=True, help='context or class conditioning with the class label')
    
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    return parser
