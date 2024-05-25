import os
import sys
import torch
import pickle

sys.path.append("..")
sys.path.append("../brats-mri")
sys.path.append('../brats-mri/brats_mri_class_cond/')
import monai
from PIL import Image
from tqdm import tqdm
from monai.utils import first
from generative.inferers import LatentDiffusionInferer
from generative.networks.schedulers import DDIMScheduler

from torch.utils.data import DataLoader

from pretrained import load_autoencoder, load_unet
import utils
from scripts.utils import compute_scale_factor

# TODO - define the correct path to data
sys.path.append('../brats-mri/brats_mri_class_cond/scripts')
from ct_rsna import CTSubset
import torch
import numpy as np

BUNDLE = '../brats-mri/brats_mri_class_cond/'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sys.path.append(BUNDLE)
from scripts.inferer import LatentDiffusionInfererWithClassConditioning

sys.path.append("../diffusion_classifier")
from ldm_classifier_brats import MonaiLdmClassifier


def get_monai_autoencoder(bundle_target, training_args, weights_override_path):
    # load autoencoder
    autoencoder = load_autoencoder(bundle_target,
                                   override_model_cfg_json=training_args.config,
                                   override_weights_load_path=weights_override_path)
    return autoencoder


def get_monai_unet(bundle_target, training_args, weights_override_path):
    unet = load_unet(bundle_target,
                     context_conditioning=training_args.conditioning == 'context',
                     override_model_cfg_json=training_args.config,
                     override_weights_load_path=weights_override_path,
                     use_conditioning=True)
    return unet


def get_monai_model_dict(bundle_target, autoencoder_weights_path, unet_weights_path, dataset):
    monai_dict = {}
    monai_dict['device'] = device
    training_args = torch.load(os.path.join(output_dir, training_name, 'training_args'))

    monai_dict['autoencoder'] = get_monai_autoencoder(bundle_target, training_args, autoencoder_weights_path)
    monai_dict['unet'] = get_monai_unet(bundle_target, training_args, unet_weights_path)

    # set scheduler
    config = utils.model_config(bundle_target, training_args.config)
    #     monai_dict['scheduler'] = config.get_parsed_content('noise_scheduler')
    scheduler = DDIMScheduler(num_train_timesteps=1000,
                              beta_start=0.0015, beta_end=0.0195,
                              schedule='scaled_linear_beta', clip_sample=False)

    scale_factor = get_scale_factor(dataset, monai_dict['autoencoder'], device)
    # set inferer
    if training_args.conditioning in ['context', 'none']:
        monai_dict['inferer'] = LatentDiffusionInferer(scheduler=scheduler, scale_factor=scale_factor)
    else:
        monai_dict['inferer'] = LatentDiffusionInfererWithClassConditioning(scheduler=scheduler,
                                                                            scale_factor=scale_factor)
    return monai_dict


def get_scale_factor(dataset, autoencoder, device):
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    scale_factor = compute_scale_factor(autoencoder, loader, device)
    return scale_factor


def get_training_ckpt_files(output_dir, training_name):
    ckpt_dir = os.path.join(output_dir, training_name, 'checkpoints')
    autoencoder_ckpt_files = [os.path.join(ckpt_dir, ckpt) for ckpt in os.listdir(ckpt_dir) if 'autoencoder' in ckpt]
    unet_ckpt_files = [os.path.join(ckpt_dir, ckpt) for ckpt in os.listdir(ckpt_dir) if 'diffusion' in ckpt]
    return autoencoder_ckpt_files, unet_ckpt_files


def strip_epoch_num_from_ckpt(ckpt_full_path):
    ckpt_name = ckpt_full_path.split('/')[-1]
    epoch_num = 1 + int(ckpt_name.split(".")[0].split("=")[-1])
    return epoch_num


def get_already_calculated(clf_dir, predictions_file_name):
    predictions_path = os.path.join(clf_dir, predictions_file_name)
    if os.path.exists(predictions_path):
        with open(predictions_path, 'rb') as f:
            pred_dict = pickle.load(f)
        n_already_calculated = len(pred_dict.keys())
    else:
        n_already_calculated = 0
    return n_already_calculated


if __name__ == "__main__":
    np.random.seed(7)
    torch.manual_seed(7)

    val_dir = '../data/ct-rsna/validation'
    output_dir = '../data/outputs'

    subset_len = 128
    predictions_file_name = f"predictions_{subset_len}_samples"

    ds = CTSubset(data_dir=val_dir, labels_file='validation_set_dropped_nans.csv',
                  size=256, flip_prob=0., subset_len=subset_len)

    trained_models = [
        'brats-2048',
        'brats-2048-contd',
    ]

    for training_name in trained_models:
        print(f"Performing Classification for training dir: {training_name}")
        clf_dir = os.path.join(output_dir, training_name, 'classification')
        if not os.path.exists(clf_dir):
            os.makedirs(clf_dir)

        clf_res_per_epoch = {}
        autoencoder_ckpt_files, unet_ckpt_files = get_training_ckpt_files(output_dir, training_name)

        n_already_calculated = get_already_calculated(clf_dir, predictions_file_name)
        if n_already_calculated == len(autoencoder_ckpt_files):
            print(
                f"training dir: {training_name} already classified for dataset of size {subset_len}, to rerun, make sure you delete previous results!")
        for i in range(n_already_calculated, len(autoencoder_ckpt_files)):
            autoenc_ckpt_file = autoencoder_ckpt_files[i]
            unetc_ckpt_file = unet_ckpt_files[i]

            epoch_num = strip_epoch_num_from_ckpt(autoenc_ckpt_file)
            monai_dict = get_monai_model_dict(bundle_target=BUNDLE,
                                              autoencoder_weights_path=autoenc_ckpt_file,
                                              unet_weights_path=unetc_ckpt_file,
                                              dataset=ds)

            clf = MonaiLdmClassifier(**monai_dict)
            df = clf.classify_dataset(dataset=ds,
                                      n_trials=3,
                                      t_sampling_stride=50)

            clf_res_per_epoch[epoch_num] = {
                'df': df
            }

            with open(os.path.join(clf_dir, f'predictions_{subset_len}_samples'), 'wb') as f:
                pickle.dump(clf_res_per_epoch, f)

            del monai_dict
            del clf
