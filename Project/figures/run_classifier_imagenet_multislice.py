import os
import sys
import torch
import pickle
import numpy as np
from omegaconf import OmegaConf

sys.path.append('./')
sys.path.append('./latent-diffusion')
sys.path.append('./taming-transformers')

from ldm.util import instantiate_from_config
from ldm.data.ct_rsna import CTSubset, MultiSliceCTDataset
from diffusion_classifier.ldm_classifier_imagenet import LdmClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, weights_only=False, map_location=torch.device('cpu'))
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.eval()
    return model.to(device), sd, pl_sd


def get_model(model_config_path, model_ckpt_path):
    config = OmegaConf.load(model_config_path)
    model, _, _ = load_model_from_config(config, model_ckpt_path)
    return model


def get_training_ckpt_files(output_dir, training_name):
  ckpt_dir = os.path.join(output_dir, training_name, 'checkpoints')
  training_ckpt_files = [os.path.join(ckpt_dir, ckpt) for ckpt in os.listdir(ckpt_dir) if 'epoch' in ckpt]
  return training_ckpt_files


def strip_epoch_num_from_ckpt(ckpt_full_path):
  ckpt_name = ckpt_full_path.split('/')[-1]
  epoch_num = 1 + int(ckpt_name.split(".")[0].split("=")[-1])
  return epoch_num


def get_training_cfg_file(output_dir, training_name):
  cfg_dir = os.path.join(output_dir, training_name, 'configs')
  model_cfg_files = [cfg for cfg in os.listdir(cfg_dir) if 'project' in cfg]
  if len(model_cfg_files) == 0:
    raise ValueError("configs dir empty, you may manualy pass the config file instead")
  if len(model_cfg_files) > 1:
    raise ValueError("more than 1 config file in configs dir, you may manualy pass the config file instead")
  
  return os.path.join(cfg_dir, model_cfg_files[0])


def get_already_calculated(clf_dir, predictions_file_name):
    predictions_path = os.path.join(clf_dir, predictions_file_name)
    if os.path.exists(predictions_path):
        with open(predictions_path, 'rb') as f:
            pred_dict = pickle.load(f)
        n_already_calculated = len(pred_dict.keys())
    else:
        pred_dict = None
        n_already_calculated = 0
    return n_already_calculated, pred_dict


if __name__ == "__main__":
  np.random.seed(7)
  torch.manual_seed(7)

  train_dir = './data/ct-rsna/train'
  val_dir = './data/ct-rsna/validation'
  data_dir = './data/ct-rsna'

  output_dir = './data/outputs'
  
  n_slices_in_study = 10
  predictions_file_name = f"predictions_multi_slice_{n_slices_in_study}_in_study"

  ds = MultiSliceCTDataset(data_dir=data_dir,
                           train_dir=train_dir,
                           val_dir=val_dir,
                           labels_file='unified_set_with_study_id_dropped_nans.csv',
                           size=256,
                           flip_prob=0.,
                           n_slices_in_study=n_slices_in_study
                           )

  trained_models = [
          '2024-05-17T21-13-47_imagenet-4096-80-epochs-fixed-cycles'
  ]

  for training_name in trained_models:
    print(f"Performing Classification for training dir: {training_name}")
    clf_dir = os.path.join(output_dir, training_name, 'classification')
    if not os.path.exists(clf_dir):
        os.makedirs(clf_dir)

    clf_res_per_epoch = {}
    training_ckpt_files = get_training_ckpt_files(output_dir, training_name)
    cfg_file = get_training_cfg_file(output_dir, training_name)

    n_already_calculated, pred_dict = get_already_calculated(clf_dir, predictions_file_name)
    if n_already_calculated > 0:
        clf_res_per_epoch = pred_dict
    if n_already_calculated == len(training_ckpt_files):
        print(f"training dir: {training_name} already classified for multi-slice of size {n_slices_in_study}, to rerun, make sure you delete previous results!")
    for i in range(0, len(training_ckpt_files)):
        ckpt_file = training_ckpt_files[i]
        epoch_num = strip_epoch_num_from_ckpt(ckpt_file)
        if epoch_num in pred_dict.keys():
            continue
        model = get_model(cfg_file, ckpt_file)
        
        clf = LdmClassifier(model)
        df = clf.classify_dataset(dataset=ds,
                                  n_trials=3,
                                  t_sampling_stride=50)
        
        clf_res_per_epoch[epoch_num] = {
                                        'df': df
                                        }

        with open(os.path.join(clf_dir, predictions_file_name), 'wb') as f:
            pickle.dump(clf_res_per_epoch, f)
        
        del model
        del clf
        
