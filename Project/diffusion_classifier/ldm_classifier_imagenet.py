import sys
from typing import List, Tuple

import torch
import tqdm
import numpy as np

from torch.utils.data import Dataset
from torch.cuda.amp import autocast

from diffusion_classifier.diffusion_classifier_interface import DiffusionClassifierInterface

sys.path.append('../latent-diffusion')
sys.path.append('../latent-diffusion/ldm/data/')
from ldm.models.diffusion.ddpm import LatentDiffusion

class LdmClassifier(DiffusionClassifierInterface):
    """
    wrapper class for LatentDiffusion model class.
    LdmClassifier implements the code from: https://github.com/diffusion-classifier/diffusion-classifier/tree/master
    in a custom manner for LatentDiffusion class.

    LdmClassifier allows to use the diffusion model as a classifier based on: https://arxiv.org/abs/2303.16203
    """
    def __init__(self,
                 model: LatentDiffusion,
                 n_noise_samples: int = 1024,
                 random_seed: int = 42):
        """
        :param model: pretrained LatentDiffusion to use a classifier
        :param n_noise_samples: the classifier uses the same noise realizations for all predictions.
                                n_noise_samples should be greater or equal w.r.t the number of diffusion time steps
        """
        assert n_noise_samples >= model.num_timesteps, f"n_noise_samples: {n_noise_samples} must be greater or equal from model.num_timesteps: {model.num_timesteps}"
        self._model = model
        super().__init__(device=model.device, n_noise_samples=n_noise_samples, random_seed=random_seed)

    @property
    def model(self):
        return self._model

    @property
    def input_dims(self):
        """
        returns the input dims for the model CxHxW (ignoring the batch dimension)
        """
        return self._model.channels, self._model.image_size, self._model.image_size

    @property
    def n_train_timesteps(self):
        return self._model.num_timesteps

    def get_latent_batch(self, batch):
        """
        prepares input for the latent diffusion model.

        :param batch: raw input batch from dataloader.
                      assuming structure {
                                            'image': List[torch.Tensor],
                                            'class_label': List[torch.Tensor],
                                            'human_label': List[str]
                                         }
        """
        x0, c_true = self._model.get_input(batch, self._model.first_stage_key)
        return x0

    def classify_batch(self,
                       x0,
                       c_hypotheses,
                       n_trials: int = 1,
                       t_sampling_stride: int = 5):
        """
        classify a single batch
        :param x0: diffusion input
        :param c_hypotheses: conditioning hypothesis input
        :param n_trials: number of trials to do for each sample. TODO - need to revisit how this is different than the batch...
        :param t_sampling_stride: sampling rate of the diffusion time steps

        :returns : torch.Tensor(L2 predicted label), torch.Tensor(L1 predicted label)
        """
        x0 = self.get_latent_batch(x0)
        return super(LdmClassifier, self).classify_batch(x0, c_hypotheses, n_trials, t_sampling_stride)

    def get_noised_input(self, x0, t, noise):
        return self._model.q_sample(x0, t, noise)

    def get_conditioning(self, cond_input):
        cond_emb = self._model.get_learned_conditioning(cond_input)
        # prepare condition embedding dict in correct form for diffusion_model.forward
        key = 'c_concat' if self._model.model.conditioning_key == 'concat' else 'c_crossattn'
        cond_emb = {key: [cond_emb]}
        return cond_emb

    def get_diffusion_noise_prediction(self, x0, t, conditioning):
        noise_pred = self._model.model(x0, t, **conditioning)
        return noise_pred

    def get_noise_prediction(self, x0, t, noise, c):
        noised_latent = self.get_noised_input(x0, t, noise)

        # prepare conditioning input
        cond_input = self.trim_cond_dict(c, len(t))

        # get condition embedding
        cond_emb = self.get_conditioning(cond_input)

        # get noise prediction from diffusion model
        noise_pred = self.get_diffusion_noise_prediction(noised_latent, t, cond_emb)
        return noise_pred