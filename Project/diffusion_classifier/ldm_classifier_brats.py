from typing import List, Tuple

import torch
import sys
import tqdm
import numpy as np

from torch.utils.data import Dataset
from torch.cuda.amp import autocast

from generative.networks.nets import DiffusionModelUNet, AutoencoderKL
from diffusion_classifier.diffusion_classifier_interface import DiffusionClassifierInterface


class MonaiLdmClassifier(DiffusionClassifierInterface):
    def __init__(self,
                 autoencoder: AutoencoderKL,
                 unet: DiffusionModelUNet,
                 inferer,
                 n_noise_samples: int = 1024,
                 random_seed: int = 42,
                 input_dims: Tuple[int, int, int] = (1, 256, 256),
                 ):
        self._input_dims = input_dims
        self._inferer = inferer
        self._unet = unet
        self._autoencoder = autoencoder
        super().__init__(device=unet.device, n_noise_samples=n_noise_samples, random_seed=random_seed)

    @property
    def unet(self):
        return self._unet

    @property
    def autoencoder(self):
        return self._autoencoder

    @property
    def inferer(self):
        return self._inferer

    @property
    def input_dims(self):
        return self._input_dims

    @property
    def n_train_timesteps(self):
        return self.inferer.scheduler.num_train_timesteps

    def get_noised_input(self, x0, t, noise):
        return self.inferer.scheduler.add_noise(x0, noise, t)

    #  TODO - figure out what to do here
    def get_conditioning(self, cond_input):
        pass

    # TODO - consider deleting this function as the true obligatory is only the below
    def get_diffusion_noise_prediction(self, x0, t, conditioning):
        pass

    def get_noise_prediction(self, x0, t, noise, c):
        return self.inferer(inputs=x0,
                            autoencoder_model=self.autoencoder,
                            diffusion_model=self.unet,
                            noise=noise,
                            timesteps=t,
                            condition=c.view(-1, 1, 1).to(dtype=torch.float32))

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
        z_mu, z_sigma = self._autoencoder.encode(batch['image'])
        return z_mu

