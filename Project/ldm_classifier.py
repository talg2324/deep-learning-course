from typing import List, Tuple

import torch
import sys
import tqdm
import numpy as np

sys.path.append('latent-diffusion')
sys.path.append('latent-diffusion/ldm/data/')

from ldm.models.diffusion.ddpm import LatentDiffusion
from generative.networks.nets import DiffusionModelUNet, AutoencoderKL

from torch.utils.data import Dataset
from torch.cuda.amp import autocast


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class DiffusionClassifierInterface:
    def __init__(self,
                 device,
                 n_noise_samples: int = 1024,
                 random_seed: int = 42,):
        self._device = device
        self._n_noise_samples = n_noise_samples
        self._random_seed = random_seed
        self.gen_noise()

    @property
    def device(self):
        return self._device

    @property
    def n_noise_samples(self):
        return self._n_noise_samples

    @property
    def input_dims(self):
        """
        returns the input dims for the model CxHxW (ignoring the batch dimension)
        """
        raise NotImplementedError

    def gen_noise(self):
        """
        generate noise for diffusion classification
        """
        c, h, w = self.input_dims
        self._noise = torch.randn(self.n_noise_samples, c, h, w,
                                  generator=torch.Generator().manual_seed(self._random_seed)).to(self._device)

    @property
    def noise(self):
        return self._noise

    @property
    def n_train_timesteps(self):
        raise NotImplementedError

    def get_noised_input(self, x0, t, noise):
        raise NotImplementedError

    def get_conditioning(self, cond_input):
        raise NotImplementedError

    def get_noise_prediction(self, x0, t, noise, c):
        raise NotImplementedError

    def get_class_hypotheses_for_batch(self, batch_size: int, classes: List[int]):
        """
        creates a list of mock labels (for each valid class hypothesis).
        this mock conditioning will be inserted to the diffusion model as class conditioning

        :param batch_size: number of input samples in batch
        :param classes: list of class hypotheses.
        """
        c_hypotheses = [{'class_label': torch.tensor([c] * batch_size, device=self._device)} for c in classes]
        return c_hypotheses
    
    def classify_dataset(self,
                         dataset: Dataset,
                         batch_size: int = 1,
                         n_trials: int = 1,
                         t_sampling_stride: int = 5,
                         classes: List[int] = None):
        """
        perform classification for a given dataset using the latent diffusion model as classifier

        :param dataset: dataset class object to iterate over and produce predictions
        :param batch_size: TODO - need to revisit this param. currently set to 1.
                            the source code does not seem to support inserting multiple different samples to the classifier.
                            the batch seems to refer to temporal dimension of the diffusion
        :param n_trials: number of trials to do for each sample. TODO - need to revisit how this is different than the batch...
        :param t_sampling_stride: sampling rate of the diffusion time steps
        :param classes: classes hypotheses to use for classification. if None is passed, all dataset labels are used

        :returns : List[L2 loss predictions], List[L1 loss predictions], List[gt]
        """
        # TODO - add here verifications to the Dataset object to make sure it has the required attributes
        if classes is None:
            classes = np.unique(dataset.labels).tolist

        assert batch_size == 1, "classifier batch size refers to a batch in the t_timesteps dimension. but it can only work on one sample at each call. dataloader batch must be set to 1"
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        l2_labels_pred = []
        l1_labels_pred = []
        true_labels = []
        for batch in tqdm.auto.tqdm(loader, desc="dataset samples"):
            c_hypotheses = self.get_class_hypotheses_for_batch(batch_size=batch_size, classes=classes)
            l2_label_pred, l1_label_pred = self.classify_batch(batch, c_hypotheses, n_trials, t_sampling_stride)

            true_labels.extend(batch['class_label'])
            l2_labels_pred.append(l2_label_pred)
            l1_labels_pred.append(l1_label_pred)

        return l2_labels_pred, l1_labels_pred, true_labels

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

        l1_c_errs = []
        l2_c_errs = []
        for c_hypo in tqdm.auto.tqdm(c_hypotheses, desc="class hypothsis"):
            l2_mean_err, l1_mean_err = self.eval_class_hypothesis_per_batch(x0=x0,
                                                                            c=c_hypo,
                                                                            n_trials=n_trials,
                                                                            t_sampling_stride=t_sampling_stride)
            l2_c_errs.append((l2_mean_err, c_hypo['class_label'].detach().cpu()))
            l1_c_errs.append((l1_mean_err, c_hypo['class_label'].detach().cpu()))

        l2_label_pred = self.extract_prediction_from_errs(l2_c_errs)
        l1_label_pred = self.extract_prediction_from_errs(l1_c_errs)
        return l2_label_pred, l1_label_pred

    def eval_class_hypothesis_per_batch(self,
                                        x0,
                                        c,
                                        n_trials: int = 1,
                                        t_sampling_stride: int = 5):
        """
        runs the core algorithm of the classifier.
        calculates noise prediction errors given the conditioning c.
        returns noise prediction errors
        TODO - currently L2, L1 only. VLB is not supported as LatentDiffusion does not predict the variance

        :param x0: diffusion model input - latent representation of the input batch
        :param c: conditioning
        :param n_trials: number of trials to do for each sample. TODO - need to revisit how this is different than the batch...
        :param t_sampling_stride: sampling rate of the diffusion time steps
        """
        ts = []
        batch_size = x0.shape[0]
        pred_errors = torch.zeros(self.n_train_timesteps // t_sampling_stride * n_trials, 2, device='cpu')

        for t in range(t_sampling_stride // 2, self.n_train_timesteps, t_sampling_stride):
            ts.extend([t] * n_trials)
        with torch.inference_mode():
            idx = 0
            for _ in tqdm.auto.trange(len(ts) // batch_size + int(len(ts) % batch_size != 0),
                                      desc="diffusion sampling"):
                with autocast(enabled=True):
                    t_input = torch.tensor(ts[idx: idx + batch_size]).to(self.device)

                    noise = self.noise[:len(t_input)]

                    x0 = x0.repeat(len(t_input), 1, 1, 1)

                    noise_pred = self.get_noise_prediction(x0, t_input, noise, c)
                    # noised_latent = self.get_noised_input(latent_, t_input, noise)
                    #
                    #
                    # # prepare conditioning input
                    # cond_input = self.trim_cond_dict(c, len(t_input))
                    #
                    # # get condition embedding
                    # cond_emb = self.get_conditioning(cond_input)
                    #
                    # # get noise prediction from diffusion model
                    # noise_pred = self.get_diffusion_noise_prediction(noised_latent, t_input, cond_emb)

                    l2_loss = mean_flat((noise - noise_pred) ** 2)
                    l1_loss = mean_flat(torch.abs(noise - noise_pred))
                    error = torch.cat([l2_loss.unsqueeze(1),
                                       l1_loss.unsqueeze(1)], dim=1)

                    pred_errors[idx: idx + len(t_input)] = error.detach().cpu()
                    idx += len(t_input)
        mean_pred_errors = pred_errors.view(self.n_train_timesteps // t_sampling_stride,
                                            n_trials,
                                            *pred_errors.shape[1:]).mean(dim=(0, 1))

        l2_mean_err = mean_pred_errors[0]
        l1_mean_err = mean_pred_errors[1]
        return l2_mean_err, l1_mean_err

    @staticmethod
    def extract_prediction_from_errs(errs_and_labels_list):
        """
        extract label prediction from noise errors of the diffusion model (output of eval_class_hypothesis_per_batch)
        """
        sorted_errs_and_labels = sorted(errs_and_labels_list, key=lambda x: x[0])
        pred_label = sorted_errs_and_labels[0][1]
        return pred_label

    @staticmethod
    def trim_cond_dict(cond_dict, length):
        for k, v in cond_dict.items():
            if len(v) > length:
                cond_dict[k] = v[:length]
        return cond_dict

    @staticmethod
    def get_classification_accuracy(pred_list, gt_list):
        """
        calculate classification accuracy
        """
        n_samples = float(len(pred_list))
        n_correct = 0.
        for y_hat, y_gt in zip(pred_list, gt_list):
            if y_hat == y_gt:
                n_correct += 1.
        return 100. * (n_correct / n_samples)


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

