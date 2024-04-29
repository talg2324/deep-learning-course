import torch
import sys
import tqdm

sys.path.append('latent-diffusion')
sys.path.append('latent-diffusion/ldm/data/')

from ldm.models.diffusion.ddpm import LatentDiffusion
from torch.utils.data import Dataset


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class LdmClassifier:
    def __init__(self,
                 model: LatentDiffusion,
                 n_noise_samples: int = 1024):
        self._model = model
        self._n_noise_samples = n_noise_samples

        self.gen_noise()

    @property
    def model(self):
        return self._model

    @property
    def n_noise_samples(self):
        return self._n_noise_samples

    @property
    def input_dims(self):
        return self._model.channels, self._model.image_size, self._model.image_size

    def gen_noise(self):
        c, h, w = self.input_dims
        self._noise = torch.randn(self.n_noise_samples, c, h, w)

    @property
    def noise(self):
        return self._noise

    @property
    def n_train_timesteps(self):
        return self._model.num_timesteps

    @staticmethod
    def get_class_hypotheses_for_batch(batch_size, n_classes: int = 6):
        c_hypotheses = [{'class_label': torch.tensor([c] * batch_size)} for c in range(n_classes)]
        return c_hypotheses

    def get_latent_batch(self, batch, n_classes: int = 6):
        x0, c_true = self._model.get_input(batch, self._model.first_stage_key)
        c_hypotheses = self.get_class_hypotheses_for_batch(batch_size=x0.shape[0])
        return x0, c_hypotheses

    @staticmethod
    def get_classification_accuracy(pred_list, gt_list):
        n_samples = float(len(pred_list))
        n_correct = 0.
        for y_hat, y_gt in zip(pred_list, gt_list):
            if y_hat == y_gt:
                n_correct += 1.
        return 100. * (n_correct / n_samples)

    def classify_dataset(self,
                         dataset: Dataset,
                         batch_size: int = 1,
                         n_trials: int = 1,
                         t_sampling_stride: int = 5):
        # TODO - add here verifications to the Dataset object to make sure it has the required attributes
        n_classes = len(dataset.labels)

        assert batch_size == 1, "classifier batch size refers to a batch in the t_timesteps dimension. but it can only work on one sample at each call. dataloader batch must be set to 1"
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        l2_labels_pred = []
        l1_labels_pred = []
        true_labels = []
        for batch in tqdm.auto.tqdm(loader, desc="dataset samples"):
            x0, c_hypotheses = self.get_latent_batch(batch, n_classes)
            l2_label_pred, l1_label_pred = self.classify_batch(x0, c_hypotheses, n_trials, t_sampling_stride)

            true_labels.extend(batch['class_label'])
            l2_labels_pred.append(l2_label_pred)
            l1_labels_pred.append(l1_label_pred)

        return l2_labels_pred, l1_labels_pred, true_labels

    def classify_batch(self,
                       x0,
                       c_hypotheses,
                       n_trials: int = 1,
                       t_sampling_stride: int = 5):
        l1_c_errs = []
        l2_c_errs = []
        for c_hypo in tqdm.auto.tqdm(c_hypotheses, desc="class hypothsis"):
            l2_mean_err, l1_mean_err = self.eval_class_hypothesis_per_batch(x0=x0,
                                                                            c=c_hypo,
                                                                            n_trials=n_trials,
                                                                            t_sampling_stride=t_sampling_stride)
            l2_c_errs.append((l2_mean_err, c_hypo['class_label']))
            l1_c_errs.append((l1_mean_err, c_hypo['class_label']))

        l2_label_pred = self.extract_prediction_from_errs(l2_c_errs)
        l1_label_pred = self.extract_prediction_from_errs(l1_c_errs)
        return l2_label_pred, l1_label_pred

    @staticmethod
    def extract_prediction_from_errs(errs_and_labels_list):
        sorted_errs_and_labels = sorted(errs_and_labels_list, key=lambda x: x[0])
        pred_label = sorted_errs_and_labels[0][1]
        return pred_label

    @staticmethod
    def trim_cond_dict(cond_dict, length):
        for k, v in cond_dict.items():
            if len(v) > length:
                cond_dict[k] = v[:length]
        return cond_dict

    def eval_class_hypothesis_per_batch(self,
                                        x0,
                                        c,
                                        n_trials: int = 1,
                                        t_sampling_stride: int = 5,
                                        output_dtype: str = 'float32'):
        ts = []
        batch_size = x0.shape[0]
        pred_errors = torch.zeros(self.n_train_timesteps // t_sampling_stride * n_trials, 2, device='cpu')

        for t in range(t_sampling_stride // 2, self.n_train_timesteps, t_sampling_stride):
            ts.extend([t] * n_trials)
        with torch.inference_mode():
            idx = 0
            for _ in tqdm.auto.trange(len(ts) // batch_size + int(len(ts) % batch_size != 0),
                                      desc="diffusion sampling"):
                batch_ts = torch.tensor(ts[idx: idx + batch_size]).to(self._model.device)

                noise = self.noise[:len(batch_ts)]
                # TODO - what is latent object on our code?
                latent_ = x0.repeat(len(batch_ts), 1, 1, 1)

                # TODO - what is diffusion object on our code?
                noised_latent = self._model.q_sample(latent_, batch_ts, noise)

                t_input = batch_ts.to(self._model.device).half() if output_dtype == 'float16' else batch_ts.to(
                    self._model.device)
                cond_input = self.trim_cond_dict(c, len(batch_ts))

                # get condition embedding
                cond_emb = self._model.get_learned_conditioning(cond_input)

                # prepare condition embedding dict in correct form for diffusion_model.forward
                key = 'c_concat' if self._model.model.conditioning_key == 'concat' else 'c_crossattn'
                cond_emb = {key: [cond_emb]}

                model_output = self._model.model(noised_latent, t_input, **cond_emb)

                # B, C = noised_latent.shape[:2]

                ### NOTICE - here I made a change that I do not understand. we must understand if our model outputs a different output than DiT_XL_2 that was originally used. ###
                ###  on one side, it seems that our model is supposed to return x_recon, and not noise prediction. but looking at the results, I am getting the impression it is indeed noise pred...

                # noise_pred, model_var_values = torch.split(model_output, C, dim=1)
                noise_pred = model_output

                ###################### VB - non functional - commented out ATM ###############

                # # compute MSE
                # mse = mean_flat((noise - noise_pred) ** 2)
                # l1_loss = mean_flat(torch.abs(noise - noise_pred))

                # # VB
                # true_mean, _, true_log_variance_clipped = model.q_posterior(x_start=latent_, x_t=noised_latent, t=t_input)

                # (
                #     model_mean,
                #     posterior_variance,
                #     posterior_log_variance,
                #     model_out
                # ) = diffusion.p_mean_variance(
                #     x=noised_latent,
                #     c=cond_emb,
                #     t=t_input,
                #     clip_denoised=False,
                #     return_model_output=True
                #     )

                # kl = normal_kl(
                #     true_mean, true_log_variance_clipped, model_mean, posterior_log_variance
                # )
                # kl = mean_flat(kl) / np.log(2.0)

                # # NLL
                # decoder_nll = -discretized_gaussian_log_likelihood(
                #     latent_, means=model_mean, log_scales=0.5 * posterior_log_variance
                # )
                # decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

                # error = torch.cat([mse.unsqueeze(1),
                #                     l1_loss.unsqueeze(1),
                #                     kl.unsqueeze(1),
                #                     decoder_nll.unsqueeze(1)], dim=1)

                l2_loss = mean_flat((noise - noise_pred) ** 2)
                l1_loss = mean_flat(torch.abs(noise - noise_pred))
                error = torch.cat([l2_loss.unsqueeze(1),
                                   l1_loss.unsqueeze(1)], dim=1)

                pred_errors[idx: idx + len(batch_ts)] = error.detach().cpu()
                idx += len(batch_ts)
        mean_pred_errors = pred_errors.view(self.n_train_timesteps // t_sampling_stride,
                                            n_trials,
                                            *pred_errors.shape[1:]).mean(dim=(0, 1))
        l2_mean_err = mean_pred_errors[0]
        l1_mean_err = mean_pred_errors[1]
        return l2_mean_err, l1_mean_err

