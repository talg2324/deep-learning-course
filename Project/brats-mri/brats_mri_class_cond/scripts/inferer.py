import math
from collections.abc import Callable, Sequence
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.data import decollate_batch
from monai.inferers import Inferer
from monai.transforms import CenterSpatialCrop, SpatialPad
from monai.utils import optional_import

from generative.networks.nets import VQVAE, SPADEAutoencoderKL, SPADEDiffusionModelUNet

tqdm, has_tqdm = optional_import("tqdm", name="tqdm")


class DiffusionInfererWithClassConditioning(Inferer):
    """
    DiffusionInferer takes a trained diffusion model and a scheduler and can be used to perform a signal forward pass
    for a training iteration, and sample from the model.

    Args:
        scheduler: diffusion scheduler.
    """

    def __init__(self, scheduler: nn.Module) -> None:
        Inferer.__init__(self)
        self.scheduler = scheduler

    def __call__(
        self,
        inputs: torch.Tensor,
        diffusion_model,  #: Callable[..., torch.Tensor],
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        condition = None,  # :torch.Tensor | None = None,
        class_labels = None,  # :torch.Tensor | None = None,
        mode: str = "crossattn",
        seg = None  # :torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Implements the forward pass for a supervised training iteration.

        Args:
            inputs: Input image to which noise is added.
            diffusion_model: diffusion model.
            noise: random noise, of the same shape as the input.
            timesteps: random timesteps.
            condition: Conditioning for network input.
            mode: Conditioning mode for the network.
            seg: if model is instance of SPADEDiffusionModelUnet, segmentation must be
            provided on the forward (for SPADE-like AE or SPADE-like DM)
        """
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")

        noisy_image = self.scheduler.add_noise(original_samples=inputs, noise=noise, timesteps=timesteps)
        if mode == "concat":
            noisy_image = torch.cat([noisy_image, condition], dim=1)
            condition = None
        diffusion_model = (
            partial(diffusion_model, seg=seg)
            if isinstance(diffusion_model, SPADEDiffusionModelUNet)
            else diffusion_model
        )
        prediction = diffusion_model(x=noisy_image, timesteps=timesteps, context=condition, class_labels=class_labels)

        return prediction

    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        diffusion_model,  # : Callable[..., torch.Tensor],
        scheduler,  # : Callable[..., torch.Tensor] | None = None,
        save_intermediates = False, # : bool | None = False,
        intermediate_steps = 100, # : int | None = 100,
        conditioning = None,  # :torch.Tensor | None = None,
        class_labels = None,  # :torch.Tensor | None = None,
        mode: str = "crossattn",
        verbose: bool = True,
        seg = None  # :torch.Tensor | None = None,
    ):
        """
        Args:
            input_noise: random noise, of the same shape as the desired sample.
            diffusion_model: model to sample from.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            verbose: if true, prints the progression bar of the sampling process.
            seg: if diffusion model is instance of SPADEDiffusionModel, segmentation must be provided.
        """
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")

        if not scheduler:
            scheduler = self.scheduler
        image = input_noise
        if verbose and has_tqdm:
            progress_bar = tqdm(scheduler.timesteps)
        else:
            progress_bar = iter(scheduler.timesteps)
        intermediates = []
        for t in progress_bar:
            # 1. predict noise model_output
            diffusion_model = (
                partial(diffusion_model, seg=seg)
                if isinstance(diffusion_model, SPADEDiffusionModelUNet)
                else diffusion_model
            )
            if mode == "concat":
                model_input = torch.cat([image, conditioning], dim=1)
                model_output = diffusion_model(
                    model_input, timesteps=torch.Tensor((t,)).to(input_noise.device), context=None, class_labels=class_labels
                )
            else:
                model_output = diffusion_model(
                    image, timesteps=torch.Tensor((t,)).to(input_noise.device), context=conditioning, class_labels=class_labels
                )

            # 2. compute previous image: x_t -> x_t-1
            image, _ = scheduler.step(model_output, t, image)
            if save_intermediates and t % intermediate_steps == 0:
                intermediates.append(image)
        if save_intermediates:
            return image, intermediates
        else:
            return image

    @torch.no_grad()
    def get_likelihood(
        self,
        inputs: torch.Tensor,
        diffusion_model,  # : Callable[..., torch.Tensor],
        scheduler,  # : Callable[..., torch.Tensor] | None = None,
        save_intermediates = False,  # : bool | None = False,
        conditioning = None,  # :torch.Tensor | None = None,
        class_labels = None,  # :torch.Tensor | None = None,
        mode: str = "crossattn",
        original_input_range = (0, 255),  # : tuple | None = (0, 255),
        scaled_input_range = (0, 1),  # : tuple | None = (0, 1),
        verbose: bool = True,
        seg = None  # :torch.Tensor | None = None,
    ):
        """
        Computes the log-likelihoods for an input.

        Args:
            inputs: input images, NxCxHxW[xD]
            diffusion_model: model to compute likelihood from
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler.
            save_intermediates: save the intermediate spatial KL maps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            original_input_range: the [min,max] intensity range of the input data before any scaling was applied.
            scaled_input_range: the [min,max] intensity range of the input data after scaling.
            verbose: if true, prints the progression bar of the sampling process.
            seg: if diffusion model is instance of SPADEDiffusionModel, segmentation must be provided.
        """

        if not scheduler:
            scheduler = self.scheduler
        if scheduler._get_name() != "DDPMScheduler":
            raise NotImplementedError(
                f"Likelihood computation is only compatible with DDPMScheduler,"
                f" you are using {scheduler._get_name()}"
            )
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")
        if verbose and has_tqdm:
            progress_bar = tqdm(scheduler.timesteps)
        else:
            progress_bar = iter(scheduler.timesteps)
        intermediates = []
        noise = torch.randn_like(inputs).to(inputs.device)
        total_kl = torch.zeros(inputs.shape[0]).to(inputs.device)
        for t in progress_bar:
            timesteps = torch.full(inputs.shape[:1], t, device=inputs.device).long()
            noisy_image = self.scheduler.add_noise(original_samples=inputs, noise=noise, timesteps=timesteps)
            diffusion_model = (
                partial(diffusion_model, seg=seg)
                if isinstance(diffusion_model, SPADEDiffusionModelUNet)
                else diffusion_model
            )
            if mode == "concat":
                noisy_image = torch.cat([noisy_image, conditioning], dim=1)
                model_output = diffusion_model(noisy_image, timesteps=timesteps, context=None, class_labels=class_labels)
            else:
                model_output = diffusion_model(x=noisy_image, timesteps=timesteps, context=conditioning, class_labels=class_labels)

            # get the model's predicted mean,  and variance if it is predicted
            if model_output.shape[1] == inputs.shape[1] * 2 and scheduler.variance_type in ["learned", "learned_range"]:
                model_output, predicted_variance = torch.split(model_output, inputs.shape[1], dim=1)
            else:
                predicted_variance = None

            # 1. compute alphas, betas
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = scheduler.alphas_cumprod[t - 1] if t > 0 else scheduler.one
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev

            # 2. compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
            if scheduler.prediction_type == "epsilon":
                pred_original_sample = (noisy_image - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            elif scheduler.prediction_type == "sample":
                pred_original_sample = model_output
            elif scheduler.prediction_type == "v_prediction":
                pred_original_sample = (alpha_prod_t**0.5) * noisy_image - (beta_prod_t**0.5) * model_output
            # 3. Clip "predicted x_0"
            if scheduler.clip_sample:
                pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

            # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * scheduler.betas[t]) / beta_prod_t
            current_sample_coeff = scheduler.alphas[t] ** (0.5) * beta_prod_t_prev / beta_prod_t

            # 5. Compute predicted previous sample µ_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            predicted_mean = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * noisy_image

            # get the posterior mean and variance
            posterior_mean = scheduler._get_mean(timestep=t, x_0=inputs, x_t=noisy_image)
            posterior_variance = scheduler._get_variance(timestep=t, predicted_variance=predicted_variance)

            log_posterior_variance = torch.log(posterior_variance)
            log_predicted_variance = torch.log(predicted_variance) if predicted_variance else log_posterior_variance

            if t == 0:
                # compute -log p(x_0|x_1)
                kl = -self._get_decoder_log_likelihood(
                    inputs=inputs,
                    means=predicted_mean,
                    log_scales=0.5 * log_predicted_variance,
                    original_input_range=original_input_range,
                    scaled_input_range=scaled_input_range,
                )
            else:
                # compute kl between two normals
                kl = 0.5 * (
                    -1.0
                    + log_predicted_variance
                    - log_posterior_variance
                    + torch.exp(log_posterior_variance - log_predicted_variance)
                    + ((posterior_mean - predicted_mean) ** 2) * torch.exp(-log_predicted_variance)
                )
            total_kl += kl.view(kl.shape[0], -1).mean(axis=1)
            if save_intermediates:
                intermediates.append(kl.cpu())

        if save_intermediates:
            return total_kl, intermediates
        else:
            return total_kl

    def _approx_standard_normal_cdf(self, x):
        """
        A fast approximation of the cumulative distribution function of the
        standard normal. Code adapted from https://github.com/openai/improved-diffusion.
        """

        return 0.5 * (
            1.0 + torch.tanh(torch.sqrt(torch.Tensor([2.0 / math.pi]).to(x.device)) * (x + 0.044715 * torch.pow(x, 3)))
        )

    def _get_decoder_log_likelihood(
        self,
        inputs: torch.Tensor,
        means: torch.Tensor,
        log_scales: torch.Tensor,
        original_input_range = (0, 255),  # : tuple | None = (0, 255),
        scaled_input_range = (0, 1),  # : tuple | None = (0, 1),
    ) -> torch.Tensor:
        """
        Compute the log-likelihood of a Gaussian distribution discretizing to a
        given image. Code adapted from https://github.com/openai/improved-diffusion.

        Args:
            input: the target images. It is assumed that this was uint8 values,
                      rescaled to the range [-1, 1].
            means: the Gaussian mean Tensor.
            log_scales: the Gaussian log stddev Tensor.
            original_input_range: the [min,max] intensity range of the input data before any scaling was applied.
            scaled_input_range: the [min,max] intensity range of the input data after scaling.
        """
        assert inputs.shape == means.shape
        bin_width = (scaled_input_range[1] - scaled_input_range[0]) / (
            original_input_range[1] - original_input_range[0]
        )
        centered_x = inputs - means
        inv_stdv = torch.exp(-log_scales)
        plus_in = inv_stdv * (centered_x + bin_width / 2)
        cdf_plus = self._approx_standard_normal_cdf(plus_in)
        min_in = inv_stdv * (centered_x - bin_width / 2)
        cdf_min = self._approx_standard_normal_cdf(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            inputs < -0.999,
            log_cdf_plus,
            torch.where(inputs > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
        )
        assert log_probs.shape == inputs.shape
        return log_probs


class LatentDiffusionInfererWithClassConditioning(DiffusionInfererWithClassConditioning):
    """
    LatentDiffusionInferer takes a stage 1 model (VQVAE or AutoencoderKL), diffusion model, and a scheduler, and can
    be used to perform a signal forward pass for a training iteration, and sample from the model.

    Args:
        scheduler: a scheduler to be used in combination with `unet` to denoise the encoded image latents.
        scale_factor: scale factor to multiply the values of the latent representation before processing it by the
            second stage.
        ldm_latent_shape: desired spatial latent space shape. Used if there is a difference in the autoencoder model's latent shape.
        autoencoder_latent_shape:  autoencoder_latent_shape: autoencoder spatial latent space shape. Used if there is a
             difference between the autoencoder's latent shape and the DM shape.
    """

    def __init__(
        self,
        scheduler: nn.Module,
        scale_factor: float = 1.0,
        ldm_latent_shape = None,  # : list | None = None,
        autoencoder_latent_shape = None,  # : list | None = None,
    ) -> None:
        super().__init__(scheduler=scheduler)
        self.scale_factor = scale_factor
        if (ldm_latent_shape is None) ^ (autoencoder_latent_shape is None):
            raise ValueError("If ldm_latent_shape is None, autoencoder_latent_shape must be None" "and vice versa.")
        self.ldm_latent_shape = ldm_latent_shape
        self.autoencoder_latent_shape = autoencoder_latent_shape
        if self.ldm_latent_shape is not None:
            self.ldm_resizer = SpatialPad(spatial_size=self.ldm_latent_shape)
            self.autoencoder_resizer = CenterSpatialCrop(roi_size=self.autoencoder_latent_shape)

    def __call__(
        self,
        inputs: torch.Tensor,
        autoencoder_model,  # : Callable[..., torch.Tensor],
        diffusion_model,  # : Callable[..., torch.Tensor],
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        condition = None,  # :torch.Tensor | None = None,
        class_labels = None,  # :torch.Tensor | None = None,
        mode: str = "crossattn",
        seg = None,  # :torch.Tensor | None = None,
        quantized: bool = True,
    ) -> torch.Tensor:
        """
        Implements the forward pass for a supervised training iteration.

        Args:
            inputs: input image to which the latent representation will be extracted and noise is added.
            autoencoder_model: first stage model.
            diffusion_model: diffusion model.
            noise: random noise, of the same shape as the latent representation.
            timesteps: random timesteps.
            condition: conditioning for network input.
            mode: Conditioning mode for the network.
            seg: if diffusion model is instance of SPADEDiffusionModel, segmentation must be provided.
            quantized: if autoencoder_model is a VQVAE, quantized controls whether the latents to the LDM
            are quantized or not.
        """
        with torch.no_grad():
            autoencode = autoencoder_model.encode_stage_2_inputs
            if isinstance(autoencoder_model, VQVAE):
                autoencode = partial(autoencoder_model.encode_stage_2_inputs, quantized=quantized)
            latent = autoencode(inputs) * self.scale_factor

        if self.ldm_latent_shape is not None:
            latent = torch.stack([self.ldm_resizer(i) for i in decollate_batch(latent)], 0)

        call = super().__call__
        if isinstance(diffusion_model, SPADEDiffusionModelUNet):
            call = partial(super().__call__, seg=seg)

        prediction = call(
            inputs=latent,
            diffusion_model=diffusion_model,
            noise=noise,
            timesteps=timesteps,
            condition=condition,
            class_labels=class_labels,
            mode=mode,
        )
        return prediction

    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        autoencoder_model,  # : Callable[..., torch.Tensor],
        diffusion_model,  # : Callable[..., torch.Tensor],
        scheduler = None,  # : Callable[..., torch.Tensor] | None = None,
        save_intermediates = False,  # : bool | None = False,
        intermediate_steps = 100,  # : int | None = 100,
        conditioning = None,  # :torch.Tensor | None = None,
        class_labels = None,  # :torch.Tensor | None = None,
        mode: str = "crossattn",
        verbose: bool = True,
        seg = None  # :torch.Tensor | None = None,
    ):
        """
        Args:
            input_noise: random noise, of the same shape as the desired latent representation.
            autoencoder_model: first stage model.
            diffusion_model: model to sample from.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler.
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            verbose: if true, prints the progression bar of the sampling process.
            seg: if diffusion model is instance of SPADEDiffusionModel, or autoencoder_model
             is instance of SPADEAutoencoderKL, segmentation must be provided.
        """

        if (
            isinstance(autoencoder_model, SPADEAutoencoderKL)
            and isinstance(diffusion_model, SPADEDiffusionModelUNet)
            and autoencoder_model.decoder.label_nc != diffusion_model.label_nc
        ):
            raise ValueError(
                "If both autoencoder_model and diffusion_model implement SPADE, the number of semantic"
                "labels for each must be compatible. "
            )

        sample = super().sample
        if isinstance(diffusion_model, SPADEDiffusionModelUNet):
            sample = partial(super().sample, seg=seg)

        outputs = sample(
            input_noise=input_noise,
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            save_intermediates=save_intermediates,
            intermediate_steps=intermediate_steps,
            conditioning=conditioning,
            class_labels=class_labels,
            mode=mode,
            verbose=verbose,
        )

        if save_intermediates:
            latent, latent_intermediates = outputs
        else:
            latent = outputs

        if self.autoencoder_latent_shape is not None:
            latent = torch.stack([self.autoencoder_resizer(i) for i in decollate_batch(latent)], 0)
            if save_intermediates:
                latent_intermediates = [
                    torch.stack([self.autoencoder_resizer(i) for i in decollate_batch(l)], 0)
                    for l in latent_intermediates
                ]

        decode = autoencoder_model.decode_stage_2_outputs
        if isinstance(autoencoder_model, SPADEAutoencoderKL):
            decode = partial(autoencoder_model.decode_stage_2_outputs, seg=seg)
        image = decode(latent / self.scale_factor)

        if save_intermediates:
            intermediates = []
            for latent_intermediate in latent_intermediates:
                decode = autoencoder_model.decode_stage_2_outputs
                if isinstance(autoencoder_model, SPADEAutoencoderKL):
                    decode = partial(autoencoder_model.decode_stage_2_outputs, seg=seg)
                intermediates.append(decode(latent_intermediate / self.scale_factor))
            return image, intermediates

        else:
            return image

    @torch.no_grad()
    def get_likelihood(
        self,
        inputs: torch.Tensor,
        autoencoder_model,  # : Callable[..., torch.Tensor],
        diffusion_model,  # : Callable[..., torch.Tensor],
        scheduler = None,  # : Callable[..., torch.Tensor] | None = None,
        save_intermediates = False,  # : bool | None = False,
        conditioning = None,  # :torch.Tensor | None = None,
        class_labels = None,  # :torch.Tensor | None = None,
        mode: str = "crossattn",
        original_input_range = (0, 255),  # : tuple | None = (0, 255),
        scaled_input_range = (0, 1),  # : tuple | None = (0, 1),
        verbose: bool = True,
        resample_latent_likelihoods: bool = False,
        resample_interpolation_mode: str = "nearest",
        seg = None,  # :torch.Tensor | None = None,
        quantized: bool = True
    ):
        """
        Computes the log-likelihoods of the latent representations of the input.

        Args:
            inputs: input images, NxCxHxW[xD]
            autoencoder_model: first stage model.
            diffusion_model: model to compute likelihood from
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler
            save_intermediates: save the intermediate spatial KL maps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            original_input_range: the [min,max] intensity range of the input data before any scaling was applied.
            scaled_input_range: the [min,max] intensity range of the input data after scaling.
            verbose: if true, prints the progression bar of the sampling process.
            resample_latent_likelihoods: if true, resamples the intermediate likelihood maps to have the same spatial
                dimension as the input images.
            resample_interpolation_mode: if use resample_latent_likelihoods, select interpolation 'nearest', 'bilinear',
                or 'trilinear;
            seg: if diffusion model is instance of SPADEDiffusionModel, or autoencoder_model
             is instance of SPADEAutoencoderKL, segmentation must be provided.
            quantized: if autoencoder_model is a VQVAE, quantized controls whether the latents to the LDM
            are quantized or not.
        """
        if resample_latent_likelihoods and resample_interpolation_mode not in ("nearest", "bilinear", "trilinear"):
            raise ValueError(
                f"resample_interpolation mode should be either nearest, bilinear, or trilinear, got {resample_interpolation_mode}"
            )

        autoencode = autoencoder_model.encode_stage_2_inputs
        if isinstance(autoencoder_model, VQVAE):
            autoencode = partial(autoencoder_model.encode_stage_2_inputs, quantized=quantized)
        latents = autoencode(inputs) * self.scale_factor

        if self.ldm_latent_shape is not None:
            latents = torch.stack([self.ldm_resizer(i) for i in decollate_batch(latents)], 0)

        get_likelihood = super().get_likelihood
        if isinstance(diffusion_model, SPADEDiffusionModelUNet):
            get_likelihood = partial(super().get_likelihood, seg=seg)

        outputs = get_likelihood(
            inputs=latents,
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            save_intermediates=save_intermediates,
            conditioning=conditioning,
            class_labels=class_labels,
            mode=mode,
            verbose=verbose,
        )

        if save_intermediates and resample_latent_likelihoods:
            intermediates = outputs[1]
            resizer = nn.Upsample(size=inputs.shape[2:], mode=resample_interpolation_mode)
            intermediates = [resizer(x) for x in intermediates]
            outputs = (outputs[0], intermediates)
        return outputs