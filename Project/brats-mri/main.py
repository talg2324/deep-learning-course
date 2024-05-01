import sys
import torch
import monai
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda.amp import  autocast

from pretrained import load_autoencoder, load_unet
import utils

BUNDLE = 'brats_mri_class_cond'
sys.path.append(BUNDLE)
from scripts.inferer import LatentDiffusionInfererWithClassConditioning
from scripts.utils import compute_scale_factor
from scripts.ct_rsna import CTSubset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

n_epochs = 10
lr = 1e-5
batch_size = 2

if __name__ == "__main__":

    monai.utils.set_determinism(seed=7)
    utils.download_weights_if_not_already(BUNDLE)
    train_loader = DataLoader(CTSubset('../data/ct-rsna/train/', 'train_set_dropped_nans.csv',
                                        256, 0.5, 8), batch_size=batch_size)

    # Initialize models
    autoencoder = load_autoencoder(BUNDLE)
    unet = load_unet(BUNDLE)

    # Train
    config = utils.model_config(BUNDLE, 'train_diffusion.json')
    scale_factor = compute_scale_factor(autoencoder, train_loader, device)

    scheduler = config.get_parsed_content('noise_scheduler')
    latent_shape = config.get_parsed_content('latent_shape')

    inferer = LatentDiffusionInfererWithClassConditioning(scheduler, scale_factor)

    optimizer = Adam(list(unet.parameters()) + list(autoencoder.parameters()), lr=lr)
    L = torch.nn.MSELoss().to(device)

    for e in range(1, n_epochs+1):
        total_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f"Epoch {e}")
        for step, batch in progress_bar:
            ims = batch['image'].to(device)
            labels = batch['class_label'].to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                # Generate random noise
                noise = torch.randn([batch_size] + latent_shape, device=device)

                # Create timesteps
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps, (ims.shape[0],), device=device
                ).long()

                # Get model prediction
                noise_pred = inferer(inputs=ims,
                                     autoencoder_model=autoencoder,
                                     diffusion_model=unet,
                                     noise=noise,
                                     timesteps=timesteps,
                                     class_labels=labels)

                loss = L(noise_pred.float(), noise.float())
                loss.backward()
                optimizer.step()
            total_loss += loss.item()

            progress_bar.set_postfix({"loss": total_loss / (step + 1)})
