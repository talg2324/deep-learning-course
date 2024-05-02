import sys
import torch
import monai
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda.amp import autocast
import os
import pickle

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


def run_init():
    arg_parser = utils.create_arg_parser()
    args = arg_parser.parse_args()

    monai.utils.set_determinism(seed=args.seed)
    utils.download_weights_if_not_already(BUNDLE)

    print("Started training Model:", args.name)
    print("Random seed: ", args.seed)
    print("Config json: ", args.config)
    print("Number of epochs: ", args.num_epochs)
    print("Resuming training from checkpoint: ", args.resume_from_ckpt)
    if args.save_ckpt_every_n:
        print(f"saving checkpoints each {args.save_ckpt_every_n} epochs")
    else:
        args.save_ckpt_every_n = args.num_epochs - 1
        print(f"saving only on last epoch")

    return args



if __name__ == "__main__":
    args = run_init()

    logdir = os.path.join(BUNDLE, 'models', args.name)
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    print(f"training logs dir: {logdir}")

    train_loader = DataLoader(CTSubset('../data/ct-rsna/train/', 'train_set_dropped_nans.csv',
                                        256, 0.5, 8), batch_size=batch_size)

    # Initialize models
    autoencoder = load_autoencoder(bundle_target=BUNDLE,
                                   override_model_cfg_json=args.config,
                                   override_weights_load_path=args.resume_from_ckpt)
    unet = load_unet(bundle_target=BUNDLE,
                     override_model_cfg_json=args.config,
                     override_weights_load_path=args.resume_from_ckpt)

    # Train
    # TODO - notice that the config below contains also model architecture params...
    #  consider removing / using same config also to load the models above
    config = utils.model_config(BUNDLE, 'train_diffusion.json')
    scale_factor = compute_scale_factor(autoencoder, train_loader, device)

    scheduler = config.get_parsed_content('noise_scheduler')
    latent_shape = config.get_parsed_content('latent_shape')

    # TODO - use the latent shape correctly - autoencoder / ldm ?
    inferer = LatentDiffusionInfererWithClassConditioning(scheduler=scheduler,
                                                          scale_factor=scale_factor,
                                                          ldm_latent_shape=None,
                                                          autoencoder_latent_shape=None)

    optimizer = Adam(list(unet.parameters()) + list(autoencoder.parameters()), lr=lr)

    # TODO - the autoencoder does not train with MSE,
    #  also the diffusion might have more complex loss components, we should be careful here
    L = torch.nn.MSELoss().to(device)

    losses = {
        'total': [],

    }
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

        losses['total'].append(total_loss)
        # Check if it's time to save the checkpoint
        if e % args.save_ckpt_every_n == 0 and e > 0:
            autoencoder_ckpt_path = os.path.join(logdir, f"{args.name}_autoencoder_epoch_{e}.ckpt")
            diffusion_ckpt_path = os.path.join(logdir, f"{args.name}_diffusion_epoch_{e}.ckpt")
            losses_path = os.path.join(logdir, f"{args.name}_losses_dict_epoch_{e}")
            print("Saving checkpoint at epoch ", e)
            print(f"autoencoder save path: {autoencoder_ckpt_path}")
            print(f"diffusion save path: {diffusion_ckpt_path}")
            print(f"losses list save path: {}")
            # Save the checkpoint
            torch.save(autoencoder.state_dict(), autoencoder_ckpt_path)
            torch.save(unet.state_dict(), diffusion_ckpt_path)
            with (losses_path, 'wb') as f:
                pickle.dump(losses, f)
