import sys
import torch
import monai
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda.amp import autocast, GradScaler
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
    print("Resuming training from checkpoint: ", args.resume_from_ckpt)
    print("Number of epochs: ", args.num_epochs)
    print(f"Performing validation every {args.val_every_n_epochs} epochs")
    if args.save_ckpt_every_n:
        print(f"saving checkpoints each {args.save_ckpt_every_n} epochs")
    else:
        args.save_ckpt_every_n = args.num_epochs
        print(f"saving only on last epoch")

    return args


def save_epoch(logdir: str, epoch: int, autoencoder, unet, losses_dict: dict):
    autoencoder_ckpt_path = os.path.join(logdir, f"autoencoder_epoch_{epoch}.ckpt")
    diffusion_ckpt_path = os.path.join(logdir, f"diffusion_epoch_{epoch}.ckpt")
    losses_path = os.path.join(logdir, f"losses_dict_epoch_{epoch}")
    print("Saving checkpoint at epoch ", epoch)

    # print(f"autoencoder save path: {autoencoder_ckpt_path}")
    # print(f"diffusion save path: {diffusion_ckpt_path}")
    # print(f"losses list save path: {losses_path}")

    # Save the checkpoint
    torch.save(autoencoder.state_dict(), autoencoder_ckpt_path)
    torch.save(unet.state_dict(), diffusion_ckpt_path)
    with open(losses_path, 'wb') as f:
        pickle.dump(losses_dict, f)


if __name__ == "__main__":
    args = run_init()

    logdir = os.path.join(BUNDLE, 'models', args.name)
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    print(f"training logs dir: {logdir}")

    train_loader = DataLoader(CTSubset('../data/ct-rsna/train/', 'train_set_dropped_nans.csv',
                                        256, 0.5, 8), batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(CTSubset('../data/ct-rsna/validation/', 'validation_set_dropped_nans.csv',
                                       256, 0., 8), batch_size=batch_size, shuffle=False)

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
    lr_scheduler = MultiStepLR(optimizer, milestones=[1000], gamma=0.1)

    # TODO - the autoencoder does not train with MSE,
    #  also the diffusion might have more complex loss components, we should be careful here
    L = torch.nn.MSELoss().to(device)

    losses = {
        'train': [],
        'validation': []
    }

    # TODO - do we need grad scaler?
    scaler = GradScaler()

    for e in range(1, args.num_epochs+1):
        autoencoder.train()
        unet.train()

        total_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f"Epoch {e} - train")
        # training loop
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
                lr_scheduler.step()

            total_loss += loss.item()

            progress_bar.set_postfix({"loss": total_loss / (step + 1)})

        # normalize loss by number of batches
        total_loss /= step

        losses['train'].append(total_loss / step)

        # validation loop
        if e % args.val_every_n_epochs == 0:
            autoencoder.eval()
            unet.eval()

            val_loss = 0
            with torch.no_grad():
                val_progress_bar = tqdm(enumerate(val_loader), total=len(val_loader))
                val_progress_bar.set_description(f"Epoch {e} - validation")
                for val_step, batch in val_progress_bar:
                    ims = batch['image'].to(device)
                    labels = batch['class_label'].to(device)

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

                    val_loss += loss.item()
                    val_progress_bar.set_postfix({"val loss": val_loss / (val_step + 1)})

            # normalize loss by number of batches
            val_loss /= val_step

            losses['validation'].append((e, val_loss))

        # Check if it's time to save the checkpoint
        if e % args.save_ckpt_every_n == 0:
            save_epoch(logdir=logdir,
                       epoch=e,
                       autoencoder=autoencoder,
                       unet=unet,
                       losses_dict=losses)


    # progress_bar.close()
    # val_progress_bar.close()

    torch.cuda.empty_cache()
