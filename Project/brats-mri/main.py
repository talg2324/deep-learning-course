import os
import sys
import pickle
import torch
import monai
from PIL import Image
from tqdm import tqdm
from monai.utils import first
from generative.inferers import LatentDiffusionInferer
from generative.networks.schedulers import DDIMScheduler

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from pretrained import load_autoencoder, load_unet
import utils

BUNDLE = 'brats_mri_class_cond'
sys.path.append(BUNDLE)
from scripts.inferer import LatentDiffusionInfererWithClassConditioning
from scripts.utils import compute_scale_factor
from scripts.ct_rsna import CTSubset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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

    # Save the checkpoint
    torch.save(autoencoder.state_dict(), autoencoder_ckpt_path)
    torch.save(unet.state_dict(), diffusion_ckpt_path)
    with open(losses_path, 'wb') as f:
        pickle.dump(losses_dict, f)


def train_loop(unet, autoencoder, inferer, dl, L, optimizer, use_context, noise_shape):
    unet.train()
    autoencoder.train()
    total_loss = 0
    with tqdm(dl, desc='  Training loop', total=len(dl)) as pbar:
        for batch in pbar:
            ims = batch['image'].to(device)
            labels = batch['class_label'].to(device=device)
            noise = torch.randn(noise_shape, device=device)
            timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps,
                                        (ims.shape[0],), device=device).long()
            if use_context:
                if not use_conditioning:
                    condition = None
                else:
                    condition = labels.view(-1, 1, 1).to(dtype=torch.float32)
                noise_pred = inferer(inputs=ims,
                                    autoencoder_model=autoencoder,
                                    diffusion_model=unet,
                                    noise=noise,
                                    timesteps=timesteps,
                                    condition=condition)
            else:
                noise_pred = inferer(inputs=ims,
                                    autoencoder_model=autoencoder,
                                    diffusion_model=unet,
                                    noise=noise,
                                    timesteps=timesteps,
                                    class_labels=labels)
            loss = L(noise_pred.float(), noise.float())
            pbar.set_postfix({"loss": loss.item()})

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / len(dl)
        pbar.set_postfix({"loss": avg_loss})
    return avg_loss


def val_loop(unet, autoencoder, inferer, dl, L, use_context, noise_shape):
    unet.eval()
    autoencoder.eval()
    total_loss = 0
    with torch.no_grad():
        with tqdm(dl, desc='  Validation loop', total=len(dl)) as pbar:
            for batch in pbar:
                ims = batch['image'].to(device)
                labels = batch['class_label'].to(device)
                noise = torch.randn(noise_shape, device=device)
                timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps,
                                          (ims.shape[0],), device=device).long()
                if use_context:
                    if not use_conditioning:
                        condition = None
                    else:
                        condition = labels.view(-1, 1, 1).to(dtype=torch.float32)
                    noise_pred = inferer(inputs=ims,
                                        autoencoder_model=autoencoder,
                                        diffusion_model=unet,
                                        noise=noise,
                                        timesteps=timesteps,
                                        condition=condition)
                else:
                    noise_pred = inferer(inputs=ims,
                                        autoencoder_model=autoencoder,
                                        diffusion_model=unet,
                                        noise=noise,
                                        timesteps=timesteps,
                                        class_labels=labels)
                loss = L(noise_pred.float(), noise.float())
                total_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
        avg_loss = total_loss / len(dl)
        pbar.set_postfix({"loss": avg_loss})
    return avg_loss


def log_ims(unet, autoencoder, inferer, noise_shape,
            im_tag, data_sample, n_classes=6, max_ims=4):

    unet.eval()
    autoencoder.eval()

    max_ims = min(max_ims, len(data_sample['image']))

    input_ims = data_sample['image'].to(device)[:max_ims]
    encode_decode, _, _ = autoencoder(input_ims)

    input_col = torch.vstack([im.squeeze() for im in input_ims])
    output_col = torch.vstack([im.squeeze() for im in encode_decode])
    log_im = torch.hstack((input_col, output_col)).clamp(0, 1) * 255

    log_im = log_im.to(torch.uint8).cpu().numpy()
    log_im = Image.fromarray(log_im)
    log_im.save(im_tag + '_encoder.png')

    scheduler = DDIMScheduler(num_train_timesteps=1000,
                              beta_start=0.0015, beta_end=0.0195,
                              schedule='scaled_linear_beta', clip_sample=False)
    scheduler.set_timesteps(num_inference_steps=50)
    rows = []
    for n in range(4):
        noise = torch.randn(noise_shape, device=device)
        # TODO - better pass this as argument and not use as global
        if use_context:
            if not use_conditioning:
                label = None
            else:
                label = torch.full((1, 1, 1), n, dtype=torch.float32, device=device)
            _, images = inferer.sample(input_noise=noise,
                                    save_intermediates=True,
                                    intermediate_steps=250,
                                    autoencoder_model=autoencoder,
                                    diffusion_model=unet,
                                    scheduler=scheduler,
                                    conditioning=label)
        else:
            label = torch.full((1,), n, dtype=torch.long, device=device)
            _, images = inferer.sample(input_noise=noise,
                                    save_intermediates=True,
                                    intermediate_steps=250,
                                    autoencoder_model=autoencoder,
                                    diffusion_model=unet,
                                    scheduler=scheduler,
                                    class_labels=label)   

        row = torch.cat(images, dim=-1).squeeze() # H x (4xW)
        rows.append(row)
    rows = torch.vstack(rows).clamp(0, 1) * 255
    log_im = Image.fromarray(rows.to(torch.uint8).cpu().numpy())
    log_im.save(im_tag + '_sample.png')


if __name__ == "__main__":
    args = run_init()

    logdir = os.path.join('../data/', 'outputs', args.name)
    im_log = os.path.join(logdir, 'image_logs')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
        os.mkdir(im_log)

    print(f"Training dir: {logdir}")

    train_loader = DataLoader(CTSubset('../data/ct-rsna/train/', 'train_set_dropped_nans.csv',
                                        size=256, flip_prob=0.5, subset_len=64),
                                        batch_size=args.batch_size, shuffle=True, drop_last=True)

    val_loader = DataLoader(CTSubset('../data/ct-rsna/validation/', 'validation_set_dropped_nans.csv',
                                        size=256, flip_prob=0., subset_len=64),
                                        batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Initialize models
    autoencoder = load_autoencoder(bundle_target=BUNDLE,
                                   override_model_cfg_json=args.config,
                                   override_weights_load_path=args.resume_from_ckpt)

    config = utils.model_config(BUNDLE, 'train_diffusion.json')
    scale_factor = compute_scale_factor(autoencoder, train_loader, device)

    scheduler = config.get_parsed_content('noise_scheduler')
    latent_shape = config.get_parsed_content('latent_shape')

    use_conditioning = args.conditioning != 'none'
    if args.conditioning in ['context', 'none']:
        use_context = True
        inferer = LatentDiffusionInferer(scheduler=scheduler, scale_factor=scale_factor)
    else:
        use_context = False
        inferer = LatentDiffusionInfererWithClassConditioning(scheduler=scheduler, scale_factor=scale_factor)
    
    unet = load_unet(bundle_target=BUNDLE,
                     use_conditioning=use_conditioning,
                     context_conditioning=use_context,
                     override_model_cfg_json=args.config,
                     override_weights_load_path=args.resume_from_ckpt,
                     )

    optimizer = Adam(list(unet.parameters()), lr=args.lr)
    lr_scheduler = CosineAnnealingLR(optimizer, args.num_epochs, eta_min=1e-6)

    # TODO - the autoencoder does not train with MSE,
    L = torch.nn.MSELoss().to(device)
    train_noise_shape = [args.batch_size] + latent_shape
    sample_noise_shape = [1] + latent_shape

    losses = {
        'train': [],
        'validation': []
    }

    im_tag = os.path.join(im_log, '000')
    log_ims(unet, autoencoder, inferer, sample_noise_shape,
            im_tag, first(val_loader), len(train_loader.dataset.class_names))

    for e in range(1, args.num_epochs+1):
        print(f'Epoch #[{e}/{args.num_epochs}]:')
        
        
        train_loss = train_loop(unet, autoencoder, inferer, train_loader,
                                L, optimizer, use_context, train_noise_shape)
        
        losses['train'].append((e, train_loss))
        lr_scheduler.step()

        if e % args.val_every_n_epochs == 0:
            val_loss = val_loop(unet, autoencoder, inferer, val_loader,
                                    L, use_context, train_noise_shape)

            losses['validation'].append((e, val_loss))
            im_tag = os.path.join(im_log, f'{e}'.zfill(3))
            log_ims(unet, autoencoder, inferer, sample_noise_shape,
                   im_tag, first(val_loader), len(train_loader.dataset.class_names))

        if e % args.save_ckpt_every_n == 0:
            save_epoch(logdir=logdir,
                       epoch=e,
                       autoencoder=autoencoder,
                       unet=unet,
                       losses_dict=losses)
    torch.cuda.empty_cache()
