import os
import sys
import pickle
import torch
import monai
from PIL import Image
from tqdm import tqdm
from generative.inferers import LatentDiffusionInferer

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda.amp import autocast, GradScaler

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


def train_loop(unet, autoencoder, inferer, dl, L, optimizer, scaler, use_context, noise_shape):
    unet.train()
    # autoencoder.train()
    total_loss = 0
    with tqdm(dl, desc='  Training loop', total=len(dl)) as pbar:
        for batch in pbar:
            ims = batch['image'].to(device)
            labels = batch['class_label'].to(device=device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):

                noise = torch.randn(noise_shape, device=device)
                timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps,
                                          (ims.shape[0],), device=device).long()
                if use_context:
                    noise_pred = inferer(inputs=ims,
                                        autoencoder_model=autoencoder,
                                        diffusion_model=unet,
                                        noise=noise,
                                        timesteps=timesteps,
                                        condition=labels.view(-1, 1, 1).to(dtype=torch.float32))
                else:
                    noise_pred = inferer(inputs=ims,
                                        autoencoder_model=autoencoder,
                                        diffusion_model=unet,
                                        noise=noise,
                                        timesteps=timesteps,
                                        class_labels=labels)
                loss = L(noise_pred.float(), noise.float())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
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
                    noise_pred = inferer(inputs=ims,
                                        autoencoder_model=autoencoder,
                                        diffusion_model=unet,
                                        noise=noise,
                                        timesteps=timesteps,
                                        condition=labels.view(-1, 1, 1).to(dtype=torch.float32))
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


def sample(unet, autoencoder, inferer, scheduler, noise_shape, im_log_path, n_classes=6):
    unet.eval()
    autoencoder.eval()
    scheduler.set_timesteps(num_inference_steps=1000)

    rows = []
    for n in range(n_classes):
        noise = torch.randn(noise_shape, device=device)
        with autocast(enabled=True):
            if use_context:
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
    log_im.save(im_log_path)


if __name__ == "__main__":
    args = run_init()

    logdir = os.path.join('../data/', 'outputs', args.name)
    im_log = os.path.join(logdir, 'image_logs')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
        os.mkdir(im_log)

    print(f"Training dir: {logdir}")

    train_loader = DataLoader(CTSubset('../data/ct-rsna/train/', 'train_set_dropped_nans.csv',
                                        size=256, flip_prob=0.5, subset_len=2048),
                                        batch_size=args.batch_size, shuffle=True, drop_last=True)

    val_loader = DataLoader(CTSubset('../data/ct-rsna/validation/', 'validation_set_dropped_nans.csv',
                                        size=256, flip_prob=0., subset_len=2048),
                                        batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Initialize models
    autoencoder = load_autoencoder(bundle_target=BUNDLE,
                                   override_model_cfg_json=args.config,
                                   override_weights_load_path=args.resume_from_ckpt)

    config = utils.model_config(BUNDLE, 'train_diffusion.json')
    scale_factor = compute_scale_factor(autoencoder, train_loader, device)

    scheduler = config.get_parsed_content('noise_scheduler')
    latent_shape = config.get_parsed_content('latent_shape')

    if args.conditioning == 'context':
        use_context = True
        inferer = LatentDiffusionInferer(scheduler=scheduler,scale_factor=scale_factor)
    else:
        use_context = False
        inferer = LatentDiffusionInfererWithClassConditioning(scheduler=scheduler, scale_factor=scale_factor)
    
    unet = load_unet(bundle_target=BUNDLE,
                     context_conditioning=use_context,
                     override_model_cfg_json=args.config,
                     override_weights_load_path=args.resume_from_ckpt)

    optimizer = Adam(list(unet.parameters()), lr=args.lr)
    # optimizer = Adam(list(unet.parameters()) + list(autoencoder.parameters()), lr=args.lr)
    lr_scheduler = MultiStepLR(optimizer, milestones=[1000], gamma=0.1)
    scaler = GradScaler()

    # TODO - the autoencoder does not train with MSE,
    #  also the diffusion might have more complex loss components, we should be careful here
    L = torch.nn.MSELoss().to(device)
    train_noise_shape = [args.batch_size] + latent_shape
    sample_noise_shape = [1] + latent_shape

    losses = {
        'train': [],
        'validation': []
    }

    for e in range(1, args.num_epochs+1):
        print(f'Epoch #[{e}/{args.num_epochs}]:')
        sample(unet, autoencoder, inferer, scheduler, sample_noise_shape,
                '123', len(train_loader.dataset.class_names))

        train_loss = train_loop(unet, autoencoder, inferer, train_loader,
                                L, optimizer, scaler, use_context, train_noise_shape)
        
        losses['train'].append((e, train_loss))
        lr_scheduler.step()

        if e % args.val_every_n_epochs == 0:
            val_loss = val_loop(unet, autoencoder, inferer, val_loader,
                                    L, use_context, train_noise_shape)

            losses['validation'].append((e, val_loss))
            im_file = os.path.join(im_log, f'{e}'.zfill(3) + '.png')
            sample(unet, autoencoder, inferer, scheduler, sample_noise_shape,
                   im_file, len(train_loader.dataset.class_names))

        if e % args.save_ckpt_every_n == 0:
            save_epoch(logdir=logdir,
                       epoch=e,
                       autoencoder=autoencoder,
                       unet=unet,
                       losses_dict=losses)
    torch.cuda.empty_cache()
