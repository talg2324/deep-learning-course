import os
import sys
import torch
import monai

from tqdm import tqdm
import argparse

from monai.utils import first
from PIL import Image
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from generative.losses import PerceptualLoss


from pretrained import load_autoencoder
import utils


BUNDLE = 'brats_mri_class_cond'
sys.path.append(BUNDLE)
from scripts.ct_rsna import CTSubset


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def save_epoch(logdir: str, epoch: int, autoencoder, losses_dict: dict):
    autoencoder_ckpt_path = os.path.join(logdir, f"autoencoder_epoch_{epoch}.ckpt")
    losses_path = os.path.join(logdir, f"losses_dict_epoch_{epoch}")
    print("Saving checkpoint at epoch ", epoch)

    # Save the checkpoint
    torch.save(autoencoder.state_dict(), autoencoder_ckpt_path)
    torch.save(losses_dict, losses_path)

@torch.no_grad()
def log_ims(autoencoder, im_tag, data_sample, max_ims=4):
    autoencoder.eval()

    max_ims = min(max_ims, len(data_sample['image']))

    input_ims = data_sample['image'].to(device)[:max_ims]
    encode_decode, _, _ = autoencoder(input_ims)

    input_col = torch.vstack([im.squeeze() for im in input_ims])
    output_col = torch.vstack([im.squeeze() for im in encode_decode])
    log_im = utils.rescale_outputs(torch.hstack((input_col, output_col)))

    log_im = Image.fromarray(log_im)
    log_im.save(im_tag + '_encoder.png')


def perceptual_loss(perceptual_loss_model_weights_path = None):
    loss_p = PerceptualLoss(spatial_dims=2,
                        network_type="resnet50",
                        pretrained=True,
                        pretrained_path=perceptual_loss_model_weights_path,
                        pretrained_state_dict_key="state_dict"
                       )
    return loss_p


def compute_kl_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=list(range(1, len(z_sigma.shape)))
    )
    return torch.sum(kl_loss) / kl_loss.shape[0]


# TODO - move this global somewhere?
kl_weight = 1e-6


def naive_train_loop(autoencoder, dataloader, L, optimizer, use_perceptual_loss: bool = False):
    autoencoder.train()
    if use_perceptual_loss:
        loss_perceptual = perceptual_loss().to(device)
    total_loss = 0
    with tqdm(dataloader, desc='  Training loop', total=len(dataloader)) as pbar:
        for batch in pbar:
            ims = batch['image'].to(device)
            ims_recon, z_mu, z_sigma = autoencoder(ims)
            l1_loss = L(ims.float(), ims_recon.float())
            kl_loss = compute_kl_loss(z_mu, z_sigma)
            loss = l1_loss + kl_weight * kl_loss
            if use_perceptual_loss:
                loss += loss_perceptual(ims_recon.float(), ims.float())

            pbar.set_postfix({"loss": loss.item()})

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        pbar.set_postfix({"loss": avg_loss})
    return avg_loss


@torch.no_grad
def naive_val_loop(autoencoder, dataloader, L, use_perceptual_loss: bool = False):
    autoencoder.eval()
    if use_perceptual_loss:
        loss_perceptual = perceptual_loss().to(device)
    total_loss = 0
    with tqdm(dataloader, desc='  Validation loop', total=len(dataloader)) as pbar:
        for batch in pbar:
            ims = batch['image'].to(device)
            ims_recon, z_mu, z_sigma = autoencoder(ims)
            l1_loss = L(ims.float(), ims_recon.float())
            kl_loss = compute_kl_loss(z_mu, z_sigma)
            loss = kl_weight * kl_loss + l1_loss
            if use_perceptual_loss:
                loss += loss_perceptual(ims_recon.float(), ims.float())
            pbar.set_postfix({"loss": loss.item()})

            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        pbar.set_postfix({"loss": avg_loss})
    return avg_loss


def autoencoder_arg_parser():
    parser = argparse.ArgumentParser(description='AutoencoderKL Training Script')

    # Argument for number of epochs
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs for training')

    # Argument for setting epochs interval for validation
    parser.add_argument('--val_every_n_epochs', type=int, default=1,
                        help='epochs interval between validation loops, default is 1')

    # Argument for name of the model
    parser.add_argument('--name', type=str, default='model',
                        help='Name of the model')
    # Argument for seed
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed setting')

    # Argument for periodic checkpoint saving
    parser.add_argument('--save_ckpt_every_n', type=int, default=None,
                        help='periodic ckpt saving')

    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--use_perceptual_loss', type=bool, default=False,
                        help='use perceptual loss term in Autoencoder training')
    return parser


if __name__ == "__main__":
    arg_parser = autoencoder_arg_parser()
    args = arg_parser.parse_args()
    print("Started training AutoencoderKL:", args.name)
    print("Random seed: ", args.seed)
    print("Number of epochs: ", args.num_epochs)
    print(f"Performing validation every {args.val_every_n_epochs} epochs")
    if args.save_ckpt_every_n:
        print(f"saving checkpoints each {args.save_ckpt_every_n} epochs")
    else:
        args.save_ckpt_every_n = args.num_epochs
        print(f"saving only on last epoch")

    monai.utils.set_determinism(seed=args.seed)
    utils.download_weights_if_not_already(BUNDLE)

    logdir = os.path.join('../data/', 'outputs', args.name)
    im_log = os.path.join(logdir, 'image_logs')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
        os.mkdir(im_log)

    print(f"Training dir: {logdir}")
    # save training args
    torch.save(args, os.path.join(logdir, 'training_args'))

    train_loader = DataLoader(CTSubset('../data/ct-rsna/train/', 'train_set_dropped_nans.csv',
                                       size=256, flip_prob=0.5, subset_len=128),
                              batch_size=args.batch_size, shuffle=True, drop_last=True)

    val_loader = DataLoader(CTSubset('../data/ct-rsna/validation/', 'validation_set_dropped_nans.csv',
                                     size=256, flip_prob=0., subset_len=32),
                            batch_size=args.batch_size, shuffle=True, drop_last=True)
    # load autoencoder
    autoencoder = load_autoencoder(bundle_target=BUNDLE)

    num_epochs = 20
    val_every_n_epochs = 2
    lr = 1e-5

    L = torch.nn.L1Loss()
    optimizer = Adam(list(autoencoder.parameters()), lr=args.lr)
    lr_scheduler = CosineAnnealingLR(optimizer, args.num_epochs, eta_min=1e-6)

    losses = {
        'train': [],
        'validation': [],
    }

    # log pre-train
    log_ims(autoencoder, 'pretraining', first(val_loader), max_ims=4)
    for e in range(1, args.num_epochs + 1):
        print(f'Epoch #[{e}/{args.num_epochs}]:')
        train_loss = naive_train_loop(autoencoder, train_loader, L, optimizer, args.use_perceptual_loss)
        losses['train'].append((e, train_loss))
        lr_scheduler.step()
        if e % val_every_n_epochs == 0:
            val_loss = naive_val_loop(autoencoder, val_loader, L)
            losses['validation'].append((e, val_loss))

        if e % args.save_ckpt_every_n == 0:
            save_epoch(logdir=logdir,
                       epoch=e,
                       autoencoder=autoencoder,
                       losses_dict=losses)
            log_ims(autoencoder, os.path.join(im_log, f'epoch_{e}'), first(val_loader), max_ims=4)
        torch.cuda.empty_cache()
