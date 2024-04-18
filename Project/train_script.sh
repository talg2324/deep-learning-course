cd latent-diffusion
CUDA_VISIBLE_DEVICES=0 python main.py --base configs/latent-diffusion/ct-rsna.yaml -t --gpus 0, --resume logs/2024-04-18T18-57-41_ct-rsna/checkpoints/last.ckpt