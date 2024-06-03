#!/bin/bash

## Resume training - comment out to start from scratch
# Ask the user for an experiment name
echo "Please enter the experiment name:"
read name

## Training
echo "Training ${name}..."
cd latent-diffusion
LD_LIBRARY_PATH=/opt/conda/lib \
CUDA_VISIBLE_DEVICES=0 \
python main_shortened.py \
  --base configs/latent-diffusion/ct-rsna.yaml \
  -t \
  --gpus 0, \
  --max_epochs 80 \
  --num_sanity_val_steps 0 \
  --scale_lr False \
  --logdir ../data/outputs \
  --name ${name} \
  --resume_from_checkpoint ../data/outputs/overfit_30epochs/checkpoints/init_hacked_last.ckpt
