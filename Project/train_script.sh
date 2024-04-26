#!/bin/bash

## Resume training - comment out to start from scratch
# Ask the user for an experiment name
echo "Please enter the experiment name:"
read name

echo "Copying base model..."
# Copy the directory to use the provided name
mkdir -p ./latent-diffusion/logs/${name}
cp -r ./data/outputs/base-model/* latent-diffusion/logs/${name}/.

## Training
echo "Training ${name}..."
cd latent-diffusion
LD_LIBRARY_PATH=/opt/conda/lib \
CUDA_VISIBLE_DEVICES=0 \
python main.py \
  --base configs/latent-diffusion/ct-rsna.yaml \
  -t \
  --gpus 0, \
  --max_epochs 300 \
  --num_sanity_val_steps 0 \
  --logdir logs \
  --resume logs/${name}/checkpoints/last.ckpt
