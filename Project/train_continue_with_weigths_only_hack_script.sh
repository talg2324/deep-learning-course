#!/bin/bash

## Resume training - comment out to start from scratch
# Ask the user for an experiment name
echo "Please enter the relative path to a .ckpt file to continue from"
read ckpt_file_path

## Training
echo "copying weights from ${ckpt_file_path}..."
n_weight_only_files=`echo ${ckpt_file_path%/*}`
python save_model_weights_only.py ${ckpt_file_path}
hacked_weights_ckpt_path=`echo "${ckpt_file_path//.ckpt/_weights_only_${n_weight_only_files}.ckpt}"`
echo "continuing training with ${hacked_weights_ckpt_path}"

cd latent-diffusion
LD_LIBRARY_PATH=/opt/conda/lib \
CUDA_VISIBLE_DEVICES=0 \
python main.py \
  --base configs/latent-diffusion/ct-rsna.yaml \
  -t \
  --gpus 0, \
  --max_epochs 20 \
  --num_sanity_val_steps 0 \
  --scale_lr False \
  --resume ../${hacked_weights_ckpt_path}
