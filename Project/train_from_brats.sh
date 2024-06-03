#!/bin/bash

## Resume training - comment out to start from scratch
# Ask the user for an experiment name
echo "Please enter the experiment name:"
read name

## Training
echo "Training ${name}..."
cd brats-mri
LD_LIBRARY_PATH=/opt/conda/lib \
python main.py \
  --num_epochs 500 \
  --val_every_n_epochs 100 \
  --name ${name} \
  --conditioning class \
  --batch_size 32 \
  --lr 0.0001 \
  --config inference_new.json \
  --subset_len 4096 \
  --save_ckpt_every_n 100
