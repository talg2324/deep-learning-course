#!/bin/bash

## Resume training - comment out to start from scratch
# Ask the user for an experiment name
echo "Please enter the experiment name:"
read name

## Training
echo "Training ${name}..."
cd brats-mri
LD_LIBRARY_PATH=/opt/conda/lib \
python train_autoencoder.py \
  --num_epochs 30 \
  --val_every_n_epochs 3 \
  --name ${name} \
  --batch_size 4\
  --lr 0.0001