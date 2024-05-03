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
  --num_epochs 20 \
  --val_every_n_epochs 5 \
  --name ${name} \
  --conditioning context