# Exercise 3 - Q4 - GANS

run the first 4 cells to:
- set the imports
- mount the drive
- set device
- set seed
- load fashion mnist dataset

# DCGAN
- hyperparams are set in the first cell in the section
- implementation according to https://arxiv.org/pdf/1511.06434.pdf, with modification to grayscale images
- training loop
- losses plots

# Wasserstein GAN
- hyperparams are set in the first cell in the section
- implementation according to  https://arxiv.org/pdf/1704.00028.pdf - Section F, with modification to grayscale images
  we implemented WGAN-GP. the WGAN descriminator and Generator are formed from ResUp/ResDown blocks.
- gradient penalty loss is implemented 
- training loop
- losses plots

# Inference
- loading pretrained models
- display random sample generation

To run inference alone - make sure you run the first 4 cells, as well as the models implementations. no need to run the training loops or plottings.