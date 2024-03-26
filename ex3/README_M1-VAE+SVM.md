# Exercise 3

# data preprocessing
First - make sure you choose your dataset (MNIST / FashionMNIST). this choice also applies for the evaluation sections

run the "data preprocessing" cells to load the dataset and data loaders.

# VAE Model
contains the VAE class and 2 subclasses - VaeEncoder, VaeDecoder, for convenience and clarity.
this section implements the VAE model according to details from the article: "Semi-supervised Learning with
Deep Generative Models", by Kingsma et al.

the VAE class simply uses the encoder and decoder components to form the whole architecture.
VaeEncoder: receives a batch of data samples, and outputs 2 vectors representing the latent representation mean and log-variance
VaeDecoder: receives a batch of mean & logvar vectors, generates random vectors according to the mean and variance (reparameterization trick) and outputs a reconstructed sample at the encoder input dimension

vae_loss() - calculates the kl divergence term and the log likelihood term, for a given batch of inputs, outputs, means, and logvars.

# svm utils
utility functions (balanced sampling, save & load)

#hyperparams
run them to set the number of train epochs
and set the input dim and device.

all 4 first sections are mandatory for any run (data preprocessing, VAE Model, svm utils, hyperparams)

# VAE training
training loop. run to train a model.
use the saving cell to save your model

#SVM fitting
cell 1#
this section starts with loading a pre-trained VAE so that no need to waste time on the longer VAE training for the short SVM fit.

cell 2#
run to fit SVMs (for different number of labels in train set)

## Evaluation
this section requires running the 4 first sections (data preprocessing, VAE Model, svm utils, hyperparams)

it will output the error rate on the test-set for both vanilla-svm (trained directy on input images) and M1-svm (trained on VAE encoder output).
as well, a visualization is added to demonstrate the reconstructed images of the VAE.