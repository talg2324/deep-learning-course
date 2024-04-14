
# ImageNet model
- run training as is
- create a pass-through class embedder - see 
[https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/encoders/modules.py#L21]
- create a new dataset class for one of our datasets. see:
[https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/data/imagenet.py]
- create yaml config that matches our dataset dimensions. see:
[https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/configs/latent-diffusion/cin256-v2.yaml]
- Run a short training (sanity for our dataset and config)
- load pre-trained weights from imagenet to our model.
- Run pre-trained LDM

# MONAI model
- Run pre-trained LDM

