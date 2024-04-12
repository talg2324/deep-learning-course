import torch
from omegaconf import OmegaConf

import sys
sys.path.append("./latent-diffusion/")
from ldm.util import instantiate_from_config


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("./latent-diffusion/models/ldm/cin256/config.yaml")  
    model = load_model_from_config(config, "./latent-diffusion/models/ldm/cin256/model.ckpt")
    return model

if __name__ == "__main__":

    model = get_model()
    print(1)