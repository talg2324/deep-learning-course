import torch
from omegaconf import OmegaConf
from matplotlib import pyplot as plt

import sys
sys.path.append("./latent-diffusion/")
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("./latent-diffusion/models/ldm/cin256/config.yaml")
    model = load_model_from_config(config, "./latent-diffusion/models/ldm/cin256/model.ckpt")
    return model.to(device)

if __name__ == "__main__":

    model = get_model()
    sampler = DDIMSampler(model)

    class_label = 25
    n_samples = 1

    ddim_steps = 20
    ddim_eta = 0.0
    scale = 3.0   # for unconditional guidance

    with torch.no_grad():
        with model.ema_scope():
            x0 = torch.tensor(n_samples*[999], device=device)
            xc = torch.tensor(n_samples*[class_label], device=device)

            uc = model.get_learned_conditioning({model.cond_stage_key: x0})
            c = model.get_learned_conditioning({model.cond_stage_key: xc})

            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=c,
                                             batch_size=n_samples,
                                             shape=[4, 64, 64],
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc, 
                                             eta=ddim_eta)
            
            
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                         min=0.0, max=1.0)
            
            plt.imshow(x_samples_ddim.squeeze().permute(1,2,0).cpu())
            plt.show()