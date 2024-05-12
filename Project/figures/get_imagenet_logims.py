import torch
from omegaconf import OmegaConf
from matplotlib import pyplot as plt

import sys
sys.path.append("./latent-diffusion/")
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, weights_only=False, map_location=torch.device('cpu'))
    sd = pl_sd["state_dict"]
    torch.save(sd, './tmp_sd')
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(torch.load('./tmp_sd', map_location=device), strict=False)
    model.eval()
    return model.to(device), sd, pl_sd


def get_model(model_config_path, model_ckpt_path):
    config = OmegaConf.load(model_config_path)
    model, sd, pl_sd = load_model_from_config(config, model_ckpt_path)
    return model


def get_training_ckpt_files(output_dir, training_name):
  ckpt_dir = os.path.join(output_dir, training_name, 'checkpoints')
  training_ckpt_files = [os.path.join(ckpt_dir, ckpt) for ckpt in os.listdir(ckpt_dir) if 'epoch' in ckpt]
  return training_ckpt_files


def strip_epoch_num_from_ckpt(ckpt_full_path):
  ckpt_name = ckpt_full_path.split('/')[-1]
  epoch_num = 1 + int(ckpt_name.split(".")[0].split("=")[-1])
  return epoch_num


def get_training_cfg_file(output_dir, training_name):
  cfg_dir = os.path.join(output_dir, training_name, 'configs')
  model_cfg_files = [cfg for cfg in os.listdir(cfg_dir) if 'project' in cfg]
  if len(model_cfg_files) == 0:
    raise ValueError("configs dir empty, you may manualy pass the config file instead")
  if len(model_cfg_files) > 1:
    raise ValueError("more than 1 config file in configs dir, you may manualy pass the config file instead")
  
  return os.path.join(cfg_dir, model_cfg_files[0])

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
            x0 = torch.tensor(n_samples*[1000], device=device)
            xc = torch.tensor(n_samples*[class_label], device=device)

            uc = model.get_learned_conditioning({model.cond_stage_key: x0})
            c = model.get_learned_conditioning({model.cond_stage_key: xc})

            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=c,
                                             batch_size=n_samples,
                                             shape=[3, 64, 64],
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc, 
                                             eta=ddim_eta)
            
            
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                         min=0.0, max=1.0)
            
            plt.imshow(x_samples_ddim.squeeze().permute(1,2,0).cpu())
            plt.show()