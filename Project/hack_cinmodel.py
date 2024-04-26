import os
import sys
import torch

"""
Start a train run without using the --resume flag
Run this script with the log file that is generated
"""

path_to_model_weights = "./latent-diffusion/models/ldm/cin256-v2/model.ckpt"

if __name__ == "__main__":
    # The model we want to use
    src = torch.load(path_to_model_weights)
    state_dict = src['state_dict']

    pass_through_embedding = torch.zeros((6, 512), dtype=torch.float32)
    for i in range(6):
        pass_through_embedding[i, i] = i


    state_dict['cond_stage_model.embedding.weight'] = pass_through_embedding
    del state_dict['model_ema.decay'], state_dict['model_ema.num_updates']

    
    log_dir = os.path.join('./latent-diffusion/logs', sys.argv[1])
    log_file = os.path.join(log_dir, 'checkpoints/last.ckpt')
    dst = torch.load(log_file)
    if dst is None:
        print("nonetype dst ckpt")
    dst['state_dict'] = state_dict
    torch.save(dst, log_file)
