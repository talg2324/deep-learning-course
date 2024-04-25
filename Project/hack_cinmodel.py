import os
import sys
import torch

"""
Start a train run without using the --resume flag
Run this script with the log file that is generated
"""

path_to_model_weights = "./latent-diffusion/models/ldm/cin256-v2/model.ckpt"

if __name__ == "__main__":

    log_input = sys.argv[1]
    n_classes = int(sys.argv[2])

    pass_through_embedding = torch.zeros((n_classes, 512), dtype=torch.float32)
    for i in range(n_classes):
        pass_through_embedding[i, i] = i

    # The model we want to use
    src = torch.load(path_to_model_weights)
    state_dict = src['state_dict']
    state_dict['cond_stage_model.embedding.weight'] = pass_through_embedding
    del state_dict['model_ema.decay'], state_dict['model_ema.num_updates']

    # Output
    log_file = os.path.join('./data/outputs', log_input, 'checkpoints/last.ckpt')
    print(f'Copying weights from {path_to_model_weights} to {log_file}...')

    dst = torch.load(log_file)
    dst['state_dict'] = state_dict
    torch.save(dst, log_file)

    print('Done copying weights')
