import os
import sys
import torch

"""
Start a train run without using the --resume flag
Run this script with the log file that is generated
"""


if __name__ == "__main__":
    path_to_model = sys.argv[1]
    # The model we want to use
    src = torch.load(path_to_model)
    dst = {'state_dict': src['state_dict']}
    path_to_model_weights_only = path_to_model.split(".")[0] + "_weights_only.ckpt"
    torch.save(dst, path_to_model_weights_only)

    print('Done copying weights')
