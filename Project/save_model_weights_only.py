import os
import sys
import torch
import glob

"""
Start a train run without using the --resume flag
Run this script with the log file that is generated
"""

model_template = "./data/outputs/overfit_30epochs/checkpoints/init_hacked.ckpt"


if __name__ == "__main__":
    path_to_model = sys.argv[1]
    # The model we want to use
    template = torch.load(model_template)
    src = torch.load(path_to_model)
    template['state_dict'] = src['state_dict']
    training_dir = "/".join(path_to_model.split("/")[:-1])
    num_weight_only_files = len(glob.glob(os.path.join(training_dir, '*_weights_only*')))

    path_to_model_weights_only = path_to_model.split(".")[0] + f"_weights_only_{num_weight_only_files}.ckpt"
    torch.save(template, path_to_model_weights_only)

    print('Done copying weights')