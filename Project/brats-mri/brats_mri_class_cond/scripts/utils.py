# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import numpy as np
import torch
import tqdm
from monai.utils import first
from monai.utils.type_conversion import convert_to_numpy


def compute_scale_factor(autoencoder, train_loader, device, fast=False):
    latent_vectors = []
    with torch.no_grad():
        if fast:
            data = first(train_loader)
            z = autoencoder.encode_stage_2_inputs(data["image"].to(device))
            latent_vectors.append(z.cpu())

        else:
            with tqdm.tqdm(train_loader, desc='Computing scale factor') as pbar:
                for data in pbar:
                    z = autoencoder.encode_stage_2_inputs(data["image"].to(device))
                    latent_vectors.append(z.cpu())
        scale_factor = 1. / torch.cat(latent_vectors).std().item()
    print(f'Scale factor: {scale_factor:.5f}')
    return scale_factor
    


def normalize_image_to_uint8(image):
    """
    Normalize image to uint8
    Args:
        image: numpy array
    """
    draw_img = image
    if np.amin(draw_img) < 0:
        draw_img[draw_img < 0] = 0
    if np.amax(draw_img) > 0.1:
        draw_img /= np.amax(draw_img)
    draw_img = (255 * draw_img).astype(np.uint8)
    return draw_img


def visualize_2d_image(image):
    """
    Prepare a 2D image for visualization.
    Args:
        image: image numpy array, sized (H, W)
    """
    image = convert_to_numpy(image)
    # draw image
    draw_img = normalize_image_to_uint8(image)
    draw_img = np.stack([draw_img, draw_img, draw_img], axis=-1)
    return draw_img
