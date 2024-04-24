import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from torchvision import transforms


class CTDataset(Dataset):
    def __init__(self, data_dir, labels_file, size, flip_prob) -> None:
        super().__init__()

        labels = pd.read_csv(os.path.join(data_dir, labels_file))
        labels = labels.drop(['sum', 'Unnamed: 0'], axis=1)
        self.ids = labels['ID']
        self.data_dir = data_dir

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(flip_prob),
        ])

        self.class_names = labels.columns.to_list()[1:]
        self.class_names[0] = 'none'

        labels = labels.iloc[:, 1:]
        labels['any'] = 1 - labels['any']
        self.labels = labels.apply(lambda x: np.flatnonzero(x)[0], axis=1).to_numpy()

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, n):
        id, label = self.ids.iloc[n], self.labels[n]
        abs_path = os.path.join(self.data_dir, f'{id}.npy')
        im = np.load(abs_path).astype(np.float32)

        if self.transform:
            im = self.transform(im)

        # TODO - decide if we want the data to have mean=0, std=1, or to be in the range [-1, 1].. it is much different
        # rescale image dynamic range to [-1, 1]
        im = rescale_im_dynamic_range(im)

        # Grayscale to RGB
        im = torch.stack((im, im, im), axis=-1).squeeze()

        human_label = self.class_names[int(label)]

        return {'image': im, 'class_label': label, 'human_label': human_label}
    
class CTOverfit(CTDataset):
    """
    Class for testing model overfit capability 
    """
    def __init__(self, data_dir, labels_file, size, flip_prob) -> None:
        super().__init__(data_dir, labels_file, size, flip_prob)

    def __len__(self):
        return 16


def rescale_im_dynamic_range(im):
    im_min = im.min()
    im_max = im.max()
    im = 2 * ((im - im_min) / (im_max - im_min)) - 1
    return im
