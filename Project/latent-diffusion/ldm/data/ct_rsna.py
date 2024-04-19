import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
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

        # rescale image dynamic range to [-1, 1]
        im = self.rescale_im_dynamic_range(im)

        # Grayscale to RGB and remap to [-1, 1]
        im = np.stack((im, im, im), axis=0)

        return {'image': im, 'class_label': label}

    @staticmethod
    def rescale_im_dynamic_range(im):
        im_min = np.min(im)
        im_max = np.max(im)
        im = (im - im_min) / (im_max - im_min)
        im = (2 * im) - 1
        return im
