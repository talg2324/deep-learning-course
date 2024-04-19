import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES = ['acl', 'meniscus']
SLICES = ['axial', 'coronal', 'sagittal']


class MRNet(Dataset):
    def __init__(self, rt_dir, train_or_val, size) -> None:
        super().__init__()
        assert train_or_val in ['train', 'valid'], "select either 'train' or 'valid'"

        self.data_dir = os.path.join(rt_dir, train_or_val)

        all_labels = pd.DataFrame(columns=['ID', 'None']+CLASSES)
        for c in CLASSES:
            label = pd.read_csv(os.path.join(rt_dir, f'{train_or_val}-{c}.csv'))
            label.columns = ['ID', c]
            all_labels['ID'] = label['ID']
            all_labels[c] = label[c]

        all_labels['sum'] = all_labels.iloc[:, 2:].sum(axis=1)
        all_labels['None'] = (all_labels['sum']==0).astype(int)

        all_labels = all_labels[all_labels['sum'] <= 1].drop('sum', axis=1)

        self.ids = all_labels['ID'].apply(lambda x: str(x).zfill(4))
        self.labels = all_labels.drop('ID', axis=1).apply(lambda x: np.flatnonzero(x)[0], axis=1).to_numpy()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size),
        ])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, n):
        id, label = self.ids.iloc[n], self.labels[n]

        im_slices = []
        for slice in SLICES:
            abs_path = os.path.join(self.data_dir, f'{slice}/{id}.npy')
            im_slices.append(np.load(abs_path).astype(np.float32))

        im = np.stack(im_slices, axis=0)

        if self.transform:
            im = self.transform(im)

        # TODO - decide if we want the data to have mean=0, std=1, or to be in the range [-1, 1].. it is much different
        # rescale image dynamic range to [-1, 1]
        im = self.rescale_im_dynamic_range(im)

        return {'image': im, 'class_label': label}

    @staticmethod
    def rescale_im_dynamic_range(im):
        im_min = im.min()
        im_max = im.max()
        im = 2 * ((im - im_min) / (im_max - im_min)) - 1

        return im
