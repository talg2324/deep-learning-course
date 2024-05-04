import os
from typing import List

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms


class ImageNetMiniDataset(Dataset):
    def __init__(self, data_dir, labels_file, size) -> None:
        super().__init__()

        labels = pd.read_csv(os.path.join(data_dir, labels_file), index_col=0)
        labels = labels.sample(frac=1)

        unnamed = [c for c in labels.columns if 'Unnamed' in c]
        # labels = labels.drop(['sum'] + unnamed, axis=1)
        self.ids = labels['id']
        self.data_dir = data_dir

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size),
        ])

        # self.class_names = labels.columns.to_list()[1:]
        self.class_names = labels.set_index('class_label')['human_label'].to_dict()
        self.labels = labels['class_label'].to_numpy()

    def class_distribution(self, file_name, labels):
        unique, counts = np.unique(labels, return_counts=True)
        prob = 100 * counts / counts.sum()

        print(f'Data source: {file_name}')
        for i, c in enumerate(unique):
            print(f'    Class {self.class_names[c]}: {prob[i]:.1f}%')

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, n):
        id, label = self.ids.iloc[n], self.labels[n]
        abs_path = os.path.join(self.data_dir, f'{id}.npy')
        im = np.load(abs_path).astype(np.float32)
        
        if self.transform:
            im = self.transform(im)
        im = im.permute(1, 2, 0)
        human_label = self.class_names[int(label)]

        return {'image': im, 'class_label': label, 'human_label': human_label}


class ImageNetMiniSubset(ImageNetMiniDataset):
    """
    Class for testing model on part of the data
    """

    def __init__(self, data_dir, labels_file, size, classes: List[int]) -> None:
        super().__init__(data_dir, labels_file, size)
        self.classes = classes
        self.filter_classes()
        self.class_distribution(labels_file, self.labels)

    def filter_classes(self):
        valid_indices = [i for i, label in enumerate(self.labels) if label in self.classes]
        # self.labels = filter(lambda label: label in self.classes, self.labels)
        self.class_names = {label: self.class_names.get(label) for label in self.classes}
        self.labels = self.labels[valid_indices]
        self.ids = self.ids.iloc[valid_indices]
        # i for i, label in enumerate(self.labels) if label in self.classes]
