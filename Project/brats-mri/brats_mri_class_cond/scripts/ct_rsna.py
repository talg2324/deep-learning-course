import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms


class CTDataset(Dataset):
    def __init__(self, data_dir, labels_file, size, flip_prob) -> None:
        super().__init__()

        labels = pd.read_csv(os.path.join(data_dir, labels_file))
        labels = labels.sample(frac=1)

        unnamed = [c for c in labels.columns if 'Unnamed' in c]
        labels = labels.drop(['sum'] + unnamed, axis=1)
        self.ids = labels['ID']
        self.data_dir = data_dir

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(flip_prob),
        ])

        self.class_names = labels.columns.to_list()[1:]
        self.class_names[0] = 'none'

        labels = labels.iloc[:, 1:]
        labels['any'] = 1 - labels['any']
        self.labels = labels.apply(lambda x: np.flatnonzero(x)[-1], axis=1).to_numpy()

    def class_distribution(self, file_name, labels):
        nbins = len(self.class_names)
        counts, _ = np.histogram(labels, bins=nbins)
        prob = 100 * counts / counts.sum()

        print(f'Data source: {file_name}')
        for c in range(nbins):
            print(f'    Class {self.class_names[c]}: {prob[c]:.1f}%')

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, n):
        id, label = self.ids.iloc[n], self.labels[n]
        abs_path = os.path.join(self.data_dir, f'{id}.npy')
        im = np.load(abs_path).astype(np.float32).T

        if self.transform:
            im = self.transform(im)

        # TODO - decide if we want the data to have mean=0, std=1, or to be in the range [-1, 1].. it is much different
        # rescale image dynamic range to [0, 1]
        im = rescale_im_dynamic_range(im)

        human_label = self.class_names[int(label)]

        return {'image': im, 'class_label': label, 'human_label': human_label, 'id': id}


class CTSubset(CTDataset):
    """
    Class for testing model on part of the data 
    """
    def __init__(self, data_dir, labels_file, size, flip_prob, subset_len) -> None:
        super().__init__(data_dir, labels_file, size, flip_prob)
        self.subset_len = subset_len
        self.class_distribution(labels_file, self.labels[:subset_len])
        self.labels = self.labels[:subset_len]

    def __len__(self):
        return self.subset_len


class MultiSliceCTDataset(CTDataset):
    """
    Class for testing model on part of the data
    """

    def __init__(self, data_dir, train_dir, val_dir, labels_file, size, flip_prob, n_slices_in_study) -> None:
        Dataset.__init__(self)

        labels = pd.read_csv(os.path.join(data_dir, labels_file))
        unnamed = [c for c in labels.columns if 'Unnamed' in c]
        labels = labels.drop(['sum'] + unnamed, axis=1)
        self.class_names = labels.columns.to_list()[1:-1]
        
        labels = self.filter_by_slices_per_study(labels, n_slices_in_study)
        self.ids = labels['ID']
        self.study_ids = labels['StudyInstanceUID']
        self.data_dir = data_dir
        self.train_dir = train_dir
        self.val_dir = val_dir

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(flip_prob),
        ])

        self.class_names[0] = 'none'

        labels = labels.iloc[:, 1:-1]
        labels['any'] = 1 - labels['any']
        self.labels = labels.apply(lambda x: np.flatnonzero(x)[-1], axis=1).to_numpy()

    @staticmethod
    def filter_by_slices_per_study(df, n_slices_in_study):
        grouped = df.groupby('StudyInstanceUID')

        df_combined = pd.DataFrame()
        for study_id, gdf in grouped:
            if len(gdf) == n_slices_in_study:
                df_combined = pd.concat([df_combined, gdf], ignore_index=True)
        return df_combined

    def __getitem__(self, n):
        id = self.ids.iloc[n]
        train_path = os.path.join(self.train_dir, f'{id}.npy')
        val_path = os.path.join(self.val_dir, f'{id}.npy')
        tmp_data_dir = self.data_dir
        if os.path.exists(train_path):
            self.data_dir = self.train_dir
        elif os.path.exists(val_path):
            self.data_dir = self.val_dir
        else:
            raise Exception(f"file {id}.npy not found")

        item = super(MultiSliceCTDataset, self).__getitem__(n)
        item['study_id'] = self.study_ids.iloc[n]
        self.data_dir = tmp_data_dir
        return item



def rescale_im_dynamic_range(im):
    im_min = im.min()
    im_max = im.max()
    if im_min == im_max:
        return torch.zeros_like(im)
    else:
        im = (im - im_min) / (im_max - im_min)
    return im
