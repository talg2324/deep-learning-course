import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class CTDataset(Dataset):
    def __init__(self, data_dir, labels_path) -> None:
        super().__init__()

        labels = pd.read_csv(os.path.join(data_dir, labels_path))
        labels = labels.drop(['sum', 'Unnamed: 0'], axis=1)
        self.ids = labels['ID']
        self.data_dir = data_dir

        
        labels = labels.iloc[:, 1:]
        labels['any'] = 1 - labels['any']
        self.labels = labels.apply(lambda x: np.flatnonzero(x)[0], axis=1).to_numpy()

    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, n):
        id, label = self.ids.iloc[n], self.labels[n]
        abs_path = os.path.join(self.data_dir, f'{id}.npy')
        im = np.load(abs_path)
        return im, label

if __name__ == "__main__":
    ctd = CTDataset('./data/ct-rsna/train', 'train_set.csv')
    ctd[0]