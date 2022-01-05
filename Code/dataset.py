import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader

class CustomDataset(Dataset):

    def __init__(self, annotations_file, sample_dir, transform=None, target_transform=None):
        self.sample_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
        print( self.sample_labels)
        self.sample_dir = sample_dir
        #self.transform = transform
        #$self.target_transform = target_transform
    
    def __len__(self):
        return len(self.sample_labels)

    def __getitem__(self, idx):
        sample_path = os.path.join(self.sample_dir, self.sample_labels.iloc[idx, 0])
        #print(sample_path)
        sample = pd.read_csv(sample_path, header=None)
        sample = torch.tensor(sample.values)
        #print('\n')
        #print(sample)
        label = self.sample_labels.iloc[idx, 1]
        # if self.transform:
        #     sample = self.transform(sample)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return sample, label