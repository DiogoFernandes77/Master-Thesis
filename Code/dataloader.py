import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataset import CustomDataset





##--------------------DataLoader----------------------------
current_dir = os.getcwd()
data_dir = current_dir + '/data'

annotations_file = data_dir + '/labels.csv'

training_data = CustomDataset(annotations_file, data_dir)



train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
##-----------------------------------------------------------