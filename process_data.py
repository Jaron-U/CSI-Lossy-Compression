# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 08:24:41 2023

@author: Umt
"""
import pandas as pd
import os
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader
from math import sqrt
from PIL import Image
import torchvision.transforms as T

class DeepVerseChallengeLoaderOne(Dataset):
    def __init__(self, csv_path,
                 noise_var=10**(-74-30/10.0) # Noise power -74dBm
                 ): 
        self.table = pd.read_csv(csv_path)
        self.dataset_folder = os.path.dirname(csv_path)
        self.noise_std = sqrt(noise_var)
        
    def __len__(self):
        return len(self.table)
    
    def __getitem__(self, idx):
        data_path = os.path.join(self.dataset_folder, self.table.loc[idx, 'channel'])
        H = torch.from_numpy(scipy.io.loadmat(data_path)['channel']) # (2, 64, 64)
        if self.noise_std > 0:
            H += (self.noise_std/sqrt(2))*(torch.randn(H.shape) + 1j*torch.randn(H.shape))
        
        # split the complex channel matrix into real and imaginary parts
        H = torch.cat((H.real, H.imag), dim=0)
        return H

def cal_mean_std(loader, num_samples):
    sum = torch.zeros((2, 64, 64))
    sum_of_squares = torch.zeros((2, 64, 64))

    for H in loader:
        sum += torch.sum(H, dim=0)
        sum_of_squares += torch.sum(H**2, dim=0)

    mean = sum / num_samples
    varicance = (sum_of_squares / num_samples) - (mean ** 2)
    std = torch.sqrt(varicance)
    
    return mean, std

def load_data(batch_size=4, num_workers=4, shuffle=True):
    train_dataset = DeepVerseChallengeLoaderOne(csv_path = "dataset_train.csv")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_dataset = DeepVerseChallengeLoaderOne(csv_path = "dataset_test.csv")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return train_loader, test_loader

if __name__ == '__main__':
    train_dataset = DeepVerseChallengeLoaderOne(csv_path = "dataset_train.csv")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    for i, H in enumerate(train_loader):
        print(H.shape) # torch.Size([4, 2, 64, 64])
        break
    

class DeepVerseChallengeLoaderTwo(Dataset):
    def __init__(self, csv_path, 
                 ch_noise_var=10**(-74-30/10.0), # Noise power -74dBm
                 pos_noise_var=4,
                 single_image=True,
                 out_image_size=(256, 256)
                 ): 
        self.table = pd.read_csv(csv_path)
        self.dataset_folder = os.path.dirname(csv_path)
        self.ch_noise_std = sqrt(ch_noise_var)
        self.pos_noise_std = sqrt(pos_noise_var)
        self.position = self.table[['x', 'y', 'z']].to_numpy()
        self.images = self.table[['cam_left', 'cam_mid', 'cam_right']].to_numpy()
        self.image_sel = self.table['cam_select'].to_numpy()+1
        self.single_image = single_image
        self.out_image_size = out_image_size
        
    def __len__(self):
        return len(self.table)
    
    def __getitem__(self, idx):
        # Position
        P = torch.from_numpy(self.position[idx])
        
        # Channel
        ch_data_path = os.path.join(self.dataset_folder, self.table.loc[idx, 'channel'])
        H = torch.from_numpy(scipy.io.loadmat(ch_data_path)['channel'])
        if self.ch_noise_std > 0:
            H += (self.ch_noise_std/sqrt(2))*(torch.randn(H.shape) + 1j*torch.randn(H.shape))
            
        # Position
        if self.pos_noise_std > 0:
            P += self.pos_noise_std * torch.randn(P.shape)
            
        # Return a single image or all three images from left/center/right cameras
        if self.single_image:
            I = [T.Resize(size=self.out_image_size)(Image.open(self.images[idx, self.image_sel[idx]]))]
        else:
            I = [T.Resize(size=self.out_image_size)(Image.open(self.images[idx, i])) for i in range(3)]
        
        # Radar
        radar_data_path = os.path.join(self.dataset_folder, self.table.loc[idx, 'radar'])
        R = torch.from_numpy(scipy.io.loadmat(radar_data_path)['ra'])
        
        return (H, P, R, *I), H

# if __name__ == '__main__':

#     train_dataset = DeepVerseChallengeLoaderTwo(csv_path = "dataset_train.csv")
#     train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    
#     test_dataset = DeepVerseChallengeLoaderTwo(csv_path = r'C:\Users\Umt\Dropbox (ASU)\challenge\challenge_data\dataset_test.csv')
#     test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)


class RandomDataset(Dataset):
    def __init__(self, num_samples=30, shape=(1, 64, 64)):
        self.num_samples = num_samples
        self.shape = shape

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random tensor of the specified shape
        random_tensor = torch.randn(self.shape, dtype=torch.float32)
        return random_tensor

def load_random_data(batch_size=4, num_workers=4):
    dataset = RandomDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val = RandomDataset()
    val_dataloader = DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader, val_dataloader