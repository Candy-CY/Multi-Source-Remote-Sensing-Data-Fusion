import os
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from scipy import io

class Multimodal_Dataset_Train(Dataset):
    def __init__(self, Filename='Trento', MM_Data='LIDAR'):
        
        HSI = io.loadmat('./'+str(Filename)+'11x11/HSI_Tr.mat')
        LIDAR = io.loadmat('./'+str(Filename)+'11x11/'+str(MM_Data)+'_Tr.mat')
        label = io.loadmat('./'+str(Filename)+'11x11/TrLabel.mat')
        
        #self.hs_ims = torch.from_numpy(HSI['Data'].astype(np.float32)).permute(0,3,1,2)
        self.hs_ims = (torch.from_numpy(HSI['Data'].astype(np.float32)).to(torch.float32)).permute(0,3,1,2) 
        self.lid_ims = (torch.from_numpy(LIDAR['Data'].astype(np.float32)).to(torch.float32)).permute(0,3,1,2)
        self.lbs = ((torch.from_numpy(label['Data'])-1).long()).reshape(-1)

    def __len__(self):
        return self.hs_ims.shape[0]
        
    def __getitem__(self, i):
        return self.hs_ims[i], self.lid_ims[i], self.lbs[i]

class Multimodal_Dataset_Test(Dataset):
    def __init__(self, Filename='Trento', MM_Data='LIDAR'):
        
        HSI = io.loadmat('./'+str(Filename)+'11x11/HSI_Te.mat')
        LIDAR = io.loadmat('./'+str(Filename)+'11x11/'+str(MM_Data)+'_Te.mat')
        label = io.loadmat('./'+str(Filename)+'11x11/TeLabel.mat')
        
        self.hs_ims = (torch.from_numpy(HSI['Data'].astype(np.float32)).to(torch.float32)).permute(0,3,1,2) 
        self.lid_ims = (torch.from_numpy(LIDAR['Data'].astype(np.float32)).to(torch.float32)).permute(0,3,1,2)
        self.lbs = ((torch.from_numpy(label['Data'])-1).long()).reshape(-1)

    def __len__(self):
        return self.hs_ims.shape[0]
        
    def __getitem__(self, i):
        return self.hs_ims[i], self.lid_ims[i], self.lbs[i]

