import os, sys
from os.path import join, dirname
from glob import glob
import numpy as np

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

###################################  DATA LOADERS   ##########################################

class arrayDataset(Dataset):
    """Class for manipulating the IMAGEN Dataset. Inherits from the torch Dataset class.
    Parameters
    ----------
    X: Input data, i.e MRI images.
    y: Labels for the data.
    transfrom: Function for transforming the data into the appropriate format.
    mask: A mask that can be applied to the data with load_nifti.
    z_factor: A zoom factor that can be applied to the data with load_nifti.
    dtype: The desired data type of the data.      
    """
    def __init__(self, X, y, cf=None, cf_name=None, transform=None, 
                 lbl_type="classif", 
                 soft_labels=False):
        assert len(X) == len(y) 
        assert lbl_type in ['classif_binary', 'classif', 'regression'], f"unsupported lbl_type = {lbl_type}"
        # TODO test with multitask and 'other' modes
        self.X = X
        self.y = y
        self.cf = cf
        self.transform = transform
        self.lbl_type = lbl_type
        self.cf_name = cf_name
        
        self.soft_labels = soft_labels
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image = self.X[idx]
        if self.transform: image = self.transform(image)
        
        label = self.y[idx]
        if self.lbl_type=='classif_binary':
            label = torch.FloatTensor([label])[0]
            # use soft-probability label if requested
            if self.soft_labels:  ##TODO test
                if label[0]==1.0: label = label-0.01
                if label[0]==0.0: label = label+0.01
                
        elif self.lbl_type=='classif':
            label = torch.LongTensor([label])[0]
        else: # self.lbl_type=='regression'
            label = torch.FloatTensor([label])[0]
            
            
        if self.cf is not None:
            confounder = self.cf[idx]
            
        if self.cf is not None:
            sample = {"image" : image, "label" : label , self.cf_name : confounder} 
        else:
            sample = {"image" : image, "label" : label}
        return sample
    
    
class arrayDatasetWithSubID(arrayDataset):
    """Same as the 'arrayDataset' class but also returns the subject's idx / ID
       from the __getitem__"""
    def __init__(self, X, y, sub_ids, cf=None, cf_name=None, **kwargs):
        super().__init__(X, y, cf, cf_name, **kwargs) 
        self.i = sub_ids
        assert len(y) == len(sub_ids)
    
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sample.update({'i': int(self.i[idx])})
        return sample