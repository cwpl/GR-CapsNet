import os, torch
import torchvision
import numpy as np
import scipy.io as sio

from utils import *
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image
import pickle
from typing import Any, Callable, Optional, Tuple



class affNIST(Dataset):
    ''' In:
            data_path (string): path to the dataset split folder, i.e. train/valid/test
            transform (callable, optional): transform to be applied on a sample.
        Out:
            image, label: sample data and respective label'''

    def __init__(self, data_path, is_Train=True, shuffle=True, transform=None):

        self.data_path = data_path
        self.shuffle = shuffle
        self.transform = transform
        #self.split = self.data_path.split('/')[-1]
        self.split=is_Train
        if self.split:
            data_path = os.path.join(data_path,'train')
            for i, file in enumerate(os.listdir(data_path)):
                # load dataset .mat file batch
                self.dataset = sio.loadmat(os.path.join(data_path, file))
                # concatenate the 32 .mat files to make full dataset
                if i == 0:
                    self.data = np.array(self.dataset['affNISTdata']['image'][0][0])
                    self.labels = np.array(self.dataset['affNISTdata']['label_int'][0][0])
                else:
                    self.data = np.concatenate((self.data,
                        np.array(self.dataset['affNISTdata']['image'][0][0])), axis=1)
                    self.labels = np.concatenate((self.labels,
                        np.array(self.dataset['affNISTdata']['label_int'][0][0])), axis=1)

            # (N, 1, 40, 40) <- (1, 40, 40, N) <- (40*40, N)
            self.data = np.moveaxis(self.data.reshape(1,40,40,-1), -1, 0)
            self.data = self.data.transpose((0,2,3,1))
            # (N,)
            self.labels = self.labels.squeeze()
            #a=2
        else:
            # load valid/test dataset .mat file
            data_path = os.path.join(data_path, 'test')
            self.dataset = sio.loadmat(os.path.join(data_path, 'test.mat'))
            # (40*40, N)
            self.data = np.array(self.dataset['affNISTdata']['image'][0][0])
            # (N, 1, 40, 40) <- (1, 40, 40, N) <- (40*40, N)
            self.data = np.moveaxis(self.data.reshape(1,40,40,-1), -1, 0)
            self.data = self.data.transpose((0, 2, 3, 1))
            # (N,)
            self.labels = np.array(self.dataset['affNISTdata']['label_int'][0][0]).squeeze()

        self.data = self.data.squeeze()

        if self.shuffle: # shuffle the dataset
            idx = np.random.permutation(self.data.shape[0])
            self.data = self.data[idx]
            self.labels = self.labels[idx]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        img = self.data[idx]

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, self.labels[idx] # (X, Y)