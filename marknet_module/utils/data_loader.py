
# markdataのloaderクラス

import torch
from torch.utils.data  import Dataset
import cv2
import os 
from PIL import Image 
import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np



class DatasetLoader(Dataset):

    def __init__(self, csv_file_path, transform=None, transform_label=None):

        self.df = pd.read_csv(csv_file_path)
        self.transform = transform
        self.transform_label = transform_label
        #  label = [torch.eye(2)[i].numpy() for i in self.df['label'].tolist()]
        X_train, X_valid, y_train, y_valid = \
            train_test_split(self.df['image'].tolist(), self.df['label'].tolist(), test_size=0.2, random_state=42)
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid

    def __len__(self):
        """for return dataset size 
        """
        return len(self.X_train)

    def __getitem__(self, idx):
        """ to support the indexing such that dataset[i] can be used to get ith sample
        """

        train_img = Image.open(self.X_train[idx])
        train_label = self.y_train[idx]
#        val_img = Image.open(X_valid[idx])
#        val_label = y_valid[idx]

        # 画像をtensorに変換
        if self.transform:
            train_img = self.transform(train_img)

        return train_img, train_label#, val_img, val_label


    