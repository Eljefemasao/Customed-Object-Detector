
# markdataのloaderクラス

import torch
from torch.utils.data  import Dataset
import cv2
import os 
from PIL import Image 
import pandas as pd 
from sklearn.model_selection import train_test_split



class DatasetLoader(Dataset):

    def __init__(self, csv_file_path,transform=None):
        self.df = pd.read_csv(csv_file_path)
        self.transform = transform

    def __len__(self):
        """for return dataset size 
        """
        return len(self.df)

    def __getitem__(self, idx):
        """ to support the indexing such that dataset[i] can be used to get ith sample
        """

        X_train, X_valid, y_train, y_valid = \
            train_test_split(self.df['image'].tolist(), self.df['label'].tolist(), test_size=0.2, random_state=42)

        #train_data = {'image': X_train,'label': y_train}
        #validation_data = {'image':X_valid, 'label':y_valid}        

        train_img = Image.open(X_train[idx])
        train_label = y_train[idx]
        val_img = Image.open(X_valid[idx])
        val_label = y_valid[idx]

        if self.transform:
            train_img = self.transform(train_img)

        return train_img, train_label, val_img, val_label

    