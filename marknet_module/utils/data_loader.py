
# markdataのloaderクラス

import torch
from torch.utils.data  import Dataset
import cv2
import os 
from PIL  import image 




class DatasetLoader(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        """for return dataset size 
        """
        return len(self.df)

    def __getitem__(self, idx):
        """ to support the indexing such that dataset[i] can be used to get ith sample
        """

        data = {'image': ,'label': }
        return data

    