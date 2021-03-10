from PIL import Image
import glob
import torch
import torch.utils.data as data
# torch.utils.data.dataset is an abstract class representing a dataset
from torch.utils.data.dataset import Dataset
# https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html
import os
import torch
import numpy as np
import pandas as pd
import sys
import csv


'''
Uses pytorch Dataset to load some low and high resolution image data.
'''


class DIV2K(Dataset):
    """
    CIFAR dataset
    Implements Dataset (torch.utils.data.dataset)
    """

    def __init__(self, hr_dir, lr_dir, transHR, transLR):
        """
        Args:
            data_dir (string): Directory with all the images
        """
        # gets the data from the directory
        self.lr_image_list = glob.glob(lr_dir + '*')
        self.hr_image_list = glob.glob(hr_dir + '*')

        # calculates the length of image_list
        # print(self.hr_image_list)
        # print(self.lr_image_list)
        assert (len(self.hr_image_list) == len(self.lr_image_list))

        self.data_len = len(self.hr_image_list)

        self.transLR = transLR
        self.transHR = transHR

    def __getitem__(self, index):
        """
        Get item at certain INDEX

        """
        LR_single_image_path = self.lr_image_list[index]
        HR_single_image_path = self.hr_image_list[index]

        LR_image = Image.open(LR_single_image_path)
        HR_image = Image.open(HR_single_image_path)

        LR_trans = self.transLR(LR_image)
        HR_trans = self.transHR(HR_image)

        return (LR_trans, HR_trans)

    def __len__(self):
        return self.data_len
