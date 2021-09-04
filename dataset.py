import torch
import cv2
import numpy as np
from params import *


class Dataset(torch.utils.data.Dataset):
    def __init__(self, start_index, dir_path, flatten):
        self.start_index = start_index
        self.dir_path = dir_path
        self.length = len(os.listdir(self.dir_path))
        self.flatten = flatten

    def __len__(self):
        return self.length - self.start_index

    def __getitem__(self, index):
        image_path = self.dir_path + "//frame" + str(index) + ".jpg"
        image = cv2.imread(image_path)
        image = image / 255
        image = np.float32(image)
        if self.flatten:
            image = image.flatten()
        else:
            image = np.reshape(image, (3, 200, 160))
        return image, image


class CompressDataset(torch.utils.data.Dataset):
    def __init__(self, images, flatten):
        self.images = images
        self.length = len(self.images)
        self.flatten = flatten

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image = self.images[index]
        image = image / 255
        image = np.float32(image)
        if self.flatten:
            image = image.flatten()
        else:
            image = np.reshape(image, (3, 200, 160))
        return image, image