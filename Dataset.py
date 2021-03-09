import cv2
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import time
import random

class TrafficDataset(data.Dataset):
    def _init_(self, list_IDs, labels, height, width, fns):
        
        self.labels = labels
        self.list_IDs = list_IDs
        self.height = height
        self.fns = fns

    def _getitem_(self, index):
        ID = self.list_IDs[index]
        image = Image.open(ID)
        
        rcrop = transforms.RandomResizedCrop(size=(self.width, self.height), scale=(0.08, 1.0))
        if self.fns[0]:
            image = rcrop(image)
        #Resize
        resize = transforms.Resize(size=(self.width, self.height))
        else:
            image = resize(image)

        # ColorJitter
        color = transforms.ColorJitter(brightness=.05, contrast=.05, hue=.05, saturation=.05)
        if self.fns[1]:
            image = color(image)

        raffine = transforms.RandomAffine(degrees = alpha, translate=beta, scale=gamma, shear=delta)
        if self.fns[2]:
            image = raffine(image)

        gblur = transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
        if self.fns[3]:
            image = gblur(image)

        if self.fns[4]:
            image = transforms.functional.adjust_sharpness(image, sharpness_factor: float_value)

        return image

    def _len_(self):
        return len(self.list_IDs)