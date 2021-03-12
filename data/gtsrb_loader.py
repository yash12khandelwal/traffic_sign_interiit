import os
import os.path as osp
from PIL import Image
import numpy as np
from collections import namedtuple
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_loader(args, dataset):
    """ Function that returns dataloader

    Args:
        args (TrainOptions): TrainOptions class (refer options/train_options.py)
        dataset (Dataset): Custom Dataset class

    Returns:
        DataLoader: Dataloader for training or testing
    """
    params = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'shuffle': True
    }

    trainloader = DataLoader(dataset, **params)
    return trainloader

def get_train_tuple(train_path):
    """ Generates a list of images and ground truths for Train DataLoader

    Recursive folder traversal to return the above

    Args:
        train_path (str): Path of the Train or Valid Dataset
        Ex. dataset/GTSRB/train

    Returns:
        tuple: (List of train/val images, List of ground truths)
    """

    train_list = []
    traingt_list = []

    for root, dirs, files in os.walk(train_path):
        for class_dir in dirs:
            # print(class_dir, (os.listdir(osp.join(root, class_dir))))
            mapper = lambda x: osp.join(root, class_dir, x)
            img_loc = list(map(mapper, os.listdir(osp.join(root, class_dir))))
            img_loc = [ f for f in img_loc if not f.endswith('.csv') ]
            class_id = [int(class_dir)]*len(img_loc)
            # print(int(class_dir))
            train_list.extend(img_loc)
            traingt_list.extend(class_id)

    return (train_list, traingt_list)

def get_test_tuple(test_path):
    """ Generates a list of images and ground truths for Test DataLoader

    Reads a csv file provided with GTSRB Dataset and returns the above

    Args:
        test_path (str): Path of the tes Dataset
        Ex. dataset/GTSRB/test

    Returns:
        tuple: (List of test images, List of ground truths)
    """

    test_list = []
    test_ids = []

    test_csv = osp.join(test_path, 'GT-final_test.csv')
    
    with open(test_csv) as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)
        for row in reader:
            filename = row[0] # filename is in the 0th column
            label = int(row[7]) # label is in the 7th column
            test_list.append(osp.join(test_path, filename))
            test_ids.append(label)

    return test_list, test_ids

class GTSRB(Dataset):
    """ 
    Dataset class for GTSRB
    """

    def __init__(self, args, setname='train'):
        """ Init functions calls get_train_sanple or get_test_sample

        Args:
            args (TrainOptions): TrainOptions class (refer options/train_options.py)
            setname (str, optional): Possible values train, val, test for Dataset. Defaults to 'train'.
        """

        self.classes = args.num_classes
        self.setname = setname
        self.path = osp.join(args.data_dir, self.setname)
        self.size = tuple(args.size)
        if self.setname == 'train' or self.setname == 'valid':
            self.imgs, self.ids = get_train_tuple(self.path)
        elif self.setname == 'test':
            self.imgs, self.ids = get_test_tuple(self.path)

    def __len__(self):
        """ Gives the length of Dataset

        Returns:
            int: Length of Dataset
        """

        return len(self.imgs)
    
    def transform(self, image):
        """ Function to apply tranformations

        TODO
        Proper api for working with augmentations

        Args:
            image (PIL Image): PIL Image for applying tranforms

        Returns:
            TorchTensor: Transformed Tensor
        """

        tran = transforms.Compose([
                transforms.Resize(self.size),
                transforms.ToTensor(),
                transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
            ])

        return tran(image)

    def __getitem__(self, idx):
        """ Dataset Method for returning image and class at idx in list

        Args:
            idx (int): DataLoader provides this

        Returns:
            tuple: (Image, Ground Truth) for a setname
        """

        img = Image.open(self.imgs[idx])
        gt = self.ids[idx]

        img = self.transform(img)

        return img, gt