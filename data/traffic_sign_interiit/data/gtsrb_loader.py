import os
import os.path as osp
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from augments.augs import load_augments
import cv2

root_dir = "data/traffic_sign_interiit/"

def get_loader(args, dataset):
    """ 
    Function that returns dataloader

    Args:
        args (TrainOptions): TrainOptions class (refer options/train_options.py)
        dataset (Dataset): Custom Dataset class

    Returns:
        DataLoader: Dataloader for training or testing
    """

    args = args['experiment']
    params = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'shuffle': True
    }

    dataloader = DataLoader(dataset, **params)
    return dataloader


def get_train_tuple(train_path, extra_train_path=None):
    """
    Generates a list of images and ground truths for Train DataLoader
    Recursive folder traversal to return the above

    Args:
        train_path (str): Path of the Train or Valid Dataset
        extra_train_path (str, optional): Path of the Extra Classes Train or Valid Dataset. Defaults to None.

    Returns:
        tuple: (List of train/val images, List of ground truths)
    """

    train_list = []
    traingt_list = []

    for classid in os.listdir(root_dir + train_path):
        class_csv = osp.join(root_dir + train_path, classid, f'GT-{classid}.csv')

        reader = csv.reader(open(class_csv, 'r'), delimiter=';')
        next(reader)

        for row in reader:
            train_list.append(row[0])
            traingt_list.append(int(row[1]))

    if extra_train_path is not None:
        for classid in os.listdir(extra_train_path):
            class_csv = open(
                osp.join(extra_train_path, classid, f'GT-{classid}.csv'))

            reader = csv.reader(class_csv, delimiter=';')
            next(reader)

            for row in reader:
                train_list.append(row[0])
                traingt_list.append(int(row[1]))

    return train_list, traingt_list


def get_test_tuple(test_path, extra_test_path=None):
    """ 
    Generates a list of images and ground truths for Test DataLoader
    Reads a csv file provided with GTSRB Dataset and returns the above

    Args:
        test_path (str): Path of the Test Dataset
        extra_train_path (str, optional): Path of the Extra Classes Test Dataset. Defaults to None.

    Returns:
        tuple: (List of test images, List of ground truths)
    """

    test_list = []
    test_ids = []

    test_csv = osp.join(test_path, 'GT-Test.csv')

    with open(root_dir + test_csv) as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)
        for row in reader:
            test_list.append(row[0])
            test_ids.append(int(row[1]))

    if extra_test_path is not None:
        extra_test_csv = osp.join(root_dir, osp.join(extra_test_path, 'GT-Test.csv'))

        with open(extra_test_csv) as f:
            reader = csv.reader(f, delimiter=';')
            next(reader)
            for row in reader:
                test_list.append(row[0])
                test_ids.append(int(row[1]))

    return (test_list, test_ids)

class GTSRB(Dataset):
    """
    Dataset class for GTSRB
    """

    def __init__(self, args, setname='train'):
        """ 
        Constructor calls get_train_sample or get_test_sample, based on setname

        Args:
            args (TrainOptions): TrainOptions class (refer options/train_options.py)
            setname (str, optional): Possible values train, val, test for Dataset. Defaults to 'train'.
        """

        self.args = args['experiment']
        self.augment_args = args['augmentations']
        self.classes = self.args.num_classes
        self.setname = setname
        self.path = osp.join(self.args.data_dir, self.setname)
        self.size = tuple(self.args.size)

        extra_class_path = None if self.args.extra_path is None else \
            osp.join(self.args.extra_path, self.setname)

        if self.setname == 'train' or self.setname == 'valid':
            self.imgs, self.ids = get_train_tuple(self.path, extra_class_path)
        elif self.setname == 'test':
            self.imgs, self.ids = get_test_tuple(self.path, extra_class_path)

    def __len__(self):
        """
        Gives the length of Dataset

        Returns:
            int: Length of Dataset
        """

        return len(self.imgs)

    def transform(self, image):
        """ 
        Function to apply tranformations

        TODO
        Proper api for working with augmentations

        Args:
            image (np.array): OpenCV Image (np.array) for applying tranforms

        Returns:
            TorchTensor: Transformed Tensor
        """
        image = cv2.resize(image, self.size)
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171),
                                 (0.2672, 0.2564, 0.2629))
        ])

        if self.setname == 'train':
            # Applying augmentations to the image
            image = load_augments(self.augment_args, top=1)(image=image)
            return trans(image)
        else:
            return trans(image)

    def __getitem__(self, idx):
        """ 
        Dataset Method for returning image and class at idx in list

        Args:
            idx (int): DataLoader provides this

        Returns:
            tuple: (Image, Ground Truth) for a setname
        """

        img = cv2.imread(osp.join(root_dir, self.imgs[idx]), 1)
        gt = self.ids[idx]

        img = self.transform(img)

        return img, gt
