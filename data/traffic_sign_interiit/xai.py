import torch
import argparse
from model.micronet import MicroNet
from data.gtsrb_loader import GTSRB, get_loader
from utils.utils import set_seed
from options.train_options import *
from utils.evaluate import calc_acc_n_loss, calc_acc_n_loss_2
from utils.wandb_utils import init_wandb, wandb_save_summary
import wandb
import argparse
import sys, math
import os
import matplotlib.pyplot as plt 
import torch 
import numpy as np
from skimage.transform import resize
import model
from options.train_options import *
from data.gtsrb_loader import GTSRB, get_loader
from utils.trainer import train_engine
from utils.evaluate import calc_acc_n_loss
from utils.utils import set_seed
from utils.wandb_utils import init_wandb, wandb_save_summary
from RISE.evaluation import CausalMetric, auc, gkern
from RISE.explanations import RISEBatch
from tqdm import tqdm
import csv
from PIL import Image 
import matplotlib as mpl
import csv

def rise(model1, dataloader, num_classes, cl, save_path, image_size=(48, 48), device=torch.device("cpu")):
    """
    A function to create the saliency maps for all the images in the dataloader

    Args:
    model: The pretrained model (make sure to have it in eval mode)
    dataloader: the dataloader whos images are to be used.
    num_classes: number of output classes
    save_path: the path to the directory where the images will be saved.
    device: the device where you want to compute the maps.
    """

    #Create the explainer
    explainer = RISEBatch(model1, image_size, gpu_batch=20)

    #Create the masks once, and save them. For later use, simply load the masks.
    # explainer.generate_masks(N = 4000, s= 8, p1 = 0.1, savepath = 'masks.npy')
    explainer.load_masks('masks.npy')
    
    for data in tqdm(dataloader):
        img = data[0]
        img = img.float().to(device)

        #Generate the saliency map
        sal = explainer(img.to(device)).cpu().numpy()

        #Display the saliency map

        img_cpu = img.clone().cpu().numpy()[0]
        mean = [0.3337, 0.3064, 0.3171]
        std =  [0.2672, 0.2564, 0.2629]

        for channel in range(3):
            img_cpu[channel] = img_cpu[channel]*std[channel] + mean[channel]

        img_cpu = img_cpu*255

        img_cpu = img_cpu.transpose((1,2,0))
        img_cpu[img_cpu < 0] = 0

        explanation = sal[0, cl]
        explanation -= np.min(explanation)
        explanation /= np.max(explanation)
        cm = mpl.cm.get_cmap('jet')
        explanation = cm(explanation)
        explanation = explanation[:, :, :3]
        explanation *=255

        final_img = explanation * 0.5 + img_cpu * 0.5
        final_img = final_img.astype(np.uint8)
        final_img = Image.fromarray(final_img)
        print(os.path.join(save_path, 'rise.jpg'))
        final_img.save(os.path.join(save_path, 'rise.jpg'))