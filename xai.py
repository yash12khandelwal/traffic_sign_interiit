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

def rise(model, dataloader, num_classes, image_size, save_path, device=torch.device("cuda:0")):
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
        explainer = RISEBatch(model, image_size, gpu_batch=20)

        #Create the masks once, and save them. For later use, simply load the masks.
        explainer.generate_masks(N = 4000, s= 8, p1 = 0.1, savepath = 'masks.npy')
        # explainer.load_masks('masks.npy')

        #Iterate over the dataloader
        for i, data in tqdm(enumerate(dataloader, 0), desc="Images" ):
                #Currently stopping after 5 images
                if i > 5:
                        break

                #Extract the image and class from the data element
                img = data[0]
                img = img.float().to(device)
                cl = data[1]

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
                # explanation = explanation.astype(np.uint8)

                final_img = explanation * 0.5 + img_cpu * 0.5
                final_img = final_img.astype(np.uint8)
                final_img = Image.fromarray(final_img)

                final_img.save(save_path + '/rise_' + str(i) + '.jpg')
                print("image: {}".format(i), flush = True)
                
opt = TrainOptions()
args = opt.initialize()
train_dataset = GTSRB(args, setname='train')

trainloader = get_loader(args, train_dataset)
net, optimizer, schedular = model.CreateModel(args=args)
net.load_state_dict(torch.load("/root/traffic_sign_interiit/opt_micronet_2021-3-11_11-30.pth"))
net.eval()
rise(net, trainloader, 43, (48, 48), "/root/traffic_sign_interiit/sample_outputs", torch.device("cuda:0"), )