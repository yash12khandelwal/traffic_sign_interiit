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
import xai

def test(config_file=""):

    opts = TrainOptions()
    args = opts.initialize(config_file)

    set_seed(int(args['experiment'].seed))

    model = MicroNet(args['experiment'])
    device = args['experiment'].device

    PATH = args['experiment'].restore_from
    checkpoint = torch.load(PATH, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if args['experiment'].wandb:
        init_wandb(model, args)

    test_dataset = GTSRB(args, setname='test')
    testloader = get_loader(args, test_dataset)

    log_confusion = True if args['experiment'].wandb else False
    out, histo= calc_acc_n_loss_2(args['experiment'], model, testloader, log_matrix=log_confusion)

    xai.rise(model, testloader, args["experiment"].num_classes, out, "data/traffic_sign_interiit/dataset/New_Test/")

    return out, histo

