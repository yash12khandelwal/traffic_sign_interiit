import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm
import torch
import torch.nn as nn
import json 
from utils.wandb_utils import wandb_log_conf_matrix
from sklearn.metrics import f1_score, multilabel_confusion_matrix, recall_score, precision_score, classification_report


def calc_acc_n_loss(args, model, loader, log_matrix=False):
    """ Function to calculate the Accuracy and Loss given a loader
    Args:
        args (TrainOptions): TrainOptions class (refer options/train_options.py)
        model (Torch Model): Current model object to evaluate
        loader (DataLoader): DataLoader for dataset
        log_matrix   (bool): Whether to log confusion matrix
    Returns:
        tuple: (Model Accuracy, Model Loss)
    """

    model.eval()
    # device='cpu'

    device = args.device

    y_pred = []
    y_true = []

    criterion = nn.CrossEntropyLoss()
    loss = 0

    for img, gt in tqdm(loader):
        img = img.to(device)
        gt = gt.to(device)
        out = model(img)

        loss += criterion(out, gt).item()

        out = torch.argmax(out, dim=-1)

        y_pred.extend(list(out.cpu().numpy()))
        y_true.extend(list(gt.cpu().numpy()))

    f1=f1_score(y_true,y_pred,average='weighted')
    cm=multilabel_confusion_matrix(y_true,y_pred).tolist()
    precision=precision_score(y_true,y_pred,average='weighted')
    recall=recall_score(y_true,y_pred,average='weighted')

    print(classification_report(y_true, y_pred))

    if log_matrix == True:
        wandb_log_conf_matrix(y_true, y_pred)

    acc = sum(1 for x, y in zip(y_true, y_pred) if x == y) * 100 / len(y_true)

    return acc, (loss/len(loader)), f1, cm, precision, recall
