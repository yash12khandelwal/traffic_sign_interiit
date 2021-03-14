import pandas as pd
import os.path as osp
from tqdm import tqdm
import torch
import torch.nn as nn

def calc_acc_n_loss(args, model, loader):
    """ Function to calculate the Accuracy and Loss given a loader

    Args:
        args (TrainOptions): TrainOptions class (refer options/train_options.py)
        model (Torch Model): Current model object to evaluate
        loader (DataLoader): DataLoader for dataset

    Returns:
        tuple: (Model Accuracy, Model Loss)
    """

    model.eval()

    device = args.device

    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()
    loss = 0

    for img, gt in tqdm(loader):
        img = img.to(device)
        gt = gt.to(device)
        out = model(img)

        loss += criterion(out, gt).item()

        out = torch.argmax(out, dim=-1)

        correct += (out == gt).sum()
        total += len(gt)

    return (correct/total).item()*100, (loss/len(loader))