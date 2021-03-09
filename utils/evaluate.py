import pandas as pd
import os.path as osp
from sklearn.metrics import accuracy_score
from torch.nn.modules import loss
from tqdm import tqdm
import torch
import torch.nn as nn

def calc_acc_n_loss(args, model, loader):
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