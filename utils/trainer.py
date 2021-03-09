from numpy.core.fromnumeric import argmax
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.evaluate import calc_acc_n_loss
from utils.wandb_utils import wandb_log, save_model_wandb


def train_engine(args, train_dataset, val_dataset, model, optimizer, scheduler=None):
    criterion = nn.CrossEntropyLoss()

    device = args.device

    params = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'shuffle': True
    }

    trainloader = DataLoader(train_dataset, **params)
    valloader = DataLoader(val_dataset, **params)

    for i in range(args.epochs):

        model.train()

        train_loss = 0.0

        print('-'*50)
        print('\nEpoch =', i)
        for (img, gt) in tqdm(trainloader):
            optimizer.zero_grad()

            img, gt = img.to(device), gt.to(device)
            out = model(img)
            loss = criterion(out, gt)

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
            curr_lr = scheduler.get_last_lr()

        print('\nValidating ...')
        val_acc, val_loss = calc_acc_n_loss(args, model, valloader)
        print(f'Valid Accuracy = {val_acc} %')
        print('Valid loss =', val_loss)
        print('-'*50)

        if (i+1) % args.save_pred_every == 0:
            print('Taking snapshot ...')
            if not os.path.exists(args.snapshotdir):
                os.makedirs(args.snapshotdir)
            save_path = os.path.join(args.snapshotdir, str(i+1) + '.pth')
            torch.save(model.state_dict(), save_path)

        wandb_log(train_loss/len(trainloader), val_loss, val_acc, i)
