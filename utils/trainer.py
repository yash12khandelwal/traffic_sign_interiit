import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.evaluate import calc_acc_n_loss
from utils.wandb_utils import wandb_log, save_model_wandb
from utils.utils import convert_onnx
import numpy as np
import os
import datetime


def train_engine(args, trainloader, valloader, model, optimizer, scheduler=None):
    """ Generic Train function for training

    Args:
        args (TrainOptions): TrainOptions class (refer options/train_options.py)
        train_dataset (Dataset): Train Dataset class object
        val_dataset ([type]): Valid Dataset class object
        model (Torch Model): Model for training
        optimizer (Optimizer): Optimizer
        scheduler (LR Schedular, optional): Changing learning rate according to a function. Defaults to None.
    """

    args = args['experiment']

    device = args.device
    if args.class_weights is None:
        weight = None
    else:
        if os.path.isfile(args.class_weights):
            weight = torch.from_numpy(np.load(args.class_weights))
            weight = weight.type(torch.FloatTensor).to(device)
        else:
            raise ValueError('Class weights file not found')

    criterion = nn.CrossEntropyLoss(weight=weight)

    for i in range(args.epochs):

        model.train()

        train_loss = 0.0
        correct = 0
        total = 0
        print('-'*50)
        print('\nEpoch =', i)
        for (img, gt) in tqdm(trainloader):
            optimizer.zero_grad()

            img, gt = img.to(device), gt.to(device)
            out = model(img)
            loss = criterion(out, gt)

            train_loss += loss.item()

            out = torch.argmax(out, dim=-1)
            correct += (out == gt).sum()
            total += len(gt)

            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
            curr_lr = scheduler.get_last_lr()
            print('\nCurrent Learning Rate =', curr_lr)

        train_acc = (correct/total).item()*100
        train_loss /= len(trainloader)
        print(f'Train Accuracy = {train_acc} %')
        print(f'Train loss = {train_loss}')

        print('\nValidating ...')
        val_acc, val_loss, val_f1, val_cm, val_precision, val_recall = calc_acc_n_loss(args, model, valloader, False)
        print(f'Valid Accuracy = {val_acc} %')
        print(f'Valid loss = {val_loss} %')
        print(f'Valid f1 = {val_f1} %')
        print(f'Valid ConfusionMatrix = {val_cm} %')
        print(f'Valid precision = {val_precision} %')
        print(f'Valid recall = {val_recall}')
        print('-'*50)

        if (i+1) % args.save_pred_every == 0:
            print('Taking snapshot ...')
            if not os.path.exists(args.snapshot_dir):
                os.makedirs(args.snapshot_dir)
            save_path = os.path.join(
                args.snapshot_dir, f'{args.model}_{i+1}.pt')
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, save_path)
            if args.wandb:
                save_model_wandb(save_path)

        if args.wandb:
            wandb_log(train_loss, val_loss, train_acc, val_acc, i)

    t = datetime.datetime.now()
    name = f'final_{args.model}{t.year}-{t.month}-{t.day}{t.hour}-{t.minute}.pt'

    save_path = os.path.join(args.snapshot_dir, name)
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
    }, save_path)

    onnx_save_path = convert_onnx(model, args)

    if args.wandb:
        save_model_wandb(save_path)
        save_model_wandb(onnx_save_path)

    return model
