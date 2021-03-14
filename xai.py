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

def rise_eval(model, device, dataloader, num_classes):
        explainer = RISEBatch(model, (48, 48), gpu_batch=4)
        # explainer.load_masks('masks.npy')
        explainer.generate_masks(N = 4000, s= 8, p1 = 0.1, savepath = 'masks.npy')

        klen = 11
        ksig = 5
        kern = gkern(klen, ksig)

        blur = lambda x: torch.nn.functional.conv2d(x, kern, padding=klen//2)

        insertion = CausalMetric(model, 'ins', 48*8, substrate_fn=blur, n_classes=num_classes, device=device)
        deletion = CausalMetric(model, 'del', 48*8, substrate_fn=torch.zeros_like, n_classes=num_classes, device=device)

        for i, data in tqdm(enumerate(dataloader, 0), desc="Images" ):
                if i > 5:
                        break
                img = data[0]
                print(img.shape)
                img = img.float().to(device)
                cl = data[1]
                sal = explainer(img.to(device)).cpu().numpy()
                print("Sal Shape : {}".format(sal.shape))

                """
                sal_batch = np.empty((1, sal.shape[2], sal.shape[3]))
                sal_batch[0, :, :] = sal[:, cl, :, :]
                scores2 = deletion.single_run(img.cpu(), sal_batch, verbose=0, save_to =None)
                scores1 = insertion.single_run(img.cpu(), sal_batch, verbose=0, save_to = None)
                """

                img_cpu = img.clone().cpu().numpy()[0]
                print(img_cpu.shape)
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

                final_img.save('/root/traffic_sign_interiit' + '/rise_' + str(i) + '.jpg')
                # plt.imshow(img_cpu.astype(np.uint8))
                # plt.imshow(sal_batch[0], alpha=0.5, cmap='jet')
                # plt.axis('off')
                # plt.savefig(args.image_path + '/rise_' + str(idx) + '.jpg')

                # ins_auc = auc(scores1)
                # del_auc = auc(scores2)
                # ins_auc = 0
                # del_auc = 0
                # print("------------Summary (so far)----------------------------", flush=True)
                print("image: {}".format(i), flush = True)
                # print("insertion score : ",ins_auc, flush = True)
                # print("deletion score", del_auc, flush = True)
                # print("--------------------------------------------------------", flush = True)
        del explainer
        explainer = None
        # torch.cuda.empty_cache()
        # return img_path, cl
        # return img_path, ins_auc, del_auc, cl

opt = TrainOptions()
args = opt.initialize()
train_dataset = GTSRB(args, setname='train')

trainloader = get_loader(args, train_dataset)
net, optimizer, schedular = model.CreateModel(args=args)
net.load_state_dict(torch.load("/root/traffic_sign_interiit/opt_micronet_2021-3-11_11-30.pth"))
net.eval()

print("---")
print(net)
print(args)
print("---")

# x = torch.ones((4, 3, 48, 48))
# x = x.float().cuda()
# net(x)
rise_eval(net, torch.device("cuda:0"), trainloader, 43)