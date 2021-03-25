import argparse
import sys, math
import os
import matplotlib.pyplot as plt 
import torch 
import numpy as np
import tqdm.notebook as tq
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

data_wrong = []
data_right = []

def rise(model1, dataloader, num_classes, image_size, save_path, device=torch.device("cuda:0")):
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
        explainer.generate_masks(N = 4000, s= 8, p1 = 0.1, savepath = 'masks.npy')
        explainer.load_masks('masks.npy')
        c_cnt = np.zeros((48))
        i=0
        j=0
        for data in tq.tqdm(dataloader):
        	if (i>=5 and j>=5):
        		break
            flag = 0
            img = data[0]
            cl = data[1]
            out = model1(img.to(device)).cpu().detach().numpy()[0]
            # print(cl.cpu().detach().numpy(), np.argmax(out))
            if cl[0].cpu().detach().numpy() == np.argmax(out) and c_cnt[cl.cpu().detach().numpy()]<2:
            	i+=1
                c_cnt[cl.cpu().detach().numpy()] += 1
                flag = 1
                # print(str(cl[0].cpu().detach().numpy()), out[np.argmax(out)])
                # data_right.append((name[:-4], str(cl[0].cpu().detach().numpy()), out[np.argmax(out)]))
            elif cl != np.argmax(out):
            	j+=1
                flag = 2
                # data_wrong.append((name[:-4], str(cl[0].cpu().detach().numpy()), np.argmax(out), out[np.argmax(out)]))
            if (flag!=0):
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
                if flag==1:
                    final_img.save(save_path + '/C_Classifications/rise_' + str(i) + '_' + str(cl[0].cpu().detach().numpy()) + '.jpg')
                else:
                    final_img.save(save_path + '/Misclassifications/rise_' + str(j) + '_' + str(cl[0].cpu().detach().numpy()) + '.jpg')
        # filename = save_path + '/c_predictions.csv'
        # with open(filename, 'w') as csvfile:  
        #     # creating a csv writer object  
        #     csvwriter = csv.writer(csvfile)  
                
        #     # writing the fields  
        #     csvwriter.writerow(['Image', 'Real Class', 'Probability'])  
                
        #     # writing the data rows  
        #     csvwriter.writerows(data_right)
        # filename = save_path + '/mispredictions.csv'
        # with open(filename, 'w') as csvfile:  
        #     # creating a csv writer object  
        #     csvwriter = csv.writer(csvfile)  
                
        #     # writing the fields  
        #     csvwriter.writerow(['Image', 'Real Class', 'Predicted Class', 'Probability'])  
                
        #     # writing the data rows  
        #     csvwriter.writerows(data_wrong)
                
opt = TrainOptions()
args = opt.initialize()
test_dataset = GTSRB(args, setname='test')

testloader = get_loader(args, test_dataset)
net, optimizer, schedular = model.CreateModel(args=args)
net.load_state_dict(torch.load("/root/traffic_sign_interiit/opt_micronet_2021-3-11_11-30.pth"))
net.eval()
rise(net, testloader, args['experiment'].num_classes, (48, 48), "/root/traffic_sign_interiit/sample_outputs", torch.device("cpu"))