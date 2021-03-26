import torch
import model
from options.train_options import *
from data.gtsrb_loader import GTSRB, get_loader
from utils.trainer import train_engine
from utils.evaluate import calc_acc_n_loss
from utils.utils import set_seed
from utils.wandb_utils import init_wandb, wandb_save_summary

root_dir = "data/traffic_sign_interiit/checkpoints/logs/"

def train(config_file=""):

    opt = TrainOptions()
    args = opt.initialize(config_file=config_file)
    opt.print_options(args)

    # setting seed system wide for proper reproducibility
    set_seed(int(args['experiment'].seed))

    train_dataset = GTSRB(args, setname='train')
    val_dataset = GTSRB(args, setname='valid')
    test_dataset = GTSRB(args, setname='test')

    trainloader = get_loader(args, train_dataset)
    valloader = get_loader(args, val_dataset)
    testloader = get_loader(args, test_dataset)

    net, optimizer, schedular = model.CreateModel(args=args)

    if args['experiment'].restore_from:
        device = torch.device(args['experiment'].device)
        PATH = args['experiment'].restore_from
        checkpoints = torch.load(PATH, map_location=device)

        net.load_state_dict(checkpoints['model_state_dict'])
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])

    if args['experiment'].wandb:
        init_wandb(net, args)

    train_engine(args=args, trainloader=trainloader,
                valloader=valloader, model=net, optimizer=optimizer, scheduler=schedular, next_config=config_file)

    log_confusion = True if args['experiment'].wandb else False
    test_acc, test_loss,test_f1,cm,test_precision,test_recall = calc_acc_n_loss(args['experiment'], net, testloader, log_confusion)

    print(f'Test Accuracy = {test_acc}')
    print(f'Test Loss = {test_loss}')
    print(f'F1 Score = {test_f1}')
    print(f'Test Precision = {test_precision}')
    print(f'Test Recall = {test_recall}')

    f = open(root_dir + config_file + "/" + config_file+"_train.txt","w+")
    f.write(str(round(test_acc, 3)) + " " + str(round(test_loss, 3)) + " " + str(round(test_f1, 3)) + " " + str(round(test_precision, 3)) + " " + str(round(test_recall, 3)))    
    f.close()

    if args['experiment'].wandb:
        wandb_save_summary(test_acc=test_acc,test_f1=test_f1,test_precision=test_precision,test_recall=test_recall)
