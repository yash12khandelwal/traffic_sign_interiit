import torch
from model.micronet import MicroNet
from data.gtsrb_loader import GTSRB, get_loader
from utils.utils import set_seed
from options.train_options import *
from utils.evaluate import calc_acc_n_loss, calc_acc_n_loss_2
from utils.wandb_utils import init_wandb, wandb_save_summary
import wandb

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
    out = calc_acc_n_loss_2(args['experiment'], model, testloader, log_matrix=log_confusion)

    return out

