import torch
from model.micronet import MicroNet
from data.gtsrb_loader import GTSRB, get_loader
from utils.utils import set_seed
from options.train_options import *
from utils.evaluate import calc_acc_n_loss
from utils.wandb_utils import init_wandb, wandb_save_summary
import wandb

if __name__ == "__main__":

    opts = TrainOptions()
    args = opts.initialize()

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
    acc, loss, f1, cm, precision, recall = calc_acc_n_loss(args['experiment'], model, testloader, log_matrix=log_confusion)

    print(f'Test Accuracy: {acc}')
    print(f'F1 Score: {f1}')
    print(f'Precision Score: {precision}')
    print(f'Recall Score: {recall}')

    if args['experiment'].wandb:
        wandb.run.summary['test_accuracy'] = acc
        wandb.run.summary["test_f1"] = f1*100
        wandb.run.summary["test_precision"] = precision*100
        wandb.run.summary["test_recall"] = recall*100
