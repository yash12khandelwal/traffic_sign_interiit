import model
from options.train_options import *
from data.gtsrb_loader import GTSRB, get_loader
from utils.trainer import train_engine
from utils.evaluate import calc_acc_n_loss
from utils.utils import set_seed
from utils.wandb_utils import init_wandb, wandb_save_summary

if __name__ == "__main__":

    opt = TrainOptions()
    args = opt.initialize()
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

    if args['experiment'].wandb:
        init_wandb(net, args)

    train_engine(args=args, trainloader=trainloader,
                valloader=valloader, model=net, optimizer=optimizer, scheduler=schedular)

    log_confusion = True if args['experiment'].wandb else False
    test_acc, test_loss,test_f1,cm,test_precision,test_recall = calc_acc_n_loss(args['experiment'], net, testloader, log_confusion)

    print(f'Test Accuracy = {test_acc}')
    print(f'Test Loss = {test_loss}')
    print(f'F1 Score = {test_f1}')
    print(f'Test Precision = {test_precision}')
    print(f'Test Recall = {test_recall}')

    if args['experiment'].wandb:
        wandb_save_summary(test_acc=test_acc,test_f1=test_f1,test_precision=test_precision,test_recall=test_recall)
