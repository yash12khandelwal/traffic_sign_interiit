import model
from options.train_options import *
from data.gtsrb_loader import GTSRB, get_loader
from utils.trainer import train_engine
from utils.evaluate import calc_acc_n_loss
from utils.utils import set_seed
from utils.wandb_utils import init_wandb, wandb_save_summary

if __name__ == "__main__":

    #TODO Instead of using so many arguments, use a config file for each run. Just change the relevant parameters in the config file.
    
    opt = TrainOptions()
    args = opt.initialize()
    opt.print_options(args)

    # setting seed system wide for proper reproducibility
    set_seed(int(args.seed))

    train_dataset = GTSRB(args, setname='train')
    val_dataset = GTSRB(args, setname='valid')
    test_dataset = GTSRB(args, setname='test')

    trainloader = get_loader(args, train_dataset)
    valloader = get_loader(args, val_dataset)
    testloader = get_loader(args, test_dataset)

    net, optimizer, schedular = model.CreateModel(args=args)

    if args.wandb:
        init_wandb(net, args)

    train_engine(args=args, trainloader=trainloader,
                valloader=valloader, model=net, optimizer=optimizer, scheduler=schedular)

    test_acc, test_loss = calc_acc_n_loss(args, net, testloader, True)

    print(f'Test Accuracy = {test_acc}')
    print(f'Test Loss = {test_loss}')

    if args.wandb:
        wandb_save_summary(test_acc=test_acc)
