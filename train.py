import model
from options.train_options import *
from data.gtsrb_loader import GTSRB, get_loader
from utils.trainer import train_engine
from utils.evaluate import calc_acc_n_loss
from utils.wandb_utils import init_wandb, wandb_save_summary

opt = TrainOptions()

args = opt.initialize()
opt.print_options(args)

train_dataset = GTSRB(args, setname='train')
val_dataset = GTSRB(args, setname='valid')
test_dataset = GTSRB(args, setname='test')

trainloader = get_loader(args, train_dataset)
valloader = get_loader(args, val_dataset)
testloader = get_loader(args, test_dataset)

net, optimizer, schedular = model.CreateModel(args=args)

init_wandb(net, args)

train_engine(args=args, trainloader=trainloader,
             valloader=valloader, model=net, optimizer=optimizer, scheduler=schedular)

test_acc, test_loss = calc_acc_n_loss(args, net, testloader)

print('Test Accuracy =', test_acc)
print('Test Loss =', test_loss)

wandb_save_summary(test_acc=test_acc)