import model
from options.train_options import *
from data.gtsrb_loader import GTSRB
from utils.trainer import train_engine
from utils.evaluate import calc_acc_n_loss
from torch.utils.data import DataLoader
from utils.wandb_utils import init_wandb

opt = TrainOptions()

args = opt.initialize()
opt.print_options(args)

train_dataset = GTSRB(args, setname='train')
val_dataset = GTSRB(args, setname='valid')
test_dataset = GTSRB(args, setname='test')


net, optimizer = model.CreateModel(args=args)

init_wandb(net, args)

train_engine(args=args, train_dataset=train_dataset,
             val_dataset=val_dataset, model=net, optimizer=optimizer)

params = {
    'batch_size': args.batch_size,
    'num_workers': args.num_workers,
    'shuffle': True
}

testloader = DataLoader(test_dataset, **params)

test_acc, test_loss = calc_acc_n_loss(args, net, testloader)

print('Test Accuracy =', test_acc)
print('Test Loss =', test_loss)
