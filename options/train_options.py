import argparse
import os
import os.path as osp
import datetime


class TrainOptions():
    """ Training options for cmdline
    """
    def initialize(self):
        """ Parameter definitions

        Returns:
            ArgumentParser.parse_args: Params values for training

            Command line arguments:
            --model [model-key]: Available (dks/micronet), Defaults to micronet
            --device [device]: cpu/cuda, Defaults to cpu
            --snapshot-dir [path]: Path for intermediate snapshots, Defaults to `checkpoints/logs`
            --data-dir [path]: Path to processed dataset, Defaults to `dataset/GTSRB`
            --size [img_w img_h]: Size of input image for the model (will be resized during tranforms), Defaults to [48, 48]
            --batch-size [int]: Batchsize for training and evaluation, Defaults to 1
            --epochs [int]: Number of epochs for training, Defaults to 100
            --epochs-stop [int]: Number of epochs after to stop training, Defaults to 100
            --num-workers [int]: Number workers for dataloader, Defaults to 43
            --learning-rate [float]: Learning rate, Defaults to 2.5e-4
            --momentum [float]: Momentum for optimizer, Defaulst to 0.9
            --weight-decay [float]: Weight Decay for Regularisation, Defaults to 0.0005
            --power [float]: Power for optimizer, Defaults to 0.9
            --num-classes [int]: Number of output classes, Defaults to 43 (GTSRB)
            --init-weights [pth]: Initial model weights, Defaults to None
            --restore-from [path]: Checkpoint dir to restore training, Defaults to None
            --save-pred-every [int]: Save checkpoint and predictions after every such epochs, Defaults to 5
            --print-freq [int]: Printing loss and valid accuracy after every such epochs, Defaults to 5
            --wandb-api-key [key]: Wandb api-key, Defaults to None 
            --wandb_id [id]: Wandb id, Defaults to None
            --class_weights [numpy file]: Class weights (numpy file)
        """

        parser = argparse.ArgumentParser( description="training script for InterIIT Trafic Sign Recognition" )
        parser.add_argument("--model", type=str, default='micronet', help="available options : dks/micronet")
        parser.add_argument("--device", type=str, default='cpu', help="which GPU to use")

        parser.add_argument("--snapshot-dir", type=str, default='checkpoints/logs',
                            help="Where to save snapshots of the model.")
        parser.add_argument("--data-dir", type=str, default='dataset/GTSRB',
                            help="Path to the directory containing the source dataset.")

        parser.add_argument("--size", nargs='+', type=int,
                            default=[32, 32], help="input image size.")

        parser.add_argument("--batch-size", type=int,
                            default=1, help="input batch size.")
        parser.add_argument("--epochs", type=int, default=100,
                            help="Number of training steps.")
        parser.add_argument("--epochs-stop", type=int, default=100,
                            help="Number of training steps for early stopping.")
        parser.add_argument("--num-workers", type=int,
                            default=4, help="number of threads.")
        parser.add_argument("--learning-rate", type=float,
                            default=2.5e-4, help="initial learning rate.")
        parser.add_argument("--momentum", type=float, default=0.9,
                            help="Momentum component of the optimiser.")
        parser.add_argument("--weight-decay", type=float, default=0.0005,
                            help="Regularisation parameter for L2-loss.")
        parser.add_argument("--power", type=float, default=0.9,
                            help="Decay parameter to compute the learning rate (only for deeplab).")

        parser.add_argument("--num-classes", type=int, default=43,
                            help="Number of classes for cityscapes.")
        parser.add_argument("--init-weights", type=str,
                            default=None, help="initial model.")
        parser.add_argument("--restore-from", type=str, default=None,
                            help="Where restore model parameters from.")

        parser.add_argument("--save-pred-every", type=int, default=5, help="Save summaries and checkpoint every often.")
        parser.add_argument("--print-freq", type=int, default=5, help="print loss and time fequency.")

        parser.add_argument("--wandb-api-key", type=str,
                            default=None, help="Wandb API Key")
        parser.add_argument("--wandb_id", type=str, default=None,
                            help="Wandb run resume id (valid only if restore-from != None")
        parser.add_argument("--wandb-name", type=str,
                            default='', help="Name of the wandb run")
        
        parser.add_argument("--class-weights",type=str,default=None,help="Class weights (numpy file)")

        parser.add_argument('--lr-decay-step', type=int, default=5, help="Step size for Learning rate decay")

        return parser.parse_args()

    def print_options(self, args):
        """ Function that prints and saves the output
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------\n'
        print(message)

        # save to the disk
        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)
        
        t = datetime.datetime.now()
        name = f'opt_{args.model}_{t.year}-{t.month}-{t.day}_{t.hour}-{t.minute}.txt'
        file_name = osp.join(args.snapshot_dir, name)
        with open(file_name, 'wt') as args_file:
            args_file.write(message)
            args_file.write('\n')
