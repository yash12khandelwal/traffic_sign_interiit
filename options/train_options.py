import argparse
import os
import os.path as osp


class TrainOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(
            description="training script for InterIIT Trafic Sign Recognition")
        parser.add_argument("--model", type=str,
                            default='dks', help="available options : ")
        parser.add_argument("--device", type=str,
                            default='cpu', help="which GPU to use")

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

        parser.add_argument("--save-pred-every", type=int, default=5,
                            help="Save summaries and checkpoint every often.")

        parser.add_argument("--wandb-api-key", type=str,
                            default=None, help="Wandb API Key")
        parser.add_argument("--wandb_id", type=str, default=None,
                            help="Wandb run resume id (valid only if restore-from != None")
        parser.add_argument("--wandb-name", type=str,
                            default='', help="Name of the wandb run")

        return parser.parse_args()

    def print_options(self, args):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        file_name = osp.join(args.snapshot_dir, 'opt.txt')
        with open(file_name, 'wt') as args_file:
            args_file.write(message)
            args_file.write('\n')
