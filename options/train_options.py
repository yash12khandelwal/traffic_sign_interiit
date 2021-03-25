import argparse
import os
import os.path as osp
import datetime
from config.cfg_parser import cfg_parser


class TrainOptions():
    """ 
    Training options for commandline
    """

    def initialize(self):
        """ 
        Parameter definitions

        Returns:
            ArgumentParser.parse_args: Params values for training

            Command line arguments:
            --version [str]: name of the config file to use
            --wandb [bool]: Log to wandb or not
        """

        parser = argparse.ArgumentParser(
            description="training script for InterIIT Trafic Sign Recognition")
        parser.add_argument(
            "-v",
            "--version",
            type=str,
            default="params",
            help="name of the config file to use.",
        )
        parser.add_argument(
            "-w", "--wandb", action="store_true", help="Log to wandb or not"
        )
        args = parser.parse_args()

        cfg = cfg_parser(osp.join("config", args.version + '.json'))
        cfg['experiment'].wandb = args.wandb
        cfg['experiment'].version = args.version
        cfg['experiment'].snapshot_dir = os.path.join(
            cfg['experiment'].snapshot_dir, args.version)

        return cfg

    def print_options(self, args):
        """ 
        Function that prints and saves the output
        """

        message = ''
        message += f'----------------- Options ----------------\n'
        for k, v in sorted(vars(args['experiment']).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------\n'
        print(message)

        # save to the disk
        if not os.path.exists(args['experiment'].snapshot_dir):
            os.makedirs(args['experiment'].snapshot_dir)

        t = datetime.datetime.now()

        name = f'opt_{args["experiment"].model}_{t.year}-{t.month}-{t.day}_{t.hour}-{t.minute}.txt'
        file_name = osp.join(args['experiment'].snapshot_dir, name)

        with open(file_name, 'wt') as args_file:
            args_file.write(message)
            args_file.write('\n')
