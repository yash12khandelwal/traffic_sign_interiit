import argparse
import os
import os.path as osp
import datetime

from config.cfg_parser import cfg_parser

class TrainOptions():
    """ Training options for cmdline
    """

    def initialize(self):
        """ Parameter definitions

        Returns:
            ArgumentParser.parse_args: Params values for training

            Command line arguments:
            --version [str]: name of the config file to use
            --wandb [bool]: Log to wandb or not
        """

        parser = argparse.ArgumentParser( description="training script for InterIIT Trafic Sign Recognition" )
        parser.add_argument(
            "-v",
            "--version",
            type=str,
            default="default_params.json",
            help="name of the config file to use.",
        )
        parser.add_argument(
            "-w", "--wandb", action="store_true", help="Log to wandb or not"
        )
        args=parser.parse_args()
        cfg = cfg_parser(osp.join("config", args.version))
        cfg.wandb = args.wandb
        cfg.version = args.version
        
        return cfg

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
