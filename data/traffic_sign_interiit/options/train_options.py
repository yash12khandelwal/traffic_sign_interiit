import argparse
import os
import os.path as osp
import datetime
from config.cfg_parser import cfg_parser


class TrainOptions():
    """ 
    Training options for commandline
    """

    def initialize(self, config_file):
        """ 
        Parameter definitions

        Returns:
            ArgumentParser.parse_args: Params values for training

            Command line arguments:
            --version [str]: name of the config file to use
            --wandb [bool]: Log to wandb or not
        """
        version = ""
        if config_file=="":
            version = 'params'
        else:
            version = config_file
        wandb = False
        cfg = cfg_parser(osp.join("config", version + '.json'))
        cfg['experiment'].wandb = wandb
        cfg['experiment'].version = version
        cfg['experiment'].wandb_id = "id"
        cfg['experiment'].wandb_name = "MicroNet-Tsinghwa-iter-exp3-without-dropout"
        cfg['experiment'].snapshot_dir = os.path.join(cfg['experiment'].snapshot_dir, version)

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
