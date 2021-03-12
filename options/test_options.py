import argparse
import os
import os.path as osp
import datetime

class TestOptions():
    def initialize(self):
        """ Definition of Test cmd line parameters

        Returns:
            ArgumentParser.parse_args: Params values for training

            Command line arguments:
            --model [model-key]: Available (dks/micronet), Defaults to micronet
            --device [device]: cpu/cuda, Defaults to cpu
            --data-dir-test [path]: Path to test set
            --num-workers [int]: Number workers for dataloader, Defaults to 43
            --restore-from [path]: Checkpoint dir to restore training, Defaults to None
            --save [path]: Path to save the metrics and outputs
        """
    
        parser = argparse.ArgumentParser(description="test segmentation network")
        parser.add_argument("--model", type=str, default='micronet', help="available options : dks/micronet")
        parser.add_argument("--device", type=str, default='cpu', help="which device to use")

        parser.add_argument("--data-dir-test", type=str, default='../dataset/GTSRB/test', help="Path to the directory containing the target dataset.")
        parser.add_argument("--num-classes", type=int, default=19, help="Number of classes for cityscapes.")

        parser.add_argument("--restore-from", type=str, default=None, help="restore model parameters from")

        parser.add_argument("--save", type=str, default='results', help="Path to save result.")

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