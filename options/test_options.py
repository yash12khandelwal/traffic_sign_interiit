import argparse
import os.path as osp

class TestOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="test segmentation network")
        parser.add_argument("--model", type=str, default='DeepLab', help="available options : DeepLab and VGG")
        parser.add_argument("--GPU", type=str, default='0', help="which GPU to use")

        parser.add_argument("--data-dir-test", type=str, default='../data_semseg/cityscapes', help="Path to the directory containing the target dataset.")
        parser.add_argument("--num-classes", type=int, default=19, help="Number of classes for cityscapes.")

        parser.add_argument("--restore-from", type=str, default=None, help="restore model parameters from")

        parser.add_argument("--save", type=str, default='../results', help="Path to save result.")

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
        file_name = osp.join(args.save, 'opt.txt')
        with open(file_name, 'wt') as args_file:
            args_file.write(message)
            args_file.write('\n')