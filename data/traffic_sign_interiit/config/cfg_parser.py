import json
import os

root_dir = "data/traffic_sign_interiit/"

class Config(object):
    """
    Class for all attributes and functionalities related to a particular training run.
    """

    def __init__(self, cfg_file: str, params: dict):
        """
        Constructor for Config class

        Args:
            cfg_file (str): config file path
            params (dict): parameters
        """
        self.cfg_file = cfg_file
        self.__dict__.update(params)

    def __getitem__(self, key):
        return self.__dict__[key]


def cfg_parser(cfg_file: str) -> dict:
    """
    This functions reads an input config file and instantiates objects of Config type.

    Args:
        cfg_file (string): path to config file
    Returns:
        exp_cfg (dict) : dictionary of config file
    """
    cfg = json.load(open(root_dir + cfg_file))

    exp_cfg = {
        "experiment": Config(cfg_file, cfg['experiment']),
        "augmentations": Config(cfg_file, cfg['augmentations'])
    }

    return exp_cfg
