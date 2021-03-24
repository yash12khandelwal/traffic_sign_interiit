import json
import os

root_dir = "data/traffic_sign_interiit/"

class Config(object):
    """
    Class for all attributes and functionalities related to a particular training run.
    """

    def __init__(self, cfg_file: str, params: dict):
        """
        Constructir for Config class
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
    This functions reads an input config file and instantiates objects of
    Config types.
    args:
        cfg_file (string): path to cfg file
    returns:
        exp_cfg (Config)
    """
    cfg = json.load(open(os.path.join(root_dir, cfg_file)))
    
    # cfg = json.load(open(os.path.join(app.config["DATA_PATH"], cfg_file)))
    # cfg = json.load(open(cfg_file))

    exp_cfg = {
        "experiment": Config(cfg_file, cfg['experiment']),
        "augmentations": Config(cfg_file, cfg['augmentations'])
    }

    return exp_cfg
