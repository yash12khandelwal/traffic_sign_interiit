import json

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


def cfg_parser(cfg_file: str) -> dict:
    """
    This functions reads an input config file and instantiates objects of
    Config types.
    args:
        cfg_file (string): path to cfg file
    returns:
        exp_cfg (Config)
    """
    cfg = json.load(open(cfg_file))

    exp_cfg = Config(cfg_file, cfg)

    return exp_cfg

