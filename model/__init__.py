from model.dks import DKS
from model.micronet import MicroNet
import torch.optim as optim

def CreateModel(args):
    """ Initialise model and optimiser

    Note:
    When adding a new file in models add a condition below with the params

    Args:
        args (TrainOptions): Training/Testing arguments (refer options/train_options.py)

    Raises:
        ValueError: If the model key provided in cmd arguments doesn't exist below

    Returns:
        tuple: Model and Optimiser
    """

    device = args.device

    if args.model == 'dks':
        model = DKS(args).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    elif args.model == 'micronet':
        model = MicroNet(args).cuda()
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    
    else:
        raise ValueError('The model must be dks/micronet')

    return model, optimizer