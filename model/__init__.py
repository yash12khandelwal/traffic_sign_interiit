from model.dks import DKS
from model.micronet import MicroNet
import torch.optim as optim

def CreateModel(args):
    """ Initialise model and optimiser and schedular

    Note:
    When adding a new file in models add a condition below with the params

    Args:
        args (TrainOptions): Training/Testing arguments (refer options/train_options.py)

    Raises:
        ValueError: If the model key provided in cmd arguments doesn't exist below

    Returns:
        tuple: Model and Optimiser and Schedular
    """

    device = args.device

    if args.model == 'dks':
        model = DKS(args).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=0.9)

    elif args.model == 'micronet':
        model = MicroNet(args).cuda()
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=0.9)
    
    else:
        raise ValueError('The model must be dks/micronet')

    return model, optimizer, scheduler