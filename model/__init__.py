from model.dks import DKS
from model.micronet import MicroNet
import torch.optim as optim

def CreateModel(args):
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