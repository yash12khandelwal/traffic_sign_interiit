import torch.nn as nn
import torch.nn.functional as F

class DKS(nn.Module):
    """ Model Class for DKS Net
    """
    def __init__(self, args):
        """ Init function for defining layers and params

        Args:
            args (TrainOptions): TrainOptions class (refer options/train_options.py)
            Required params from args:
                num_classes (int): Final classes for last output layer
        """

        super(DKS, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3))
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.drop = nn.Dropout(p=0.2)

        self.flat1 = nn.Linear(128, 512)
        self.flat2 = nn.Linear(512, args.num_classes)

    def forward(self, x):
        """ Forward pass for the model

        Args:
            x (TorchTensor): TorchTensor of shape (N, 3, 48, 48)
            N: Batchsize

        Returns:
            TorchTensor: TorchTensor of shape (N, args.num_classes)
        """

        x = F.relu(self.maxpool1(self.conv1(x)))
        x = F.relu(self.maxpool2(self.conv2(x)))
        x = F.relu(self.maxpool3(self.conv3(x)))
        x = F.relu(self.maxpool4(self.conv4(x)))

        x = x.view(-1, 128)
        x = self.drop(x)
        x = F.relu(self.flat1(x))
        x = (self.flat2(x))

        return x