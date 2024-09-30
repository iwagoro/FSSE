import torch.nn as nn
import torch
from .CConv import CConv2d
from .CBatchNorm import CBatchNorm2d


class Encoder(nn.Module):
    def __init__(self, filter_size=(7, 5), stride_size=(2, 2), in_channels=1, out_channels=45, padding=(0, 0)):
        super().__init__()

        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        self.cconv = CConv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.filter_size,
            stride=self.stride_size,
            padding=self.padding,
        )

        self.cbn = CBatchNorm2d(num_features=self.out_channels)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):

        conved = self.cconv(x)
        normed = self.cbn(conved)
        acted = self.leaky_relu(normed)

        return acted
