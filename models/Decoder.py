import torch.nn as nn
import torch
from .CConv import CConv2d, CConvTranspose2d
from .CBatchNorm import CBatchNorm2d


class Decoder(nn.Module):
    """
    Class of downsample block
    """

    def __init__(
        self,
        filter_size=(7, 5),
        stride_size=(2, 2),
        in_channels=1,
        out_channels=45,
        output_padding=(0, 0),
        padding=(0, 0),
        last_layer=False,
    ):
        super().__init__()

        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_padding = output_padding
        self.padding = padding

        self.last_layer = last_layer

        self.cconvt = CConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.filter_size,
            stride=self.stride_size,
            output_padding=self.output_padding,
            padding=self.padding,
        )

        self.cbn = CBatchNorm2d(num_features=self.out_channels)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):

        conved = self.cconvt(x)

        if not self.last_layer:
            normed = self.cbn(conved)
            output = self.leaky_relu(normed)
        else:
            m_phase = conved / (torch.abs(conved) + 1e-8)
            m_mag = torch.tanh(torch.abs(conved))
            output = m_phase * m_mag

        return output
