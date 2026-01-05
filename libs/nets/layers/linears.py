"""
Adapted from:
https://github.com/dipamgoswami/FeCAM/blob/main/convs/linears.py
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


class SimpleLinear(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bias: bool = True):
        """Linear layer.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            bias (bool, optional): Use bias if True. Defaults to True.
        """
        super(SimpleLinear, self).__init__()
        self.in_ch = in_ch
        self.out_features = out_ch
        self.weight = nn.Parameter(torch.Tensor(out_ch, in_ch))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_ch))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity="linear")
        nn.init.constant_(self.bias, 0)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class CosineLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, sigma: bool = True
    ):
        """Cosine linear layer.

        Args:
            in_features (int): Number of input channels.
            out_features (int): Number of output channels.
            sigma (bool, optional): Whether to use a scaling
                parameter sigma. Defaults to True.
        """
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.Tensor(self.out_features, in_features)
        )
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter("sigma", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        out = F.linear(
            F.normalize(input, p=2, dim=1),
            F.normalize(self.weight, p=2, dim=1),
        )

        if self.sigma is not None:
            out = self.sigma * out
        return out


class SplitCosineLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features1: int,
        out_features2: int,
        sigma: bool = True,
    ):
        """Split cosine linear layer.

        Args:
            in_features (int): Number of input channels.
            out_features1 (int): Number of output channels for fc1,
                typically for old task classes.
            out_features2 (int): Number of output channels for fc2,
                typically for new task classes.
            sigma (bool, optional): Whether to use a scaling
                parameter sigma. Defaults to True.
        """
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features1 + out_features2
        self.fc1 = CosineLinear(in_features, out_features1, False)
        self.fc2 = CosineLinear(in_features, out_features2, False)
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter("sigma", None)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out = torch.cat((out1, out2), dim=1)
        if self.sigma is not None:
            out = self.sigma * out

        return out
