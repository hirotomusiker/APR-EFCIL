"""
Adapted from:
https://github.com/LAMDA-CL/PyCIL/blob/master/convs/cifar_resnet.py
We use FeCAM version of stem strides for ImageNet-Subset:
https://github.com/dipamgoswami/FeCAM/blob/main/convs/resnet.py#L152
Only necessary modules for APR are included in this script.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ResNetBasicBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        ch: int,
        stride: int = 1,
        downsample: nn.Sequential | None = None,
    ):
        """ResNet block.

        Args:
            in_ch (int): Number of input channels.
            ch (int): Number of hidden / output channels.
            stride (int, optional): Spatial stride size. Defaults to 1.
            downsample (nn.Sequential | None, optional): Downsampling layers.
                Defaults to None.
        """
        super(ResNetBasicBlock, self).__init__()
        self.conv_a = nn.Conv2d(
            in_ch, ch, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn_a = nn.BatchNorm2d(ch)
        self.conv_b = nn.Conv2d(
            ch, ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn_b = nn.BatchNorm2d(ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        h = self.bn_a(self.conv_a(x))
        h = F.relu(h, inplace=True)
        h = self.bn_b(self.conv_b(h))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(identity + h, inplace=True)


class CifarResNet(nn.Module):
    def __init__(
        self,
        inplanes: int,
        num_blocks_list: list[int],
        ch_list: list[int],
        stride_list: list[int],
        for_insubset: bool = False,
    ):
        """ResNet for CIFAR dataset.

        Args:
            inplanes (int): Stem output dimension.
            num_blocks_list (list[int]): Number of blocks at each stage.
            ch_list (list[int]): Dimension size at each stage.
            stride_list (list[int]): Downsampling stride at each stage.
            for_insubset (bool, optional): _description_. Defaults to False.
        """
        super(CifarResNet, self).__init__()
        self._for_insubset = for_insubset
        self.inplanes = inplanes
        if for_insubset:
            self.stem = nn.Sequential(
                nn.Conv2d(
                    3,
                    self.inplanes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(self.inplanes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        else:
            self.stem = nn.Conv2d(
                3, inplanes, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn_1 = nn.BatchNorm2d(inplanes)

        stages = []
        for num_blocks, ch, stride in zip(
            num_blocks_list, ch_list, stride_list
        ):
            stages.append(
                self._make_stage(
                    ResNetBasicBlock, ch, num_blocks, stride=stride
                )
            )
        self.stages = nn.Sequential(*stages)
        self.out_dim = ch
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _make_stage(
        self, block: nn.Module, planes: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        """Build resnet blocks for one stage.

        Args:
            block (type): ResNet block class.
            planes (int): Feature dimension in this stage.
            num_blocks (int): Number of ResNet blocks.
            stride (int): Feature downsampling stride.

        Returns:
            nn.Sequential: Composed layers.
        """
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
            )
        else:
            downsample = None

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Apply network to the input tensor.

        Args:
            x (torch.Tensor): Input image data.

        Returns:
            dict[str, list | torch.Tensor]:
                fmaps (list[torch.Tensor]): Feature map at each
                    stage, shape (B, dim_stage, H, W).
                features (torch.Tensor): Final feature after
                    spatial pooling, shape (B, dim_final).
        """
        x = self.stem(x)  # [B, 16, 32, 32]
        if not self._for_insubset:
            x = F.relu(self.bn_1(x), inplace=True)
        fmaps = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            fmaps.append(x)
        pooled = self.avgpool(x)
        features = pooled.view(pooled.size(0), -1)
        return {"fmaps": fmaps, "features": features}


def resnet18(for_insubset: bool = False) -> nn.Module:
    """ResNet18 feature extractor network.

    Args:
        for_insubset (bool, optional):
            Whether to use FeCAM-style stem strides for
            ImageNet-Subset. Defaults to False.

    Returns:
        nn.Module: ResNet18 network.
    """
    num_blocks_list = [2, 2, 2, 2]
    ch_list = [64, 128, 256, 512]
    stride_list = [1, 2, 2, 2]
    return CifarResNet(
        inplanes=64,
        num_blocks_list=num_blocks_list,
        ch_list=ch_list,
        stride_list=stride_list,
        for_insubset=for_insubset,
    )
