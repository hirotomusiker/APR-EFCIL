"""
Adapted from:
https://github.com/dipamgoswami/FeCAM/blob/main/utils/inc_net.py#L179
"""
import torch
from torch import nn

from .backbones.build_backbone import get_backbone
from .layers.linears import CosineLinear
from .layers.linears import SplitCosineLinear


class SplitCosNet(nn.Module):
    def __init__(
        self,
        net_type: str,
        for_insubset: bool = False,
    ):
        """Network with split cosine classifier.

        Args:
            net_type (str): Extractor name, e.g. "resnet18"
            for_insubset (bool, optional): Whether to use
                FeCAM-style stem strides for ImageNet-Subset.
                Defaults to False.
        """
        super(SplitCosNet, self).__init__()
        self.net = get_backbone(net_type, for_insubset)
        self.fc = None
        self.n_classes = 0
        self.nb_proxy = 1

    def update_fc(self, n_classes: int, task_idx: int):
        """Update FC for the new task

        Args:
            n_classes (int): Number of total classes in the current task
            task_idx (int): Task index (stats from zero)
        """
        fc = self.generate_fc(self.net.out_dim, n_classes)
        if self.fc is not None:
            if task_idx == 1:
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[
                    :prev_out_features1
                ] = self.fc.fc1.weight.data
                fc.fc1.weight.data[
                    prev_out_features1:
                ] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim: int, out_dim: int) -> CosineLinear:
        """Generate a new FC layer.
        Use CosineLinear for the initial task and
        SplitCosineLinear for the rest.

        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension
            (corresponds to number of total classes)

        Returns:
            CosineLinear: Cosine-linear FC layer.
        """
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            fc = SplitCosineLinear(
                in_dim, prev_out_features, out_dim - prev_out_features
            )
        return fc

    def freeze(self):
        """Freeze all the learnable parameters.
        Used for the old-task net.

        Returns:
            nn.Module: Frozen model.
        """
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def forward(self, x: torch.Tensor):
        """forward function.

        Args:
            x (torch.Tensor): input image data.

        Returns:
            dict: output of the network that has
                features: extracted features, shape (B, D),
                logits: FC outputs, shape (B, C),
                where B is batch size, D feature dimension,
                and C number of classes.
        """
        x = self.net(x)
        out = self.fc(x["features"])
        x["logits"] = out
        return x
