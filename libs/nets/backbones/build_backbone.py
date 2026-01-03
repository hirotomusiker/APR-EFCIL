from .cifar_resnet import resnet18


def get_backbone(net_type: str, for_insubset: bool = False):
    """Select backbone class.

    Args:
        net_type (str): Network name, e.g. "resnet18".
        for_insubset (bool, optional):
            Whether to use FeCAM-style stem strides for
            ImageNet-Subset. Defaults to False.

    Raises:
        NotImplementedError: The specified net type
           is not implemented.

    Returns:
        torch.nn.Module: Backbone network object.
    """
    net_type = net_type.lower()
    if net_type == "resnet18":
        return resnet18(for_insubset=for_insubset)
    else:
        raise NotImplementedError(f"{net_type} is not implemented!")
