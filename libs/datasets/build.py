from logging import Logger

from omegaconf.dictconfig import DictConfig

from libs.datasets.base_dataset import BaseDataset
from libs.datasets.icifar100 import iCIFAR100
from libs.datasets.imagenetsubset import ImageNetSubset
from libs.datasets.tinyimagenet import TinyImageNet


def build_dataset(
    cfg: DictConfig, logger: Logger, clsseed: int
) -> BaseDataset:
    """Create a dataset.

    Args:
        cfg (DictConfig): Omegaconf object.
        logger (Logger): Logger object.
        clsseed (int): Random seed for shuffling the class order.

    Raises:
        NotImplementedError: The specified dataset
            is not implemented.

    Returns:
        BaseDataset: Dataset object.
    """
    dataset_name = cfg.data.type
    shuffle_cls = cfg.incremental.shuffle_cls
    if dataset_name == "iCIFAR100":
        dataset = iCIFAR100(
            cfg,
            logger=logger,
            seed=clsseed,
            shuffle=shuffle_cls,
        )
    elif dataset_name == "TinyImageNet":
        dataset = TinyImageNet(
            cfg,
            logger=logger,
            seed=clsseed,
            shuffle=shuffle_cls,
        )
    elif dataset_name == "ImageNetSubset":
        dataset = ImageNetSubset(
            cfg,
            logger=logger,
            seed=clsseed,
            shuffle=shuffle_cls,
        )
    else:
        raise NotImplementedError()
    return dataset
