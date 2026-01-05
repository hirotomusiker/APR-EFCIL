"""
Adapted from:
https://github.com/LAMDA-CL/PyCIL/blob/master/utils/data_manager.py
https://github.com/LAMDA-CL/PyCIL/blob/master/utils/data.py
"""
from logging import Logger

import numpy as np
from omegaconf.dictconfig import DictConfig
from torchvision import datasets
from torchvision import transforms as tf

from .autoaugment import CIFAR10Policy
from .base_dataset import BaseDataset
from .reproducible_transforms import RepCIFAR10Policy
from .reproducible_transforms import RepColorJitter
from .reproducible_transforms import RepCompose
from .reproducible_transforms import RepNormalize
from .reproducible_transforms import RepRandomCrop
from .reproducible_transforms import RepRandomHorizontalFlip
from .reproducible_transforms import RepToTensor


class iCIFAR100(BaseDataset):
    def __init__(
        self,
        cfg: DictConfig,
        shuffle: bool = True,
        seed: int = 0,
        logger: Logger = None,
    ):
        """CIFAR100 dataset for incremental learning.

        Args:
            cfg (DictConfig): Config dict.
            shuffle (bool, optional): Whether to shuffle classes.
                Defaults to True.
            seed (int, optional): Random seed for class shuffling.
                Defaults to 0.
            logger (Logger, optional): Logger object. Defaults to None.
        """
        train_dataset = datasets.cifar.CIFAR100(
            "./data", train=True, download=True
        )
        test_dataset = datasets.cifar.CIFAR100(
            "./data", train=False, download=True
        )
        self.cfg = cfg
        self.train_data = train_dataset.data
        self.test_data = test_dataset.data
        self.logger = logger
        self.logger.info(
            f"{len(self.train_data)} train and {len(self.test_data)} "
            "test data have been obtained."
        )
        self.means = (0.5071, 0.4867, 0.4408)
        self.stds = (0.2675, 0.2565, 0.2761)
        self.num_classes = len(np.unique(train_dataset.targets))
        self.class_order = [i for i in range(self.num_classes)]
        self.as_paths = False

        if shuffle:
            np.random.seed(seed)
            self.class_order = np.random.permutation(self.num_classes).tolist()
        self.train_targets = self._map_cls(np.array(train_dataset.targets))
        self.test_targets = self._map_cls(np.array(test_dataset.targets))
        self.logger.info(f"class order = {self.class_order}")

    def _build_transforms(self, mode: str) -> tf.Compose:
        """Build augmentation pipeline.

        Args:
            mode (str): If "train", random transforms are
                composed.

        Returns:
            tf.Compose: Augmentation pipeline.
        """
        transforms = []
        if mode == "train":
            transforms.append(tf.RandomCrop(32, padding=4))
            transforms.append(tf.RandomHorizontalFlip())
            if self.cfg.data.cifar10colorjitter:
                transforms.append(tf.ColorJitter(brightness=63 / 255))
            if self.cfg.data.cifar10policy:
                transforms.append(CIFAR10Policy())
        transforms.append(tf.ToTensor())
        transforms.append(tf.Normalize(mean=self.means, std=self.stds))
        return tf.Compose(transforms)

    def _build_rep_transforms(self, cfg: DictConfig) -> tf.Compose:
        """Build reproducible transforms for APR.

        Args:
            cfg (DictConfig): If "train", random transforms are
                composed.

        Returns:
            tf.Compose: Reproducible augmentation pipeline.
        """
        transforms = []
        transforms.append(RepRandomCrop(32, padding=4))
        transforms.append(RepRandomHorizontalFlip())
        if cfg.colorjitter:
            transforms.append(RepColorJitter(brightness=63 / 255))
        if cfg.cifar10policy:
            transforms.append(RepCIFAR10Policy())
        transforms.append(RepToTensor())
        transforms.append(RepNormalize(mean=self.means, std=self.stds))
        return RepCompose(transforms)
