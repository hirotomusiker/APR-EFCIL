"""
Adapted from:
https://github.com/LAMDA-CL/PyCIL/blob/master/utils/data_manager.py
https://github.com/LAMDA-CL/PyCIL/blob/master/utils/data.py
"""
from logging import Logger
from pathlib import Path

import numpy as np
from omegaconf.dictconfig import DictConfig
from torchvision import datasets
from torchvision import transforms as tf

from .base_dataset import BaseDataset
from .reproducible_transforms import RepCIFAR10Policy
from .reproducible_transforms import RepCompose
from .reproducible_transforms import RepImageNetPolicy
from .reproducible_transforms import RepNormalize
from .reproducible_transforms import RepRandomCrop
from .reproducible_transforms import RepRandomHorizontalFlip
from .reproducible_transforms import RepToTensor


class TinyImageNet(BaseDataset):
    def __init__(
        self,
        cfg: DictConfig,
        shuffle: bool = True,
        seed: int = 0,
        logger: Logger = None,
    ):
        """TinyImageNet dataset for incremental learning.

        Args:
            cfg (DictConfig): Config dict.
            shuffle (bool, optional): Whether to shuffle classes.
                Defaults to True.
            seed (int, optional): Random seed for class shuffling.
                Defaults to 0.
            logger (Logger, optional): Logger object. Defaults to None.
        """
        self.cfg = cfg
        train_dir = "./data/tiny-imagenet-200/train/"
        val_dir = "./data/tiny-imagenet-200/val/"
        train_dataset = datasets.ImageFolder(train_dir)
        self.train_data = np.array([d[0] for d in train_dataset.imgs])
        train_targets = [d[1] for d in train_dataset.imgs]
        self.num_classes = len(np.unique(train_dataset.targets))
        self.class_order = [i for i in range(self.num_classes)]
        class_names = [x.name for x in Path(train_dir).iterdir() if x.is_dir()]
        class_names.sort()
        self.means = (0.485, 0.456, 0.406)
        self.stds = (0.229, 0.224, 0.225)

        # val
        with open(str(Path(val_dir) / "val_annotations.txt"), "r") as f:
            annos = f.readlines()
        self.test_data = np.array(
            [str(Path(val_dir) / "images" / a.split("\t")[0]) for a in annos]
        )
        test_targets = [class_names.index(a.split("\t")[1]) for a in annos]

        if shuffle:
            np.random.seed(seed)
            self.class_order = np.random.permutation(self.num_classes).tolist()
        self.train_targets = self._map_cls(np.array(train_targets))
        self.test_targets = self._map_cls(np.array(test_targets))
        self.as_paths = True
        self.logger = logger
        self.logger.info(
            f"{len(self.train_data)} train and {len(self.test_data)} "
            "test data have been obtained."
        )
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
        else:
            transforms.append(tf.Resize(64))
            transforms.append(tf.CenterCrop(56))
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
        if cfg.cifar10policy:
            transforms.append(RepCIFAR10Policy())
        if cfg.imagenetpolicy:
            transforms.append(RepImageNetPolicy())
        transforms.append(RepToTensor())
        transforms.append(RepNormalize(mean=self.means, std=self.stds))
        return RepCompose(transforms)
