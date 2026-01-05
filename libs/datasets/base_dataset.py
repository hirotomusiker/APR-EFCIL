"""
Adapted from:
https://github.com/LAMDA-CL/PyCIL/blob/master/utils/data_manager.py
https://github.com/LAMDA-CL/PyCIL/blob/master/utils/data.py
"""
from logging import Logger

import numpy as np
from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset

from .incremental_dataset import IncrementalDataset


class BaseDataset:
    def __init__(
        self,
        cfg: DictConfig,
        shuffle: bool = True,
        seed: int = 0,
        logger: Logger = None,
    ):
        """Meta dataset for incremental learning.
        With `draw_dataset`, an indice array for selecting the
        dataset subset is generated and `IncrementalDataset` is
        created.

        Args:
            cfg (DictConfig): Config dict.
            shuffle (bool, optional): Whether to shuffle classes.
                Defaults to True.
            seed (int, optional): Random seed for class shuffling.
                Defaults to 0.
            logger (Logger, optional): Logger object. Defaults to None.
        """
        self.cfg = cfg
        self.train_data = []
        self.num_classes = 0
        self.class_order = []
        self.test_data = np.array([])
        if shuffle:
            np.random.seed(seed)
            self.class_order = np.random.permutation(self.num_classes).tolist()
        self.train_targets = np.array([])
        self.test_targets = np.array([])
        self.means = (0.0, 0.0, 0.0)
        self.stds = (1.0, 1.0, 1.0)
        self.logger = logger
        self.as_paths = True

    def _map_cls(self, targets: np.ndarray) -> np.ndarray:
        new_order = list(map(lambda x: self.class_order.index(x), targets))
        return np.array(new_order)

    def draw_dataset(
        self, n_known: int, n_new: int, mode: bool = "train"
    ) -> Dataset:
        if mode == "apr":
            transforms = self._build_rep_transforms(self.cfg.apr)
        elif mode == "adcapr":
            transforms = self._build_rep_transforms(self.cfg.adc)
        else:
            transforms = self._build_transforms(mode)
        pseudo_transforms = self._build_transforms(mode)
        if mode in ["train", "replay", "apr", "adcapr"]:
            class_indices = np.arange(n_known, n_known + n_new)
            x, y = self.train_data, self.train_targets
        elif mode == "test":
            class_indices = np.arange(0, n_known)
            x, y = self.test_data, self.test_targets
        else:
            raise ValueError(
                "mode must be one of "
                "['train', 'test', 'replay', 'apr', 'adcapr']."
            )
        indices = np.zeros(
            [
                0,
            ],
            dtype=np.int64,
        )
        for idx in class_indices:
            # indices
            indices = np.concatenate(
                [indices, self._select_data(y, idx, idx + 1)]
            )
        return IncrementalDataset(
            x,
            y,
            indices,
            transforms,
            pseudo_transforms,
            as_paths=self.as_paths,
            mode=mode,
        )

    def _select_data(
        self, y: np.ndarray, low_idx: int, high_idx: int
    ) -> np.ndarray:
        indices = np.where(np.logical_and(y >= low_idx, y < high_idx))[0]
        return indices

    def _build_transforms(self):
        pass

    def _build_rep_transforms(self):
        pass
