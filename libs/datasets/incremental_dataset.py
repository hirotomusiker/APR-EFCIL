"""
Adapted from:
https://github.com/LAMDA-CL/PyCIL/blob/master/utils/data_manager.py
https://github.com/LAMDA-CL/PyCIL/blob/master/utils/data.py
"""
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as tf

from .reproducible_transforms import PARAM_KEYS


class IncrementalDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        transforms: tf.Compose,
        ptransforms: tf.Compose = None,
        as_paths: bool = False,
        mode: str = "train",
    ):
        """Subset of the dataset (e.g. CIFAR100) to be used
        for a class-incremental task.

        Args:
            images (np.ndarray): The WHOLE image set of the source dataset.
            labels (np.ndarray): The WHOLE label set of the source dataset.
            indices (np.ndarray): Task-specific indices for images and labels.
            transforms (tf.Compose): Data transform pipeline.
            ptransforms (tf.Compose, optional): Deterministic data transform
                pipeline for adversarial replay. Defaults to None.
            as_paths (bool, optional): Type of the image data.
                Defaults to False.
            mode (str, optional): Dataset mode, one of ['train', 'test',
                'replay', 'apr', 'adcapr']. Defaults to 'train'.
        """
        self.images = images
        self.labels = labels
        self.indices = indices
        self.transforms = transforms
        self.pseudo_transforms = ptransforms
        self.as_paths = as_paths
        self.mode = mode
        self.rep_params = None
        self.rep_num_data_per_class = None

    def _init_params(self) -> dict[str, float]:
        """Initialize reproducible transform parameters as np.nan.

        Returns:
            dict[str, float]: Initial transform params.
        """
        params = {k: np.nan for k in PARAM_KEYS}
        return params

    def register_rep_params(
        self,
        rep_params: dict[str, list],
        data_indices: np.ndarray | None = None,
        num_data_per_class: int | None = None,
    ):
        """Register the random transform parameters
        to reproduce them afterwards.

        Args:
            rep_params (dict[str, int | float]): Transform parameters.

            data_indices (np.ndarray | None, optional): Selected indices
                of the new task data. Defaults to None.
            num_data_per_class (int | None, optional): Number of pseudo-replay
                samples per class. Pseudo labels are calculated using this
                value. Defaults to None.
        """
        self.rep_params = rep_params
        if data_indices is not None and num_data_per_class is not None:
            self.rep_num_data_per_class = num_data_per_class
            self.indices = self.indices[data_indices]

    def __len__(self):
        return len(self.indices)

    def check_old_data(self, known_classes: int):
        """Make sure no old-task data are involved.

        Args:
            known_classes (int): Number of known (old) classes.
        """
        min_label = np.min(self.labels[self.indices])
        assert (
            min_label >= known_classes
        ), f"Minimum of the labels is {min_label}: old task data found!"

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int, dict | None]:
        """Load image and label and apply transforms.
        When `self.mode` is one of "apr" and "adcapr":
        - before self.register_rep_params: obtain all random
            params used in the transforms.
        - after self.register_rep_params: apply transforms
            using registered params.
        Args:
            idx (int): Random index.

        Returns:
            tuple[np.ndarray, int, dict]:
                img (np.ndarray): Transformed image data.
                label (int): Classification label.
                dst_params (dict | None): Returns param dict
                    when `self.mode` is one of "apr" and "adcapr".
                    example: `{'RandomCrop_i': 2, ...
                               'RandomResizedCrop_i': nan,`

        """
        global_idx = self.indices[idx]
        # load image data
        if self.as_paths:
            with open(self.images[global_idx], "rb") as f:
                img = Image.open(f).convert("RGB")
        else:
            img = Image.fromarray(self.images[global_idx])
        if self.mode in ["apr", "adcapr"]:
            if self.rep_params is not None:
                idx_params = {k: v[idx] for k, v in self.rep_params.items()}
            else:
                idx_params = self._init_params()
            img, dst_params = self.transforms(img, idx_params)
            if self.rep_num_data_per_class is not None:
                label = idx // self.rep_num_data_per_class
            else:
                label = self.labels[global_idx]
            return img, label, dst_params
        else:
            img = self.transforms(img)
            label = self.labels[global_idx]
            return img, label, dict()
