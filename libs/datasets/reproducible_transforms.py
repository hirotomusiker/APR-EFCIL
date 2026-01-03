"""
Adapted from:
https://github.com/pytorch/vision/blob/v0.16.0/torchvision/transforms/transforms.py
https://github.com/LAMDA-CL/PyCIL/blob/master/utils/autoaugment.py
"""
import random

import numpy as np
import PIL
import torch
import torchvision.transforms as tf
import torchvision.transforms.functional as F
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageOps

# Parameter keys for all possible random transforms.

PARAM_KEYS = [
    "RandomCrop_i",  # RepRandomCrop
    "RandomCrop_j",  # RepRandomCrop
    "RandomCrop_h",  # RepRandomCrop
    "RandomCrop_w",  # RepRandomCrop
    "RandomResizedCrop_i",  # RepRandomResizedCrop
    "RandomResizedCrop_j",  # RepRandomResizedCrop
    "RandomResizedCrop_h",  # RepRandomResizedCrop
    "RandomResizedCrop_w",  # RepRandomResizedCrop
    "RandomHorizontalFlip",  # RepRandomHorizontalFlip
    "SubPolicy_do_op1",  # RepSubPolicy
    "SubPolicy_do_op2",  # RepSubPolicy
    "RepCIFAR10Policy_policy_idx",  # RepCIFAR10Policy
    "RepImageNetPolicy_policy_idx",  # RepImageNetPolicy
    "RepShearX",  # RepShearX
    "RepShearY",  # RepShearY
    "RepTranslateX",  # RepTranslateX
    "RepTranslateY",  # RepTranslateX
    "RepRotate",  # RepRotate
    "RepColor",  # RepColor
    "RepContrast",  # RepContrast
    "RepSharpness",  # RepSharpness
    "RepBrightness",  # RepBrightness
    "RepColorJitter_fn_idx_0",  # RepColorJitter
    "RepColorJitter_fn_idx_1",  # RepColorJitter
    "RepColorJitter_fn_idx_2",  # RepColorJitter
    "RepColorJitter_fn_idx_3",  # RepColorJitter
    "RepColorJitter_brightness_factor",  # RepColorJitter
    "RepColorJitter_contrast_factor",  # RepColorJitter
    "RepColorJitter_saturation_factor",  # RepColorJitter
    "RepColorJitter_hue_factor",  # RepColorJitter
]


class RepCompose(tf.Compose):
    def __call__(
        self, img: PIL.Image.Image, params: dict[str, int | float | bool]
    ) -> tuple[torch.Tensor, dict[str, int | float | bool]]:
        """Composed transforms.

        Args:
            img (PIL.Image.Image): Image object.
            params (dict[str, int | float | bool]): Transform parameters.
                Values are np.nan for irrelevant or unseen transforms
                and int or bool for seen transforms.

        Returns:
            tuple[torch.Tensor, dict[str, int | float | bool]: Transform
                parameters. If the new transforms are seen,
                the used parameters are registered.
        """
        for t in self.transforms:
            img, params = t(img, params)
        return img, params


class RepRandomCrop(tf.RandomCrop):
    def forward(
        self, img: PIL.Image.Image, params: dict[str, int | float | bool]
    ) -> tuple[PIL.Image.Image, dict[str, int | float | bool]]:
        """RandomCrop with reproducible parameter recording.

        Args:
            img (PIL.Image.Image): Image object.
            params (dict[str, int | float | bool]): Transform parameters.
                Values are np.nan for irrelevant or unseen transforms
                and int or bool for seen transforms.

        Returns:
            tuple[PIL.Image.Image, dict[str, int | float | bool]: Transform
                parameters. If the new transforms are seen,
                the used parameters are registered.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        _, height, width = F.get_dimensions(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        if np.isnan(params["RandomCrop_i"]):
            i, j, h, w = self.get_params(img, self.size)
            params["RandomCrop_i"] = i
            params["RandomCrop_j"] = j
            params["RandomCrop_h"] = h
            params["RandomCrop_w"] = w

        i = params["RandomCrop_i"]
        j = params["RandomCrop_j"]
        h = params["RandomCrop_h"]
        w = params["RandomCrop_w"]

        return F.crop(img, i, j, h, w), params


class RepRandomResizedCrop(tf.RandomResizedCrop):
    def forward(
        self, img: PIL.Image.Image, params: dict[str, int | float | bool]
    ) -> tuple[PIL.Image.Image, dict[str, int | float | bool]]:
        """RandomResizedCrop with reproducible parameter recording.

        Args:
            img (PIL.Image.Image): Image object.
            params (dict[str, int | float | bool]): Transform parameters.
                Values are np.nan for irrelevant or unseen transforms
                and int or bool for seen transforms.

        Returns:
            tuple[PIL.Image.Image, dict[str, int | float | bool]: Transform
                parameters. If the new transforms are seen,
                the used parameters are registered.
        """
        if np.isnan(params["RandomResizedCrop_i"]):
            i, j, h, w = self.get_params(img, self.scale, self.ratio)
            params["RandomResizedCrop_i"] = i
            params["RandomResizedCrop_j"] = j
            params["RandomResizedCrop_h"] = h
            params["RandomResizedCrop_w"] = w

        i = params["RandomResizedCrop_i"]
        j = params["RandomResizedCrop_j"]
        h = params["RandomResizedCrop_h"]
        w = params["RandomResizedCrop_w"]

        return (
            F.resized_crop(
                img,
                i,
                j,
                h,
                w,
                self.size,
                self.interpolation,
                antialias=self.antialias,
            ),
            params,
        )


class RepRandomHorizontalFlip(tf.RandomHorizontalFlip):
    def forward(
        self, img: PIL.Image.Image, params: dict[str, int | float | bool]
    ) -> tuple[PIL.Image.Image, dict[str, int | float | bool]]:
        """RandomHorizontalFlip with reproducible parameter recording.

        Args:
            img (PIL.Image.Image): Image object.
            params (dict[str, int | float | bool]): Transform parameters.
                Values are np.nan for irrelevant or unseen transforms
                and int or bool for seen transforms.
        Returns:
            tuple[PIL.Image.Image, dict[str, int | float | bool]: Transform
                parameters. If the new transforms are seen,
                the used parameters are registered.
        """
        if np.isnan(params["RandomHorizontalFlip"]):
            do_flip = torch.rand(1)[0] < self.p
            params["RandomHorizontalFlip"] = do_flip.item()
        do_flip = params["RandomHorizontalFlip"]
        if do_flip:
            img = F.hflip(img)
        return img, params


class RepColorJitter(tf.ColorJitter):
    def forward(
        self, img: PIL.Image.Image, params: dict[str, int | float | bool]
    ) -> tuple[PIL.Image.Image, dict[str, int | float | bool]]:
        """RandomColorJitter with reproducible parameter recording.

        Args:
            img (PIL.Image.Image): Image object.
            params (dict[str, int | float | bool]): Transform parameters.
                Values are np.nan for irrelevant or unseen transforms
                and int or bool for seen transforms.

        Returns:
            tuple[PIL.Image.Image, dict[str, int | float | bool]: Transform
                parameters. If the new transforms are seen,
                the used parameters are registered.
        """
        if np.isnan(params["RepColorJitter_fn_idx_0"]):
            # Shuffle fn_idx via `torch.randperm(4)`
            # Draw brightness, contrast, saturation, hue factors
            # from uniform distributions.
            # Note that this transform is used in CIFAR100 with
            # only brightness factor. Thus only brightness_factor
            # matters actually, whatever the order of fn_idx is.
            (
                fn_idx,
                brightness_factor,
                contrast_factor,
                saturation_factor,
                hue_factor,
            ) = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )
            brightness_factor = (
                np.nan if brightness_factor is None else brightness_factor
            )
            contrast_factor = (
                np.nan if contrast_factor is None else contrast_factor
            )
            saturation_factor = (
                np.nan if saturation_factor is None else saturation_factor
            )
            hue_factor = np.nan if hue_factor is None else hue_factor
            for i, fn_id in enumerate(fn_idx):
                params[f"RepColorJitter_fn_idx_{i}"] = fn_idx[i].item()
            params["RepColorJitter_brightness_factor"] = brightness_factor
            params["RepColorJitter_contrast_factor"] = contrast_factor
            params["RepColorJitter_saturation_factor"] = saturation_factor
            params["RepColorJitter_hue_factor"] = hue_factor
        else:
            # Read the experienced random variables
            fn_idx = [params[f"RepColorJitter_fn_idx_{i}"] for i in range(4)]
            brightness_factor = params["RepColorJitter_brightness_factor"]
            contrast_factor = params["RepColorJitter_contrast_factor"]
            saturation_factor = params["RepColorJitter_saturation_factor"]
            hue_factor = params["RepColorJitter_hue_factor"]

        # Execute transforms
        for fn_id in fn_idx:
            if fn_id == 0 and not np.isnan(brightness_factor):
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and not np.isnan(contrast_factor):
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and not np.isnan(saturation_factor):
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and not np.isnan(hue_factor):
                img = F.adjust_hue(img, hue_factor)

        return img, params


# AutoAugment part


class RepSubPolicy:
    def __init__(
        self,
        p1: float,
        operation1: str,
        magnitude_idx1: int,
        p2: float,
        operation2: str,
        magnitude_idx2: int,
        fillcolor=(128, 128, 128),
    ):
        """Sub policy of CIFAR100 or ImageNet Policy.
        Build two operations with pi (chance to execute),
        operationi (name) and magnitude_idxi (magnitude
        index) of the operation, where i in (1, 2).

        Args:
            p1 (float): Probability to do operation1.
            operation1 (str): Name of operation1.
            magnitude_idx1 (int): Magnitude index of operation1.
            p1 (float): Probability to do operation2.
            operation1 (str): Name of operation2.
            magnitude_idx1 (int): Magnitude index of operation2.
            fillcolor (tuple, optional): Padding value for shear
                and translate. Defaults to (128, 128, 128).
        """
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10,
        }

        func = {
            "shearX": RepShearX(fillcolor=fillcolor),
            "shearY": RepShearY(fillcolor=fillcolor),
            "translateX": RepTranslateX(fillcolor=fillcolor),
            "translateY": RepTranslateY(fillcolor=fillcolor),
            "rotate": RepRotate(),
            "color": RepColor(),
            "posterize": RepPosterize(),
            "solarize": RepSolarize(),
            "contrast": RepContrast(),
            "sharpness": RepSharpness(),
            "brightness": RepBrightness(),
            "autocontrast": RepAutoContrast(),
            "equalize": RepEqualize(),
            "invert": RepInvert(),
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(
        self, img: PIL.Image.Image, params: dict[str, int | float | bool]
    ) -> tuple[PIL.Image.Image, dict[str, int | float | bool]]:
        """Execute two operations with random occurrence probabilities.

        Args:
            img (PIL.Image.Image): Input image.
            params (dict[str, int | float | bool]): Transform parameters.
                Values are np.nan for irrelevant or unseen transforms
                and int or bool for seen transforms.
        Returns:
            tuple[PIL.Image.Image, dict[str, int | float | bool]: Transform
                parameters. If the new transforms are seen,
                the used parameters are registered.
        """
        if np.isnan(params["SubPolicy_do_op1"]):
            do_op1 = random.random() < self.p1
            do_op2 = random.random() < self.p2
            params["SubPolicy_do_op1"] = do_op1
            params["SubPolicy_do_op2"] = do_op2
        else:
            do_op1 = params["SubPolicy_do_op1"]
            do_op2 = params["SubPolicy_do_op2"]

        if do_op1:
            img, params = self.operation1(img, self.magnitude1, params)
        if do_op2:
            img, params = self.operation2(img, self.magnitude2, params)
        return img, params


class RepCIFAR10Policy:
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            RepSubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            RepSubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            RepSubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            RepSubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            RepSubPolicy(
                0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor
            ),
            RepSubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            RepSubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            RepSubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            RepSubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            RepSubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),
            RepSubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            RepSubPolicy(
                0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor
            ),
            RepSubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            RepSubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            RepSubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),
            RepSubPolicy(
                0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor
            ),
            RepSubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
            RepSubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            RepSubPolicy(
                0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor
            ),
            RepSubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),
            RepSubPolicy(
                0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor
            ),
            RepSubPolicy(
                0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor
            ),
            RepSubPolicy(
                0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor
            ),
            RepSubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            RepSubPolicy(
                0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor
            ),
        ]

    def __call__(
        self, img: PIL.Image.Image, params: dict[str, int | float | bool]
    ) -> tuple[PIL.Image.Image, dict[str, int | float | bool]]:
        """Execute a Autoaug CIFAR10 policy, containing two operations
        with random occurrence probabilities. The policy index is
        recorded for reproducible transforms.

        Args:
            img (PIL.Image.Image): Input image.
            params (dict[str, int | float | bool]): Transform parameters.
                Values are np.nan for irrelevant or unseen transforms
                and int or bool for seen transforms.
        Returns:
            tuple[PIL.Image.Image, dict[str, int | float | bool]: Transform
                parameters. If the new transforms are seen,
                the used parameters are registered.
        """
        if np.isnan(params["RepCIFAR10Policy_policy_idx"]):
            policy_idx = random.randint(0, len(self.policies) - 1)
            params["RepCIFAR10Policy_policy_idx"] = policy_idx
        else:
            policy_idx = params["RepCIFAR10Policy_policy_idx"]
        return self.policies[policy_idx](img, params)


class RepImageNetPolicy(object):
    """Randomly choose one of the best 24 Sub-policies on ImageNet.

    Example:
    >>> policy = ImageNetPolicy()
    >>> transformed = policy(image)

    Example as a PyTorch Transform:
    >>> transform = transforms.Compose([
    >>>     transforms.Resize(256),
    >>>     ImageNetPolicy(),
    >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            RepSubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            RepSubPolicy(
                0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor
            ),
            RepSubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            RepSubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            RepSubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            RepSubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            RepSubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            RepSubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            RepSubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            RepSubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),
            RepSubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            RepSubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            RepSubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            RepSubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            RepSubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            RepSubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            RepSubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            RepSubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            RepSubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            RepSubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),
            RepSubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            RepSubPolicy(
                0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor
            ),
            RepSubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            RepSubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            RepSubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
        ]

    def __call__(
        self, img: PIL.Image.Image, params: dict[str, int | float | bool]
    ) -> tuple[PIL.Image.Image, dict[str, int | float | bool]]:
        """Execute a Autoaug ImageNet policy, containing two operations
        with random occurrence probabilities. The policy index is
        recorded for reproducible transforms.

        Args:
            img (PIL.Image.Image): Input image.
            params (dict[str, int | float | bool]): Transform parameters.
                Values are np.nan for irrelevant or unseen transforms
                and int or bool for seen transforms.
        Returns:
            tuple[PIL.Image.Image, dict[str, int | float | bool]: Transform
                parameters. If the new transforms are seen,
                the used parameters are registered.
        """
        if np.isnan(params["RepImageNetPolicy_policy_idx"]):
            policy_idx = random.randint(0, len(self.policies) - 1)
            params["RepImageNetPolicy_policy_idx"] = policy_idx
        else:
            policy_idx = params["RepImageNetPolicy_policy_idx"]
        return self.policies[policy_idx](img, params)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"


class RepShearX(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(
        self,
        x: PIL.Image.Image,
        magnitude: float,
        params: dict[str, int | float | bool],
    ) -> tuple[PIL.Image.Image, dict[str, int | float | bool]]:
        """ShearX with reproducible parameter recording.
        Args:
            img (PIL.Image.Image): Input image.
            magnitude (float): random magnitude of the shear transform.
            params (dict[str, int | float | bool]): Transform parameters.
                Values are np.nan for irrelevant or unseen transforms
                and int or bool for seen transforms.
        Returns:
            tuple[PIL.Image.Image, dict[str, int | float | bool]: Transform
                parameters. If the new transforms are seen,
                the used parameters are registered.
        """
        if np.isnan(params["RepShearX"]):
            rand = random.choice([-1, 1])
            params["RepShearX"] = rand
        else:
            rand = params["RepShearX"]
        return (
            x.transform(
                x.size,
                Image.AFFINE,
                (1, magnitude * rand, 0, 0, 1, 0),
                Image.BICUBIC,
                fillcolor=self.fillcolor,
            ),
            params,
        )


class RepShearY(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(
        self,
        x: PIL.Image.Image,
        magnitude: float,
        params: dict[str, int | float | bool],
    ) -> tuple[PIL.Image.Image, dict[str, int | float | bool]]:
        """ShearY with reproducible parameter recording.
        Args:
            img (PIL.Image.Image): Input image.
            magnitude (float): random magnitude of the shear transform.
            params (dict[str, int | float | bool]): Transform parameters.
                Values are np.nan for irrelevant or unseen transforms
                and int or bool for seen transforms.
        Returns:
            tuple[PIL.Image.Image, dict[str, int | float | bool]: Transform
                parameters. If the new transforms are seen,
                the used parameters are registered.
        """
        if np.isnan(params["RepShearY"]):
            rand = random.choice([-1, 1])
            params["RepShearY"] = rand
        else:
            rand = params["RepShearY"]
        return (
            x.transform(
                x.size,
                Image.AFFINE,
                (1, 0, 0, magnitude * rand, 1, 0),
                Image.BICUBIC,
                fillcolor=self.fillcolor,
            ),
            params,
        )


class RepTranslateX(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(
        self,
        x: PIL.Image.Image,
        magnitude: float,
        params: dict[str, int | float | bool],
    ) -> tuple[PIL.Image.Image, dict[str, int | float | bool]]:
        """TranslateX with reproducible parameter recording.
        Args:
            img (PIL.Image.Image): Input image.
            magnitude (float): random magnitude of translation.
            params (dict[str, int | float | bool]): Transform parameters.
                Values are np.nan for irrelevant or unseen transforms
                and int or bool for seen transforms.
        Returns:
            tuple[PIL.Image.Image, dict[str, int | float | bool]: Transform
                parameters. If the new transforms are seen,
                the used parameters are registered.
        """
        if np.isnan(params["RepTranslateX"]):
            rand = random.choice([-1, 1])
            params["RepTranslateX"] = rand
        else:
            rand = params["RepTranslateX"]
        return (
            x.transform(
                x.size,
                Image.AFFINE,
                (1, 0, magnitude * x.size[0] * rand, 0, 1, 0),
                fillcolor=self.fillcolor,
            ),
            params,
        )


class RepTranslateY(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(
        self,
        x: PIL.Image.Image,
        magnitude: float,
        params: dict[str, int | float | bool],
    ) -> tuple[PIL.Image.Image, dict[str, int | float | bool]]:
        """TranslateY with reproducible parameter recording.
        Args:
            img (PIL.Image.Image): Input image.
            magnitude (float): random magnitude of translation.
            params (dict[str, int | float | bool]): Transform parameters.
                Values are np.nan for irrelevant or unseen transforms
                and int or bool for seen transforms.
        Returns:
            tuple[PIL.Image.Image, dict[str, int | float | bool]: Transform
                parameters. If the new transforms are seen,
                the used parameters are registered.
        """
        if np.isnan(params["RepTranslateY"]):
            rand = random.choice([-1, 1])
            params["RepTranslateY"] = rand
        else:
            rand = params["RepTranslateY"]
        return (
            x.transform(
                x.size,
                Image.AFFINE,
                (1, 0, 0, 0, 1, magnitude * x.size[1] * rand),
                fillcolor=self.fillcolor,
            ),
            params,
        )


class RepRotate(object):
    def __call__(
        self,
        x: PIL.Image.Image,
        magnitude: float,
        params: dict[str, int | float | bool],
    ) -> tuple[PIL.Image.Image, dict[str, int | float | bool]]:
        """Rotate transform with reproducible parameter recording.
        Args:
            img (PIL.Image.Image): Input image.
            magnitude (float): random magnitude of rotation.
            params (dict[str, int | float | bool]): Transform parameters.
                Values are np.nan for irrelevant or unseen transforms
                and int or bool for seen transforms.
        Returns:
            tuple[PIL.Image.Image, dict[str, int | float | bool]: Transform
                parameters. If the new transforms are seen,
                the used parameters are registered.
        """
        if np.isnan(params["RepRotate"]):
            rand = random.choice([-1, 1])
            params["RepRotate"] = rand
        else:
            rand = params["RepRotate"]
        rot = x.convert("RGBA").rotate(magnitude * rand)
        return (
            Image.composite(
                rot, Image.new("RGBA", rot.size, (128,) * 4), rot
            ).convert(x.mode),
            params,
        )


class RepColor(object):
    def __call__(
        self,
        x: PIL.Image.Image,
        magnitude: float,
        params: dict[str, int | float | bool],
    ) -> tuple[PIL.Image.Image, dict[str, int | float | bool]]:
        """Color enhancing transform with reproducible parameter recording.
        Args:
            img (PIL.Image.Image): Input image.
            magnitude (float): random enhancing magnitude.
            params (dict[str, int | float | bool]): Transform parameters.
                Values are np.nan for irrelevant or unseen transforms
                and int or bool for seen transforms.
        Returns:
            tuple[PIL.Image.Image, dict[str, int | float | bool]: Transform
                parameters. If the new transforms are seen,
                the used parameters are registered.
        """
        if np.isnan(params["RepColor"]):
            rand = random.choice([-1, 1])
            params["RepColor"] = rand
        else:
            rand = params["RepColor"]
        return ImageEnhance.Color(x).enhance(1 + magnitude * rand), params


class RepContrast(object):
    def __call__(
        self,
        x: PIL.Image.Image,
        magnitude: float,
        params: dict[str, int | float | bool],
    ) -> tuple[PIL.Image.Image, dict[str, int | float | bool]]:
        """Contrast enhancing transform with reproducible parameter recording.
        Args:
            img (PIL.Image.Image): Input image.
            magnitude (float): random enhancing magnitude.
            params (dict[str, int | float | bool]): Transform parameters.
                Values are np.nan for irrelevant or unseen transforms
                and int or bool for seen transforms.
        Returns:
            tuple[PIL.Image.Image, dict[str, int | float | bool]: Transform
                parameters. If the new transforms are seen,
                the used parameters are registered.
        """
        if np.isnan(params["RepContrast"]):
            rand = random.choice([-1, 1])
            params["RepContrast"] = rand
        else:
            rand = params["RepContrast"]
        return ImageEnhance.Contrast(x).enhance(1 + magnitude * rand), params


class RepSharpness(object):
    def __call__(
        self,
        x: PIL.Image.Image,
        magnitude: float,
        params: dict[str, int | float | bool],
    ) -> tuple[PIL.Image.Image, dict[str, int | float | bool]]:
        """Sharpness enhancing transform with reproducible parameter recording.
        Args:
            img (PIL.Image.Image): Input image.
            magnitude (float): random enhancing magnitude.
            params (dict[str, int | float | bool]): Transform parameters.
                Values are np.nan for irrelevant or unseen transforms
                and int or bool for seen transforms.
        Returns:
            tuple[PIL.Image.Image, dict[str, int | float | bool]: Transform
                parameters. If the new transforms are seen,
                the used parameters are registered.
        """
        if np.isnan(params["RepSharpness"]):
            rand = random.choice([-1, 1])
            params["RepSharpness"] = rand
        else:
            rand = params["RepSharpness"]
        return ImageEnhance.Sharpness(x).enhance(1 + magnitude * rand), params


class RepBrightness(object):
    def __call__(
        self,
        x: PIL.Image.Image,
        magnitude: float,
        params: dict[str, int | float | bool],
    ) -> tuple[PIL.Image.Image, dict[str, int | float | bool]]:
        """Brightness enhancing transform with reproducible
        parameter recording.
        Args:
            img (PIL.Image.Image): Input image.
            magnitude (float): random enhancing magnitude.
            params (dict[str, int | float | bool]): Transform parameters.
                Values are np.nan for irrelevant or unseen transforms
                and int or bool for seen transforms.
        Returns:
            tuple[PIL.Image.Image, dict[str, int | float | bool]: Transform
                parameters. If the new transforms are seen,
                the used parameters are registered.
        """
        if np.isnan(params["RepBrightness"]):
            rand = random.choice([-1, 1])
            params["RepBrightness"] = rand
        else:
            rand = params["RepBrightness"]
        return ImageEnhance.Brightness(x).enhance(1 + magnitude * rand), params


# Deterministic Ops.
# `params` just go through `_call_` method.


class RepToTensor(tf.ToTensor):
    def __call__(
        self, pic: PIL.Image.Image, params: dict[str, int | float | bool]
    ) -> tuple[torch.Tensor, dict[str, int | float | bool]]:
        """PIL image to tensor.

        Args:
            pic (PIL.Image.Image): Input PIL image.
            params (dict[str, int | float | bool]): Not used.

        Returns:
            tuple[torch.Tensor, dict[str, int | float | bool]]: Not used.
        """
        return super().__call__(pic), params


class RepNormalize(tf.Normalize):
    def forward(
        self, tensor: torch.Tensor, params: dict[str, int | float | bool]
    ) -> tuple[torch.Tensor, dict[str, int | float | bool]]:
        """Normalize with mean and std.
        e.g. `RepNormalize(mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761))`

        Args:
            tensor (torch.Tensor): Input tensor.
            params (dict[str, int | float | bool]): Not used.

        Returns:
            tuple[torch.Tensor, dict[str, int | float | bool]]: Not used.
        """
        return super().forward(tensor), params


class RepPosterize(object):
    def __call__(
        self,
        x: PIL.Image.Image,
        magnitude: float,
        params: dict[str, int | float | bool],
    ) -> tuple[PIL.Image.Image, dict[str, int | float | bool]]:
        """Posterization transform.
        No randomness included here.
        Args:
            img (PIL.Image.Image): Input image.
            magnitude (float): Posterization magnitude.
            params (dict[str, int | float | bool]): Not used.

        Returns:
            tuple[torch.Tensor, dict[str, int | float | bool]]: Not used.
        """
        return ImageOps.posterize(x, magnitude), params


class RepSolarize(object):
    def __call__(
        self,
        x: PIL.Image.Image,
        magnitude: float,
        params: dict[str, int | float | bool],
    ) -> tuple[PIL.Image.Image, dict[str, int | float | bool]]:
        """Solarization transform.
        No randomness included here.
        Args:
            img (PIL.Image.Image): Input image.
            magnitude (float): Posterization magnitude.
            params (dict[str, int | float | bool]): Not used.

        Returns:
            tuple[torch.Tensor, dict[str, int | float | bool]]: Not used.
        """
        return ImageOps.solarize(x, magnitude), params


class RepAutoContrast(object):
    def __call__(
        self,
        x: PIL.Image.Image,
        magnitude: float,
        params: dict[str, int | float | bool],
    ) -> tuple[PIL.Image.Image, dict[str, int | float | bool]]:
        """AutoContrast transform.
        No randomness included here.
        Args:
            img (PIL.Image.Image): Input image.
            magnitude (float): Not used.
            params (dict[str, int | float | bool]): Not used.

        Returns:
            tuple[torch.Tensor, dict[str, int | float | bool]]: Not used.
        """
        return ImageOps.autocontrast(x), params


class RepEqualize(object):
    def __call__(
        self,
        x: PIL.Image.Image,
        magnitude: float,
        params: dict[str, int | float | bool],
    ) -> tuple[PIL.Image.Image, dict[str, int | float | bool]]:
        """Equalization transform.
        No randomness included here.
        Args:
            img (PIL.Image.Image): Input image.
            magnitude (float): Not used.
            params (dict[str, int | float | bool]): Not used.

        Returns:
            tuple[torch.Tensor, dict[str, int | float | bool]]: Not used.
        """
        return ImageOps.equalize(x), params


class RepInvert(object):
    def __call__(
        self,
        x: PIL.Image.Image,
        magnitude: float,
        params: dict[str, int | float | bool],
    ) -> tuple[PIL.Image.Image, dict[str, int | float | bool]]:
        """Image inversion transform.
        No randomness included here.
        Args:
            img (PIL.Image.Image): Input image.
            magnitude (float): Not used.
            params (dict[str, int | float | bool]): Not used.

        Returns:
            tuple[torch.Tensor, dict[str, int | float | bool]]: Not used.
        """
        return ImageOps.invert(x), params
