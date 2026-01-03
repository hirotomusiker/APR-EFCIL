import os
import random
import subprocess

import numpy as np
import torch


def seed_everything(seed: int):
    """Set a random seed and ensure
    deterministic experiment.

    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def normalize(features, eps=1e-8):
    norm = np.linalg.norm(features.T, axis=0)
    return (features.T / norm + eps).T


def get_git_sha():
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
        return sha
    except subprocess.CalledProcessError:
        return None


def get_git_diff():
    try:
        # Run the 'git diff' command and capture its output
        result = subprocess.run(
            ["git", "diff"], capture_output=True, text=True
        )

        if result.returncode == 0:
            return result.stdout  # This contains the diff results
        else:
            return f"Error: {result.stderr}"  # Handle any errors
    except Exception as e:
        return f"Exception occurred: {str(e)}"
