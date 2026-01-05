import logging
import sys
from logging import Logger

import numpy as np


def build_logger(name: str, logfilename: str) -> Logger:
    """Create a custom logger object.

    Args:
        name (str): Logger name.
        logfilename (str): Output path where
            an output log file is saved.

    Returns:
        Logger: Logger object.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(filename=logfilename),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


def log_results(
    acc_list: dict[str, np.ndarray],
    forget_list: dict[str, np.ndarray],
    acc_matrix: dict[str, np.ndarray],
    elapsed_time: float,
    logger: Logger,
):
    """Log the final results of the incremental training.

    Args:
        acc_list (dict[str, np.ndarray]): Accuracy list of
            each task, for multiple metrics (e.g. NCM).
        forget_list (dict[str, np.ndarray]): Forgetting list
            of each task, for multiple metrics.
        acc_matrix (dict[str, np.ndarray]): Accuracy matrix
            of all the tasks and class groups, for multiple metrics.
        elapsed_time (float): Time of the training.
        logger (Logger): Logger object.
    """
    metrics = acc_list.keys()
    t = int(elapsed_time)
    logger.info(
        f"Elapsed time: {t//(60**2):d}h {(t%60**2)//60:0>2}m " f"{(t%60):0>2}s"
    )
    logger.info("Final results: ")
    for metric in metrics:
        newtask = np.trace(acc_matrix[metric][1:, 1:]) / (
            len(acc_matrix[metric]) - 1
        )
        logger.info(f"metric: {metric}")
        log_str = ""
        for acc in acc_list[metric]:
            log_str += f"{acc * 100:.2f}, "
        log_str += f"acc avg: {acc_list[metric].mean() * 100:.2f} "
        log_str += f"acc final: {acc_list[metric][-1] * 100:.2f} "
        log_str += f"newtask final: {newtask.mean() * 100:.2f} "
        log_str += f"forget final: {forget_list[metric][-1] * 100:.2f} "
        logger.info(log_str)

        # acc matrix
        for acc_row in acc_matrix[metric]:
            log_str = ""
            for acc in acc_row:
                log_str += f"{acc*100:5.2f}, "
            logger.info(log_str[:-2])
