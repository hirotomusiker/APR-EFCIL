import argparse
import sys
from logging import Logger
from pathlib import Path

import numpy as np
import pandas as pd

from .misc import get_git_diff
from .misc import get_git_sha


class Recorder:
    def __init__(self, logger: Logger, csv_path: str, metrics: list[str]):
        """Experiment recorder to record experiment settings and results
        - incl. config and arg settings, git hash, git diff, all metrics
        results and elapsed time. If the existing csv file is specified
         with `csv_path`, another experiment results are appended.

        Args:
            logger (Logger): Logger object.
            csv_path (str): Output csv path.
            metrics (list[str]): Metric list (e.g. "NCM").
        """
        self.logger = logger
        self.csv_path = csv_path
        self.csv_mirror_path = csv_path.replace(".csv", "_mirror.csv")
        self.records = []
        self.metrics = metrics
        if Path(csv_path).exists():
            self.old_df = pd.read_csv(csv_path)
            self.start_exp = self.old_df.exp.max()
        else:
            self.old_df = None
            self.start_exp = 0

    def add_header(self, cfg: dict, args: argparse.Namespace, seed: int):
        """Add the header part containint cfg, args and
        git info to the recorder.

        Args:
            cfg (dict): Dict generated via
                `OmegaConf.to_container(cfg, resolve=True)`.
            args (argparse.Namespace): Args of the train command.
            seed (int): Random seed (not the class seed).
        """
        for metric in self.metrics:
            record = dict()
            record["exp"] = self.start_exp + 1
            record["metric"] = metric
            record["git_diff"] = get_git_diff()
            record["git_sha"] = get_git_sha()
            record.update(vars(args))
            record.update(cfg)
            record["command"] = "python " + " ".join(sys.argv)
            record["seed"] = seed
            self.records.append(record)

    def add_results(
        self,
        acc_list: dict[str, np.ndarray],
        forget_list: dict[str, np.ndarray],
        acc_matrix: dict[str, np.ndarray],
        custom_metrics: dict[str, list],
        elapsed_time: float,
    ):
        """Log the final results of the incremental training.

        Args:
            acc_list (dict[str, np.ndarray]): Accuracy list of
                each task, for multiple metrics (e.g. NCM).
            forget_list (dict[str, np.ndarray]): Forgetting list
                of each task, for multiple metrics.
            acc_matrix (dict[str, np.ndarray]): Accuracy matrix
                of all the tasks and class groups, for multiple metrics.
            custom_metrics (dict[str, list]): Learner-specific
                metrics such as feature-target distance of APR.
            elapsed_time (float): Time of the training.
            logger (Logger): Logger object.
        """
        for i, metric in enumerate(self.metrics):
            result = {}
            assert self.records[i]["metric"] == metric
            result["acc_list"] = acc_list[metric]
            result["acc_inc"] = float(np.mean(acc_list[metric]))
            result["acc_last"] = acc_list[metric][-1]
            result["forget_list"] = forget_list[metric]
            result["forget_inc"] = float(np.mean(forget_list[metric]))
            result["newtask_inc"] = float(
                np.trace(acc_matrix[metric][1:, 1:])
                / (len(acc_matrix[metric]) - 1)
            )
            result["task0_last"] = acc_matrix[metric][-1, 0]
            result["acc_matrix"] = acc_matrix[metric]
            result["elapsed_time"] = elapsed_time
            for metric in custom_metrics:
                result[metric] = custom_metrics[metric]
            self.records[i].update(result)

    def log_header(self):
        """Read the header data and yield log output"""
        r = self.records[self.metrics[0]]
        self.logger.info(f"Loaded config: {r['cfg']}")
        self.logger.info(f"git sha = {r['git_sha']}")
        self.logger.info(f"git diff = {r['git_diff']}")
        self.logger.info("#" * 50)
        self.logger.info()
        self.logger.info("#" * 50)

    def save_csv(self):
        """Dump the records onto the csv file.
        A 'mirror' easy-to-view csv without git diff and
        acc matrix is generated additionally.
        """
        df = pd.json_normalize(self.records)
        if self.old_df is not None:
            df = pd.concat([self.old_df, df])
        df.to_csv(self.csv_path, index=False)
        # optional: mirror csv for viewing
        df = df.drop(columns=["git_diff", "acc_matrix"])
        df.to_csv(self.csv_mirror_path, index=False)
