"""
Adapted from:
https://github.com/LAMDA-CL/PyCIL/blob/master/models/base.py
"""
from logging import Logger

import numpy as np
import torch
from omegaconf.dictconfig import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from libs.datasets.base_dataset import BaseDataset


class BaseLearner:
    def __init__(
        self,
        cfg: DictConfig,
        net: nn.Module,
        dataset: BaseDataset,
        logger: Logger,
        ckpt: str | None,
        device: str,
    ):
        """Base learner.
        Initialize training target, task information and metrics.

        Args:
            cfg (DictConfig): Omegaconf object.
            net (nn.Module): Training target.
            dataset (BaseDataset): Dataset (e.g. CIFAR100).
            logger (Logger): Logger object.
            ckpt (str | None): Checkpoint path.
            device (str): cuda or cpu.
        """
        self._cfg = cfg
        self._net = net
        self._dataset = dataset
        self._logger = logger
        self._device = device
        self._known_classes = 0
        self._cur_total_classes = 0
        self.increments = [cfg.incremental.init_cls]
        while (
            sum(self.increments) + cfg.incremental.increment
            <= dataset.num_classes
        ):
            self.increments.append(cfg.incremental.increment)
        logger.info(f"incremental classes: {self.increments}")
        self._acc_matrix = {}
        self._acc_list = {}
        self._forget_list = {}
        for metric in cfg.log.metrics:
            self._acc_matrix[metric] = np.zeros(
                (len(self.increments), len(self.increments))
            )
            self._acc_list[metric] = np.zeros(len(self.increments))
            self._forget_list[metric] = np.zeros(len(self.increments))
        self._custom_metrics = {}
        self._prototypes = []
        self._trained_tasks = []
        self._ckpt = ckpt

    def _draw_dataset(self, new_classes: int, mode: str) -> Dataset:
        """Draw task-specific data.

        Args:
            new_classes (int): Number of new-task classes.
            mode (str): Dataset mode that determines data selection
                and transforms, one of ['train', 'test', 'replay',
                'apr', 'adcapr'].

        Returns:
            Dataset: Incremental dataset. If mode is 'test', a dataset
                contining all the known classes.
        """

        dataset = self._dataset.draw_dataset(
            self._known_classes,
            new_classes,
            mode=mode,
        )
        return dataset

    def _prepare_dataloader(
        self, new_classes: int, task_idx: int, mode: str
    ) -> DataLoader:
        """Prepare task-specific dataloader.

        Args:
            new_classes (int): Number of new-task classes.
            task_idx (int): Current task index, starting from zero.
            mode (str): Dataset mode that determines data selection
                and transforms, one of ['train', 'test', 'replay',
                'apr', 'adcapr'].

        Returns:
            DataLoader: Torch dataloader contining incremental dataset.
            If mode is 'test', a dataset contining all the known classes.
        """
        dataset = self._draw_dataset(new_classes, mode)
        self._logger.info(
            f"[{dataset.mode}] current total classes: "
            f"{self._cur_total_classes}, "
            f"known classes: {self._known_classes}, "
            f"dataset (mode = {dataset.mode}) prepared, labels:"
            f" {min(dataset.labels[dataset.indices])} ~ "
            f"{max(dataset.labels[dataset.indices])}"
        )
        if mode == "train":
            dataset.check_old_data(self._known_classes)
        cfg = self._cfg.data.train if mode == "train" else self._cfg.data.test
        batch_size = cfg.init_batch_size if task_idx == 0 else cfg.batch_size
        dataloader = DataLoader(
            dataset,
            batch_size,
            shuffle=(mode == "train"),
            num_workers=cfg.num_workers,
        )
        return dataloader

    def _configure_optimizer(self, optimizer_cfg, scheduler_cfg):
        if optimizer_cfg.type == "sgd":
            optimizer = torch.optim.SGD(
                self._net.parameters(),
                momentum=optimizer_cfg.momentum,
                lr=optimizer_cfg.lr,
                weight_decay=optimizer_cfg.weight_decay,
            )
        elif optimizer_cfg.type == "adam":
            optimizer = torch.optim.Adam(
                self._net.parameters(),
                lr=optimizer_cfg.lr,
                weight_decay=optimizer_cfg.weight_decay,
            )
        else:
            raise NotImplementedError()

        if scheduler_cfg.type == "cos":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=scheduler_cfg.total_epochs
            )
        elif scheduler_cfg.type == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=scheduler_cfg.milestones,
                gamma=scheduler_cfg.lrdecay,
            )
        else:
            raise NotImplementedError()
        return optimizer, scheduler

    def _loss(self, x, y, logits, task_idx):
        raise NotImplementedError()

    def _train_task(
        self,
        train_dataloader: DataLoader,
        optimizer_cfg: DictConfig,
        scheduler_cfg: DictConfig,
        task_idx: int,
    ):
        """Train for one initial / incremental task.

        Args:
            train_dataloader (DataLoader): Task-specific dataloader.
            optimizer_cfg (DictConfig): Config for building optimizer.
            scheduler_cfg (DictConfig): Config for building scheduler.
            task_idx (int): Current task index.
        """
        amp = self._cfg.learner.amp
        scaler = torch.cuda.amp.GradScaler() if amp else None
        self.optimizer, self.scheduler = self._configure_optimizer(
            optimizer_cfg, scheduler_cfg
        )
        total_epochs = scheduler_cfg.total_epochs
        epoch_loop = (
            tqdm(range(total_epochs))
            if self._cfg.log.use_tqdm
            else range(total_epochs)
        )
        for epoch in epoch_loop:
            self._net.train()
            losses = 0.0
            loss_dict_sum = {}
            correct, total = 0, 0
            self.optimizer.zero_grad()
            for i, (x, y, _) in enumerate(train_dataloader):
                x, y = x.to(self._device), y.to(self._device)
                with torch.cuda.amp.autocast(enabled=amp):
                    net_output = self._net(x)
                    loss_dict = self._loss(x, y, net_output, task_idx)
                    loss_dict_sum = self._add_loss_dict(
                        loss_dict_sum, loss_dict
                    )
                    loss = (
                        sum([v for v in loss_dict.values()])
                        / self.n_grad_accums
                    )
                    if amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    if i % self.n_grad_accums == 0 or i == len(
                        train_dataloader
                    ):
                        if amp:
                            scaler.step(self.optimizer)
                            scaler.update()
                        else:
                            self.optimizer.step()
                        self.optimizer.zero_grad()
                losses += (loss * self.n_grad_accums).item()
                preds = self._prediction(net_output)
                correct += preds.eq(y.expand_as(preds)).sum()
                total += len(y)
            self.scheduler.step()
            train_acc = correct.item() / total
            cur_lr = self.optimizer.param_groups[0]["lr"]
            loss_str = self._make_loss_str(
                loss_dict_sum, len(train_dataloader)
            )
            if (epoch + 1) % self._cfg.log.interval == 0:
                self._logger.info(
                    f"epoch {epoch + 1} / {total_epochs}, "
                    f"lr: {cur_lr:.4f}, {loss_str}"
                    f"acc: {train_acc * 100:.3f}%"
                )

    @staticmethod
    def _add_loss_dict(
        loss_dict_sum: dict[str, float], loss_dict: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        """Update the sum of each loss values for logging.

        Args:
            loss_dict_sum (dict[str, float]): Sum of each loss values,
                e.g. `{'loss_cls': 170.87933087348938}`.
            loss_dict (dict[str, torch.Tensor]): Losses for training.
                e.g. `{'loss_cls': tensor(2.0585, device='cuda:0',
                    grad_fn=<MulBackward0>)}`.

        Returns:
            dict[str, float]: Updated sum of each loss values.
        """
        if len(loss_dict_sum) == 0:
            # empty dict - first iteration
            loss_dict_sum = {k: v.item() for k, v in loss_dict.items()}
        else:
            for k in loss_dict:
                loss_dict_sum[k] += loss_dict[k].item()
        return loss_dict_sum

    @staticmethod
    def _make_loss_str(loss_dict_sum: dict[str, float], n_iters: int) -> str:
        """Make loss stats message out of loss_dict_sum.

        Args:
            loss_dict_sum (dict[str, float]): Sum of each loss values,
                e.g. `{'loss_cls': 170.87933087348938}`.
            n_iters (int): Number of iterations in the epoch.

        Returns:
            str: Generated message. e.g.
                `loss_cls: 0.2214, loss_kd: 19.0628, `.
        """
        loss_str = ""
        for k, v in loss_dict_sum.items():
            loss_str += f"{k}: {v / n_iters:.4f}, "
        return loss_str

    def _prediction(self, net_output: dict[torch.Tensor]) -> torch.Tensor:
        """Evaluate the network output logits and return
        predicted class indices.

        Args:
            net_output (dict[torch.Tensor]): Network output dict
                contining "logits" tensor.

        Returns:
            torch.Tensor: Predicted class indices.
        """
        return torch.max(net_output["logits"], dim=1)[1]

    def _evaluate(
        self, preds: torch.Tensor, targets: torch.Tensor, task_idx: int
    ):
        """Evaluate logit-based ('Linear') correctness.

        Args:
            preds (torch.Tensor): Predicted class ids, shape (N).
            targets (torch.Tensor): Ground-truth class ids, shape (N).
            task_idx (int): Current task index.
        """
        correct = preds == targets
        self._calc_metrics(correct, targets, task_idx, "Linear")

    def _calc_metrics(
        self,
        correct: torch.Tensor,
        targets: torch.Tensor,
        task_idx: int,
        metric: str,
    ):
        """Calculate accuracy and incremental-learning metrics:
            - avg_accuracy: Average accuracy for all the classes at each task.
            - avg_forgetting: Average forgetting for all the
                classes at each task,
                (maximum accuracy) - (current accuracy) for each class group.
            - _acc_matrix: The current task row of the accuracy matrix,
                with class group columns. Zero for unknown class groups.
            The results are registered to `self._acc_list`, `self._forget_list`
            and `self._acc_matrix`.

        Args:
            correct (torch.Tensor): Result of `preds == targets`.
            targets (torch.Tensor): Ground-truth class ids, shape (N).
            task_idx (int): Current task index.
            metric (str): Metric name (e.g. 'Linear'), used as a key
                for the all-task results (e.g. self._acc_matrix).
        """
        _acc_matrix = self._acc_matrix[metric]
        prev_cls = 0
        forget_last = np.zeros(len(self.increments))
        for k, inc in enumerate(self.increments):
            correct_k = correct[
                (targets >= prev_cls) & (targets < prev_cls + inc)
            ]
            if len(correct_k) > 0:
                _acc_matrix[task_idx, k] = correct_k.float().mean().item()
            prev_cls += inc
            if task_idx > k:
                forget_last[k] = (
                    np.max(_acc_matrix[:task_idx, k])
                    - _acc_matrix[task_idx, k]
                )
        avg_accuracy = correct.float().mean().item()
        avg_forgetting = forget_last[:task_idx].mean() if task_idx > 0 else 0
        self._logger.info(
            f"{metric} accuracy   : "
            + "".join(f"{a*100:.2f}, " for a in _acc_matrix[task_idx])
        )
        self._logger.info(
            f"{metric} forgetting : "
            + "".join(f"{f*100:.2f}, " for f in forget_last)
        )
        self._logger.info(
            f"{metric} average test accuracy and forgetting "
            f"for {self._cur_total_classes} classes : "
            f"{avg_accuracy * 100:.2f} %, {avg_forgetting * 100:.2f} %"
        )
        self._acc_list[metric][task_idx] = avg_accuracy
        self._forget_list[metric][task_idx] = avg_forgetting
        self._acc_matrix[metric] = _acc_matrix

    def _test_task(self, dataloader: DataLoader, task_idx: int):
        """Evaluate the network with the test dataset.
        Modify this method to add or change test metrics.

        Args:
            dataloader (DataLoader): Test dataloader.
            task_idx (int): Current task index.
        """
        preds, targets, _ = self._inference(dataloader)
        self._evaluate(preds, targets, task_idx)

    def _inference(
        self,
        dataloader: DataLoader,
        return_features: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Do inference for all the data in the dataloader.

        Args:
            dataloader (DataLoader): Target dataloader.
            return_features (bool, optional): If true, extracted
                features are also computed. Defaults to False.

        Returns:
            tuple[torch.Tensor, ...]:
                preds : Predicted class ids, shape (N).
                targets : Ground-truth class ids, shape (N).
                features : Extracted features, shape (N, dim).
                (N is the number of samples in the dataloader.)
        """
        self._net.eval()
        num_features = len(dataloader.dataset)
        features = torch.zeros(num_features, self._net.net.out_dim).to(
            self._device
        )
        pred_list = []
        target_list = []
        pt = 0
        for x, y, _ in dataloader:
            bs = x.shape[0]
            x = x.to(self._device)
            y = y.to(self._device)
            with torch.no_grad():
                out = self._net(x)
            pred_list.append(self._prediction(out))
            if return_features:
                features[pt : pt + bs] = out["features"]
            target_list.append(y)
            pt += bs
        preds = torch.cat(pred_list)
        targets = torch.cat(target_list)
        features = features if return_features else None
        return preds, targets, features

    def _run_task(self, task_idx: int, new_classes: int):
        """Execute one task sequence.

        Args:
            task_idx (int): Current task index. Starts from 0.
            new_classes (int): Numbe of incremental classes.
        """
        # Preprocess (e.g. update FC layer)
        self._before_task(new_classes, task_idx)
        train_dataloader = self._prepare_dataloader(
            new_classes, mode="train", task_idx=task_idx
        )
        self._logger.info(
            f"Starting task {task_idx}, "
            f"new classes: {new_classes}, "
            f"known classes: {self._known_classes}, "
            f"dataset size: {len(train_dataloader.dataset)}"
        )
        if self._ckpt is not None:
            if "{}" in self._ckpt:
                # Wildcard-style checkpoint for all the tasks.
                # Load the checkpoint for the class and skip training.
                self._load_checkpoint(self._ckpt.format(task_idx))
            else:
                # Load initial-task checkpoint only once.
                # Training for this task is skipped.
                self._load_checkpoint(self._ckpt)
                self._ckpt = None
        else:
            # Prepare optimizer and scheduler
            if task_idx == 0:
                optimizer_cfg, scheduler_cfg = (
                    self._cfg.init_optimizer,
                    self._cfg.init_scheduler,
                )
                self.n_grad_accums = self._cfg.init_optimizer.grad_accum
            else:
                optimizer_cfg, scheduler_cfg = (
                    self._cfg.incremental_optimizer,
                    self._cfg.incremental_scheduler,
                )
                self.n_grad_accums = self._cfg.incremental_optimizer.grad_accum
            # Train for the task
            self._train_task(
                train_dataloader, optimizer_cfg, scheduler_cfg, task_idx
            )
        # Postprocess (e.g. save checkpoint)
        self._after_task(task_idx)
        # Test
        test_dataloader = self._prepare_dataloader(
            new_classes=0, task_idx=task_idx, mode="test"
        )
        self._test_task(test_dataloader, task_idx)

    def _before_task(self, new_classes: int, task_idx: int):
        """Update the task information and FC layer
        for the incoming new task.

        Args:
            new_classes (int): Number of incremental classes.
            task_idx (int): Current task index.
        """
        self._cur_total_classes = self._known_classes + new_classes
        self._logger.info(
            f"[Start new task {task_idx}] "
            "current total classes is updated to "
            f"{self._cur_total_classes}"
        )
        self._net.update_fc(self._cur_total_classes, task_idx=task_idx)
        self._net.to(self._device)

    def _after_task(self, task_idx: int):
        """Do the post-process after the task.

        Args:
            task_idx (int): Current task index.
        """
        # update number of known classes
        self._known_classes = self._cur_total_classes
        if (
            task_idx in self._cfg.log.ckpt_idx
            and task_idx not in self._trained_tasks
        ):
            self._save_checkpoint(task_idx)
        self._logger.info(
            f"[End of task {task_idx}] "
            "known classes is updated to "
            f"{self._known_classes}"
        )

    def _save_checkpoint(self, task_idx):
        ckpt = dict(
            task_idx=task_idx,
            cfg=self._cfg,
            state_dict=self._net.state_dict(),
        )
        savepath = self._cfg.log.ckpt_tmpl.format(task_idx)
        torch.save(ckpt, savepath)

    def _load_checkpoint(self, ckpt_path):
        self._logger.info(f"loading from {ckpt_path}...")
        ckpt = torch.load(ckpt_path)
        task_idx = ckpt["task_idx"]

        self._trained_tasks = [i for i in range(task_idx + 1)]
        self._net.to(self._device)
        self._net.load_state_dict(ckpt["state_dict"], strict=False)

    def _log_acc_list(self):
        for metric in self._acc_matrix:
            newtask = np.trace(self._acc_matrix[metric][1:, 1:]) / len(
                self._acc_matrix[metric][1:, 1:]
            )
            log_str = f"{metric} accuracy list: "
            for acc in self._acc_list[metric]:
                log_str += f"{acc * 100:.2f}, "
            log_str += f"avg: {np.mean(self._acc_list[metric]) * 100:.2f} %, "
            log_str += f"final: {self._acc_list[metric][-1] * 100:.2f} %, "
            log_str += f"forget: {self._forget_list[metric][-1] * 100:.2f} %, "
            log_str += f"newtask: {newtask * 100:.2f} %"
            self._logger.info(log_str)

    def run(self) -> tuple[dict[str, np.ndarray], ...]:
        """Execute the learner.

        Returns:
            tuple[dict[str, np.ndarray], ...]:
                acc_list (dict[str, np.ndarray]): Accuracy list of
                    each task, for multiple metrics (e.g. NCM).
                forget_list (dict[str, np.ndarray]): Forgetting list
                    of each task, for multiple metrics.
                acc_matrix (dict[str, np.ndarray]): Accuracy matrix
                    of all the tasks and class groups,
                    for multiple metrics.
                custom_metrics (dict[str, list]): Learner-specific
                    metrics such as feature-target distance of APR.
        """
        for task_idx, new_classes in enumerate(self.increments):
            self._run_task(task_idx, new_classes)
        self._log_acc_list()
        return (
            self._acc_list,
            self._forget_list,
            self._acc_matrix,
            self._custom_metrics,
        )
