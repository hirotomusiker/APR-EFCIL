"""
References:
https://github.com/LAMDA-CL/PyCIL/blob/master/models/base.py
https://github.com/dipamgoswami/ADC/blob/main/models/lwf.py
"""
import copy
import gc
import math
from logging import Logger

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf.dictconfig import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from tqdm import tqdm

from libs.datasets.base_dataset import BaseDataset
from libs.learners.base import BaseLearner
from libs.nets.layers.linears import SimpleLinear
from libs.utils.adc import Attack
from libs.utils.cov import mahalanobis
from libs.utils.cov import shrink_cov
from libs.utils.cov import svd_compose
from libs.utils.cov import svd_decompose
from libs.utils.misc import count_parameters


class APR(BaseLearner):
    def __init__(
        self,
        cfg: DictConfig,
        net: nn.Module,
        dataset: BaseDataset,
        logger: Logger,
        ckpt: str | None,
        device: str,
    ):
        """Adversarial Pseudo Replay learner.

        Args:
            cfg (DictConfig): Omegaconf object.
            net (nn.Module): Training target.
            dataset (BaseDataset): Dataset (e.g. CIFAR100).
            logger (Logger): Logger object.
            ckpt (str | None): Checkpoint path.
            device (str): cuda or cpu.
        """
        super().__init__(cfg, net, dataset, logger, ckpt, device)
        self._custom_metrics["adc_dists"] = []
        self._custom_metrics["apr_dists"] = []
        self._custom_metrics["perturb_dists"] = []
        self._custom_metrics["perturb_counts"] = []

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
        if task_idx > 0 and self._cfg.apr.do_apr:
            self.iter_p_dataloader = iter(self.p_dataloader)  # before for-loop
        for epoch in epoch_loop:
            self._net.train()
            losses = 0.0
            dists, perturb_count = 0.0, 0
            loss_dict_sum = {}
            correct, total = 0, 0
            self.optimizer.zero_grad()
            for i, (x, y, _) in enumerate(train_dataloader):
                if task_idx > 0 and self._cfg.apr.do_apr:
                    try:
                        p_x, p_y, _ = next(self.iter_p_dataloader)
                    except StopIteration:
                        self.iter_p_dataloader = iter(self.p_dataloader)
                        p_x, p_y, _ = next(self.iter_p_dataloader)
                    p_x = p_x.to(device=self._device, dtype=x.dtype)
                    p_y = p_y.to(device=self._device, dtype=y.dtype)
                    if self._cfg.apr.perturb:
                        p_x, p_y, dist = self._perturb(p_x, p_y)
                        dists += dist
                        perturb_count += len(p_x)
                    x = torch.cat([x.to(self._device), p_x])
                    y = torch.cat([y.to(self._device), p_y])
                else:
                    x = x.to(self._device)
                    y = y.to(self._device)
                with torch.cuda.amp.autocast(enabled=amp):
                    net_output = self._net(x)
                    loss_dict = self._loss(
                        x,
                        y,
                        net_output,
                        task_idx,
                    )
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
        if task_idx > 0:
            dist_mean = dists / perturb_count if perturb_count > 0 else -1
            perturb_ratio = perturb_count / total
            self._custom_metrics["perturb_dists"].append(dist_mean)
            self._custom_metrics["perturb_counts"].append(perturb_ratio)
            self._logger.info(
                f"APR stats: dists {dist_mean:.3f} "
                f"perturbation ratio {perturb_ratio}"
            )

    def _loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        net_output: dict[torch.Tensor],
        task_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Calculate losses for APR.

        Args:
            x (torch.Tensor): Input image data, including
                pseudo replay data.
            y (torch.Tensor): Ground-truth labels, including
                pseudo labels (`< self._known_classes`).
            net_output (dict[torch.Tensor]): Network output dict.
            task_idx (int): Current task index.

        Returns:
            dict[str, torch.Tensor]: Loss tensor for each loss kind.
        """
        loss_dict = {}
        cls_lambda = self._cfg.loss.cls_lambda
        kd_lambda = self._cfg.loss.kd_lambda
        if cls_lambda > 0:
            # CE loss for current-tack classes (e.g. y >= 10)
            cur_inds = y >= self._known_classes
            loss_dict["loss_cls"] = cls_lambda * F.cross_entropy(
                net_output["logits"][cur_inds, self._known_classes :],
                y[cur_inds] - self._known_classes,
            )
        if task_idx > 0:
            with torch.no_grad():
                old_output = self._old_net(x)  # eval mode
            if kd_lambda > 0:
                # KD loss for all the data (new-task + pseudo-replay)
                loss_dict["loss_kd"] = kd_lambda * self._kd_loss(
                    net_output["logits"][:, : self._known_classes],
                    old_output["logits"],
                )
        return loss_dict

    def _before_task(self, new_classes: int, task_idx: int):
        """Update the task information and FC layer
        for the incoming new task.
        Prepare the old network and APR dataloader.

        Args:
            new_classes (int): Number of incremental classes.
            task_idx (int): Current task index.
        """
        if task_idx > 0:
            self._old_net = copy.deepcopy(self._net).freeze().to(self._device)
            n_params = count_parameters(self._old_net)
            self._logger.info(f"old net with {n_params} parameters preserved")
        super()._before_task(new_classes, task_idx)
        if task_idx > 0 and self._cfg.apr.do_apr:
            self._build_apr_dataloader()
            self._logger.info(
                f"{len(self.p_dataloader.dataset)} pseudo data prepared"
            )
        else:
            self._logger.info("Pseudo data generation is bypassed")
        gc.collect()

    def _after_task(self, task_idx: int):
        """Do the post-process after the task.
        - Build new-task prototypes (mean features) and
            covariance matrices
        - Conduct Adversarial Drift Compensation (ADC) to update
            old-task prototypes and covariance matrices
        - Update _known_classes

        Args:
            task_idx (int): Current task index.
        """
        if (
            task_idx in self._cfg.log.ckpt_idx
            and task_idx not in self._trained_tasks
        ):
            self._save_checkpoint(task_idx)

        # store new-task prototypes
        self._build_protos(self._cur_total_classes, self._known_classes)
        if task_idx > 0:
            if self._cfg.adc.do_adc:
                self._adc()
        # update number of known classes
        self._known_classes = self._cur_total_classes
        self._logger.info(
            f"[End of task {task_idx}] "
            "known classes is updated to "
            f"{self._known_classes}"
        )
        gc.collect()

    def _adc_inference(
        self,
        dataloader: DataLoader,
        old_net: bool = False,
    ):
        """Do inference for all the data in the dataloader.

        Args:
            dataloader (DataLoader): Target dataloader.
            old_net (bool, optional): If true, the old task
            network is used for inference. Defaults to False.

        Returns:
            torch.Tensor : Extracted features, shape (N, dim).
            (N is the number of samples in the dataloader.)
        """
        if old_net:
            self._old_net.eval()
        else:
            self._net.eval()
        num_features = len(dataloader.dataset)
        features = torch.zeros(num_features, self._net.net.out_dim).to(
            self._device
        )
        pt = 0
        for x, _ in dataloader:
            bs = x.shape[0]
            x = x.to(self._device)
            with torch.no_grad():
                if old_net:
                    out = self._old_net(x)
                else:
                    out = self._net(x)
            features[pt : pt + bs] = out["features"]
            pt += bs
        return features

    def _obtain_one_class_features(
        self, cls_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Do inference to obtain features for
        class prototype and covariance matrix.

        Args:
            cls_id (int): Target class id.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                features : Extracted features, shape (N, dim).
                labels : Ground-truth class ids, shape (N).
        """
        # Extract data subset in 'replay' mode:
        # 'train' data without random augmentations.
        dataset = self._dataset.draw_dataset(cls_id, 1, mode="replay")
        self._logger.info(
            f"[Proto] current total classes: {self._cur_total_classes}, "
            f"known classes: {self._known_classes}, "
            f"dataset for prototype cls_id = {cls_id}, "
            f"(mode = {dataset.mode}) "
            "prepared, labels: "
            f"{min(dataset.labels[dataset.indices])} ~ "
            f"{max(dataset.labels[dataset.indices])}"
        )
        # Check cross-task data leak
        dataset.check_old_data(self._known_classes)
        dataloader = DataLoader(
            dataset,
            batch_size=self._cfg.data.train.batch_size,
            shuffle=False,
            num_workers=self._cfg.data.train.num_workers,
        )
        _, labels, features = self._inference(
            dataloader,
            return_features=True,
        )
        return features, labels

    def _build_protos(self, cur_total_classes: int, known_classes: int):
        """Calculate prototypes (class mean, covariance).

        Args:
            cur_total_classes (int): Number of (old + new) classes.
            known_classes (int): Number of old classes.
        """
        self._logger.info(
            "[Proto] start building new protos and covs "
            f"from cls_id {known_classes} to "
            f"{cur_total_classes - 1}"
        )
        for cls_id in range(known_classes, cur_total_classes):
            cls_features, labels = self._obtain_one_class_features(cls_id)
            label = int(torch.unique(labels).item())
            cls_mean = cls_features.mean(0)
            cls_cov = self._compute_cov(
                cls_features,
            )
            proto = dict(
                proto=cls_mean,
                label=label,
                cov=cls_cov,
                n_samples=cls_features.shape[0],
            )
            del cls_features
            self._prototypes.append(proto)
            if (cls_id + 1) % 20 == 0:
                self._logger.info(
                    f"prototype {cls_id + 1}"
                    f" / {cur_total_classes} classes created"
                )

    def _compute_cov(
        self, features: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor]:
        """Calculate covariance matrix.
        Apply low-rank decomposition optionally.

        Args:
            features (torch.Tensor): Input features, shape (N, dim).

        Returns:
            torch.Tensor | tuple[torch.Tensor]:
                cls_cov (torch.Tensor): Class covariance matrix,
                    shape (N, dim).
                Optionally,
                cls_cov (tuple[torch.Tensor]): Decomposed covariance:
                    - U_k (torch.Tensor): Shape (dim, k).
                    - S_k (torch.Tensor): Shape (k, k).
                    - VT_k (torch.Tensor): Shape (k, dim).
        """
        cls_cov = torch.cov(features.T)  # (512, 500) -> (512, 512)
        if self._cfg.protos.svd_k is not None:
            cls_cov = svd_decompose(cls_cov, k=self._cfg.protos.svd_k)
        return cls_cov

    def _transfer_cov(
        self, net: nn.Module, cov: torch.Tensor | tuple[torch.Tensor]
    ) -> torch.Tensor:
        """Transfer a covariance matrix from the
        old feature space to the new one, with the
        learned linear network weights.

        Args:
            net (nn.Module): SimpleLinear layer trained
                to transfer covariance matrix.
            cov (torch.Tensor): Covariance matrix in the
                old feature space, shape (dim, dim).
                When `self._cfg.protos.svd_k is not None`,
                a tuple of decomposed matrices with the
                shapes of (dim, k), (k, k), (k, dim).

        Returns:
            torch.Tensor: Transformed covariance matrix,
                shape (dim, dim).
                When `self._cfg.protos.svd_k is not None`,
                a tuple of decomposed matrices with the
                shapes of (dim, k), (k, k), (k, dim).
        """
        with torch.no_grad():
            W = net.weight.data
            if isinstance(cov, tuple):
                U, S, VT = cov
                U = W @ U
                VT = VT @ W.T
                cov = U, S, VT
            else:
                cov = W @ cov @ W.T
        return cov

    def _kd_loss(
        self, logits: torch.Tensor, logits_old: torch.Tensor, T: float = 2.0
    ) -> torch.Tensor:
        """Calculate knowledge distillation loss.

        Args:
            logits (torch.Tensor): Input logits.
            logits_old (torch.Tensor): Reference logits.
            T (float, optional): Temperature parameter.
                Defaults to 2.0.

        Returns:
            torch.Tensor: Scalar loss tensor.
        """
        pred = torch.log_softmax(logits / T, dim=1)
        soft = torch.softmax(logits_old / T, dim=1)
        kd_loss = -1 * torch.mul(soft, pred).sum() / pred.shape[0]
        return kd_loss

    def _trans_loss(
        self, features: torch.Tensor, features_old: torch.Tensor
    ) -> torch.Tensor:
        """Feature transfer loss.

        Args:
            features (torch.Tensor): Input features.
            features_old (torch.Tensor): Reference features.

        Returns:
            torch.Tensor: Scalar loss tensor.
        """
        criterion = nn.MSELoss(reduction="none")
        loss = torch.sqrt(criterion(features, features_old).sum(dim=-1)).mean()
        return loss

    def _maha_dist(
        self,
        vectors: torch.Tensor,
        class_means: list[torch.Tensor],
        gamma1_eval: float,
    ) -> torch.Tensor:
        """Calculate Mahalanobis distances between input
        vectors and prorotypes (class_means and covariances).

        Args:
            vectors (torch.Tensor): Input vectors (features), shape (N, dim).
            class_means (list[torch.Tensor]): Prototype mean features of
                all the classes (_cur_total_classes).
            gamma1_eval (float): Covariance shrinkage parameter.

        Returns:
            torch.Tensor: Maahlanobis distances, shape (_cur_total_classes, N).
        """
        maha_dist = torch.zeros([self._cur_total_classes, len(vectors)]).to(
            self._device
        )
        ranks = []
        ranks_after_shrink = []
        for class_index in range(self._cur_total_classes):
            cov = self._prototypes[class_index]["cov"]
            if self._cfg.protos.svd_k is not None:
                cov = svd_compose(*cov)
            rank = torch.linalg.matrix_rank(cov, tol=0.01).item()
            ranks.append(rank)
            cov = shrink_cov(cov, gamma1=gamma1_eval, gamma2=gamma1_eval)
            rank = torch.linalg.matrix_rank(cov, tol=0.01).item()
            ranks_after_shrink.append(rank)
            dist = mahalanobis(
                vectors,
                class_means[class_index],
                cov,
                normalize=self._cfg.protos.normalize,
            )
            maha_dist[class_index] = dist
        del dist
        rank = sum(ranks) / len(ranks)
        rank_after_shrink = sum(ranks_after_shrink) / len(ranks_after_shrink)
        log_str = f"Avg. rank for classes 0-{class_index}: {rank:.2f}"
        log_str += (
            f"-> shrink with alpha={gamma1_eval} -> {rank_after_shrink:.2f}"
        )
        self._logger.info(log_str)

        return maha_dist

    def _test_task(self, dataloader: DataLoader, task_idx: int):
        """Evaluate the network with the test dataset with the
        metrics in `log.metrics` list.

        Args:
            dataloader (DataLoader): Test dataloader.
            task_idx (int): Current task index.
        """
        self._logger.info("test task")
        preds, targets, features = self._inference(
            dataloader, return_features=True
        )  # gpu memory spike
        self._logger.info("finished inference")
        if "Mahalanobis" in self._cfg.log.metrics:
            self._evaluate_maha(
                features, targets, task_idx, self._cfg.protos.gamma1_eval
            )
        self._logger.info("finished test")
        if "NCM" in self._cfg.log.metrics:
            self._evaluate_ncm(features, targets, task_idx)
        self._logger.info("finished test")
        self._evaluate(preds, targets, task_idx)
        self._logger.info("finished test")

    def _evaluate_maha(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        task_idx: int,
        gamma1_eval: float,
    ):
        """Calculate Mahalanobis metric.

        Args:
            features (torch.Tensor): Test data features, shape (N, dim).
            targets (torch.Tensor): Test data labels, shape (N).
            task_idx (int): Current task index.
            gamma1_eval (float): Covariance shrinkage parameter.
        """
        features = (features.T / (torch.norm(features.T, dim=0, p=2) + 1e-8)).T
        class_means = [p["proto"] for p in self._prototypes]
        scores = self._maha_dist(features, class_means, gamma1_eval).T
        preds = torch.argmin(scores, dim=1)
        del scores
        correct = preds == targets
        self._calc_metrics(correct, targets, task_idx, "Mahalanobis")

    def _evaluate_ncm(
        self, features: torch.Tensor, targets: torch.Tensor, task_idx: int
    ):
        """Calculate nearest class mean (NCM) metric.

        Args:
            features (torch.Tensor): Test data features, shape (N, dim).
            targets (torch.Tensor): Test data labels, shape (N).
            task_idx (int): Current task index.
        """
        features = (features.T / (torch.norm(features.T, dim=0, p=2) + 1e-8)).T
        class_means = torch.stack([p["proto"] for p in self._prototypes])
        class_means = F.normalize(class_means, p=2, dim=-1)
        scores = torch.cdist(class_means, features, 2).T
        preds = torch.argmin(scores, dim=1)
        del scores
        correct = preds == targets
        self._calc_metrics(correct, targets, task_idx, "NCM")

    def _adc_draw_closest_data(
        self, dataset: Dataset, closest: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw the top-k closest samples
        from the new task dataset.

        Args:
            dataset (Dataset): New-task dataset.
            closest (np.ndarray): Indices of the top-k
            closest data to the target prototype.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                x_top: Image data tensor, shape (k, 3, H, W).
                y_top: Label data tensor, shape (k).
        """
        subset_dataset = self._dataset.draw_dataset(
            self._known_classes,
            self._cur_total_classes - self._known_classes,
            mode=self._cfg.adc.input_data_mode,
        )
        subset_dataset.indices = dataset.indices[closest]
        if self._cfg.adc.input_data_mode == "adcapr":
            subset_dataset.rep_params = {
                k: dataset.rep_params[k][closest].tolist()
                for k in dataset.rep_params
            }
        subset_train_loader = DataLoader(
            subset_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=self._cfg.data.train.num_workers,
        )
        x_top = []
        y_top = []
        for x, y, _ in subset_train_loader:
            x_top.append(x.to(self._device))
            y_top.append(y.to(self._device))
        del subset_dataset
        return torch.cat(x_top, dim=0), torch.cat(y_top, dim=0)

    def _adc(self):
        """Adversarial drift compensation."""
        adc_sample_limit = self._cfg.adc.adc_sample_limit
        assert self._cfg.adc.input_data_mode in ["train", "replay", "adcapr"]
        dataset = self._dataset.draw_dataset(
            self._known_classes,
            self._cur_total_classes - self._known_classes,
            mode=self._cfg.adc.input_data_mode,
        )
        self._logger.info(
            f"[ADC] current total classes: {self._cur_total_classes}, "
            f"known classes: {self._known_classes}, "
            f"dataset for ADC (mode = {dataset.mode}) prepared, "
            f"labels: {min(dataset.labels[dataset.indices])} "
            f"~ {max(dataset.labels[dataset.indices])}"
        )
        dataset.check_old_data(self._known_classes)
        adc_train_loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=self._cfg.data.train.num_workers,
        )
        for k, (img, _, _) in enumerate(adc_train_loader):
            if k == 0:
                x_min = img.min()
                x_max = img.max()
            else:
                if img.min() < x_min:
                    x_min = img.min()
                if img.max() > x_max:
                    x_max = img.max()

        feats = []
        params = []
        for img, _, param in adc_train_loader:
            out = self._old_net(img.to(self._device))
            feats.append(out["features"])
            if self._cfg.adc.input_data_mode == "adcapr":
                params.append(param)
        feats = torch.cat(feats, dim=0)
        if self._cfg.adc.input_data_mode == "adcapr":
            params_dict = {}
            for k in params[0].keys():
                if isinstance(params[0][k], list):
                    params_dict[k] = torch.cat(
                        [torch.stack(p[k], dim=1) for p in params]
                    )
                else:
                    params_dict[k] = torch.cat([p[k] for p in params])
            dataset.register_rep_params(params_dict)
            del params_dict
            del params
        protos = [p["proto"] for p in self._prototypes[: self._known_classes]]
        new_protos, new_covs = [], []
        adc_dists = []
        for class_idx in range(0, self._known_classes):
            d = torch.cdist(feats, protos[class_idx].unsqueeze(0)).squeeze()
            closest = torch.argsort(d)[:adc_sample_limit].cpu().numpy()
            x_top, y_top = self._adc_draw_closest_data(dataset, closest)
            if self._cfg.adc.attack:
                idx_dataset = TensorDataset(x_top, y_top)
                attack_batchsize = self._cfg.adc.batchsize
                # scale alpha with adc batchsize
                alpha = self._cfg.adc.adc_alpha * math.sqrt(
                    attack_batchsize / 1000
                )
                loader = DataLoader(
                    idx_dataset,
                    batch_size=int(attack_batchsize),
                    shuffle=False,
                    num_workers=0,
                )

                torch.cuda.empty_cache()  # avoid memory leak
                attack = Attack(
                    self._old_net,
                    alpha,
                    loader,
                    protos,
                    x_min.to(self._device),
                    x_max.to(self._device),
                    class_idx,
                    self._cfg.adc,
                )

                x_, y_, d_, m_ = attack.run()
                x_, y_, d_ = x_[m_], y_[m_], d_[m_]
                if len(x_) > 0 and class_idx > 0 and (class_idx + 1) % 10 == 0:
                    self._logger.info(
                        f"successful attacks class {class_idx} -> "
                        f"{len(x_)}, {d_.min().item():.3f} "
                        f"to {d_.median().item():.3f} "
                        f"from {len(x_top)} data"
                    )
                adc_dists.append(
                    [
                        d_.min().item(),
                        d_.mean().item(),
                        d_.max().item(),
                        m_.sum().item(),
                    ]
                )
            else:
                x_, y_ = x_top, y_top
            new_proto, new_cov, _ = self._adc_transfer(x_, y_, class_idx)
            new_protos.append(new_proto)
            new_covs.append(new_cov)
            torch.cuda.empty_cache()  # avoid memory leak
        grouped_dists = self._summarize_attack(adc_dists, type="ADC")
        self._custom_metrics["adc_dists"].append(grouped_dists)
        for cls_idx in range(self._known_classes):
            self._prototypes[cls_idx]["proto"] = new_protos[cls_idx]
            self._prototypes[cls_idx]["cov"] = new_covs[cls_idx]
        if self._cfg.adc.attack:
            del attack
        del dataset

    def _summarize_attack(self, dist_stats: list, type: str) -> list:
        """Log the distance stats of the adversarial attack result.

        Args:
            dist_stats (list): Distance stats between selected
                    samples' features and the target prototype,
                    min, mean, max, number of samples.
            type (str): Attack type for logging, "ADC" or "APR".

        Returns:
            list: Mean distance for each class group.
        """

        dist_array = np.array(dist_stats)
        self._logger.info(
            f"[{type}] completed with avg. dist "
            f"{np.mean(dist_array[:, 1]):.4f}, "
            f"min {np.min(dist_array[:, 0]):.4f} "
            f"~ max {np.max(dist_array[:, 2]):.4f} "
            f"N: {np.mean(dist_array[:, 3])}"
        )
        grouped_dists = []
        grouped_num_masked = []
        prev_cls = 0
        log_str = f"{type} avg. dist: "
        for increment in self.increments:
            grouped_dists.append(
                np.mean(dist_array[prev_cls : prev_cls + increment, 1])
            )
            grouped_num_masked.append(
                np.mean(dist_array[prev_cls : prev_cls + increment, 3])
            )
            log_str += (
                f"{grouped_dists[-1]:.4f} " + f"({grouped_num_masked[-1]}), "
            )
            prev_cls += increment
            if prev_cls >= self._known_classes:
                break
        self._logger.info(log_str)
        return grouped_dists

    def _adc_transfer(
        self, x: torch.Tensor, y: torch.Tensor, class_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Mitigate effects of semantic drift for prototypes
        and covariance matrices using pseudo-old samples

        Args:
            x (torch.Tensor): Pseudo old samples generated by
                adversarial attack, shape (N, 3, H, W), where
                N is the number of successful attack results.
            y (torch.Tensor): Target class id tensor.
            class_idx (int): Target class id to select prototype
                and covariance matrix.

        Returns:
            tuple[torch.Tensor, torch.Tensor, float]:
                new_proto: Calibrated prototype mean feature, shape (dim).
                new_cov:  Calibrated covariance matrix, shape (dim, dim).
                gap: Averaged feature gap between old and new prototypes.
        """
        idx_dataset = TensorDataset(x, y)
        idx_loader = DataLoader(idx_dataset, batch_size=16, shuffle=False)
        vectors_old = self._adc_inference(idx_loader, old_net=True)
        vectors = self._adc_inference(idx_loader, old_net=False)
        MU = self._prototypes[class_idx]["proto"]
        gap = (vectors - vectors_old).mean(dim=0)
        new_proto = MU + gap
        layer = self._learn_trans(
            vectors_old,
            vectors,
        )
        new_cov = self._transfer_cov(layer, self._prototypes[class_idx]["cov"])
        return new_proto, new_cov, gap.mean().item()

    def _learn_trans(
        self, vectors_old_all: torch.Tensor, vectors_all: torch.Tensor
    ) -> nn.Module:
        """Tran a linear layer that transforms old
        feature vectors to new ones.

        Args:
            vectors_old_all (torch.Tensor): Features yielded
                with the pseudo samples and old-task network.
            vectors_all (torch.Tensor):  Features yielded
                with the pseudo samples and new-task network.

        Returns:
            nn.Module: SimpleLinear layer.
        """
        idx_dataset = TensorDataset(vectors_old_all, vectors_all)
        idx_loader = DataLoader(idx_dataset, batch_size=16, shuffle=True)

        layer = SimpleLinear(self._net.net.out_dim, self._net.net.out_dim)
        layer.weight = nn.Parameter(torch.eye(self._net.net.out_dim))
        layer.bias = nn.Parameter(torch.zeros(self._net.net.out_dim))
        layer.to(self._device)
        optimizer = torch.optim.Adam(
            layer.parameters(),
            lr=self._cfg.adc.trans_lr,
        )
        for _ in range(self._cfg.adc.trans_epochs):
            for vectors_old, vectors in idx_loader:
                trans_features = layer(vectors_old)  # train mode
                loss = self._trans_loss(vectors, trans_features)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return layer

    def _add_proto_noise(self, raw_protos: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to the target prototype
        to augment the result of adversarial attack.
        Radius (`rad`) is based on covariance stats.
        https://github.com/Impression2805/CVPR21_PASS/blob/main/PASS.py#L170

        Args:
            raw_protos (torch.Tensor): Original prototype tensor,
                shape (N_batch, dim).

        Returns:
            torch.Tensor: Augmented prototype tensor,
                shape (N_batch, dim).
        """
        rad = torch.sqrt(
            sum(
                [
                    torch.trace(p["cov"]) / p["cov"].shape[0]
                    for p in self._prototypes
                ]
            )
            / len(self._prototypes)
        )
        noise = (
            torch.normal(0, 1, size=raw_protos.shape).to(self._device) * rad
        )
        protos = raw_protos + self._cfg.apr.proto_noise_mag * noise
        return protos

    def _perturb(
        self, src_x: torch.Tensor, src_y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Perturb source image data to be close to
        the target prototype.

        Args:
            src_x (torch.Tensor): Source image data.
            src_y (torch.Tensor): Target labels.

        Returns:
            tuple[torch.Tensor, torch.Tensor, float]:
                x: Perturbed image data.
                y: Target labels.
                dist: Batch sum of feature distances
                    after perturbation.
        """
        x_min = self._apr_x_min
        x_max = self._apr_x_max
        x = src_x.clone()
        L = nn.MSELoss()
        proto_targets = torch.stack(
            [self._prototypes[ind]["proto"] for ind in src_y]
        )
        # add noise to proto targets
        if self._cfg.apr.proto_noise_mag > 0:
            proto_targets = self._add_proto_noise(proto_targets)
        for _ in range(self._cfg.apr.loops):
            x.requires_grad = True
            lr = self._cfg.apr.perturb_alpha
            feats = self._old_net(x)["features"]
            loss = L(
                feats,
                proto_targets,
            )
            self._old_net.zero_grad()
            # calculate gradients
            loss.backward()
            grad = x.grad
            x = x - (lr * grad / torch.norm(grad, keepdim=True))
            if self._cfg.apr.clamp_x:
                x = torch.clamp(x, x_min, x_max)
            x = x.detach()
        dist = F.pairwise_distance(feats.detach(), proto_targets)
        y = src_y
        torch.cuda.empty_cache()  # avoid memory leak
        return x, y, dist.sum().item()

    def _apr_extract_closest(
        self,
    ) -> tuple[np.ndarray, dict[list], list[list]]:
        """Before the new task, select the top-k closest samples
        to the class prototype, for all the known (old) classes.
        Random transform parameters are recorded to be reproduced
        during new task training.

        Returns:
            tuple[np.ndarray, dict[list], list[list]]:
                all_data: Selected indices of the new task data.
                all_params: Recorded random transform parameters.
                apr_dists: Distance stats between selected
                    samples' features and the target prototype,
                    min, mean, max, number of samples.
        """
        dataset = self._draw_dataset(
            self._cur_total_classes - self._known_classes,
            "apr",
        )
        self._logger.info(
            f"[APR] current total classes: {self._cur_total_classes}, "
            f"known classes: {self._known_classes}, "
            f"dataset for _apr_extract_closest (mode = {dataset.mode}) "
            f"prepared, labels: {min(dataset.labels[dataset.indices])}"
            f" ~ {max(dataset.labels[dataset.indices])}"
        )
        dataset.check_old_data(self._known_classes)
        loader = DataLoader(
            dataset,
            64,
            shuffle=False,
            num_workers=self._cfg.data.train.num_workers,
        )
        for k, (data, _, _) in enumerate(loader):
            if k == 0:
                x_min = data.min()
                x_max = data.max()
            else:
                if data.min() < x_min:
                    x_min = data.min()
                if data.max() > x_max:
                    x_max = data.max()
        self._apr_x_min = x_min.to(self._device)
        self._apr_x_max = x_max.to(self._device)
        feats = []
        params = []  # transform results
        for data, _, batch_params in loader:
            data = data.to(self._device)
            out = self._old_net(data)
            feats.append(out["features"])
            params.append(batch_params)
        feats = torch.cat(feats, dim=0)
        params_dict = {}
        for k in params[0].keys():
            if isinstance(params[0][k], list):
                params_dict[k] = torch.cat(
                    [torch.stack(p[k], dim=1) for p in params]
                )
            else:
                params_dict[k] = torch.cat([p[k] for p in params])

        protos = [p["proto"] for p in self._prototypes[: self._known_classes]]
        all_data = []
        all_params = {k: [] for k in params_dict}
        apr_dists = []
        for class_idx in range(0, self._known_classes):
            d = torch.cdist(feats, protos[class_idx].unsqueeze(0)).squeeze()
            sample_inds = torch.argsort(d)[: self._cfg.apr.apr_sample_limit]
            # construct pseudo data
            for ind in sample_inds:
                all_data.append(ind.item())
                for k in params_dict:
                    all_params[k].append(params_dict[k][ind].item())
            d_ = torch.tensor([d[idx] for idx in sample_inds])
            apr_dists.append(
                [d_.min().item(), d_.mean().item(), d_.max().item(), len(d_)]
            )
            torch.cuda.empty_cache()  # avoid memory leak
        all_data = np.asarray(all_data, dtype=np.int64)
        return all_data, all_params, apr_dists

    def _build_apr_dataloader(self):
        """Build the APR dataloader from the selected
        dataset by `_apr_extract_closest`.
        """
        (
            all_data,
            all_params,
            apr_dists,
        ) = self._apr_extract_closest()
        grouped_dists = self._summarize_attack(apr_dists, type="APR")
        self._custom_metrics["apr_dists"].append(grouped_dists)
        dataset = self._draw_dataset(
            self._cur_total_classes - self._known_classes,
            "apr",
        )
        self._logger.info(
            f"[APR] current total classes: {self._cur_total_classes}, "
            f"known classes: {self._known_classes}, "
            f"dataset for APR (mode = {dataset.mode}) prepared,"
            f" labels: {min(dataset.labels[dataset.indices])} "
            f"~ {max(dataset.labels[dataset.indices])}"
        )
        if self._cfg.apr.do_reproducible_trans:
            dataset.register_rep_params(
                all_params, all_data, self._cfg.apr.apr_sample_limit
            )
        dataset.check_old_data(self._known_classes)
        self.p_dataloader = DataLoader(
            dataset,
            self._cfg.apr.p_batchsize,
            shuffle=True,
            num_workers=self._cfg.data.train.num_workers,
            drop_last=True,
        )
        self.iter_p_dataloader = iter(self.p_dataloader)
