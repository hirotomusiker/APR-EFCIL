"""
Adapted from:
https://github.com/dipamgoswami/ADC/blob/main/utils/attack.py
"""
import torch
from omegaconf.dictconfig import DictConfig
from torch import nn
from torch.utils.data import DataLoader


class Attack(object):
    def __init__(
        self,
        old_model: nn.Module,
        alpha: float,
        loader: DataLoader,
        protos: list[torch.Tensor],
        x_min: torch.Tensor,
        x_max: torch.Tensor,
        target_cls_id: int,
        cfg: DictConfig,
    ):
        """Adversarial attack to resurrect old data.

        Args:
            old_model (nn.Module): Frozen old task model.
            alpha (float): Attack magnitude.
            loader (DataLoader): Class-specific data loader.
            protos (list[torch.Tensor]): Mean features of the known classes.
            x_min (torch.Tensor): Minimum image data value for clipping.
            x_max (torch.Tensor): Maximum image data value for clipping.
            target_cls_id (int): Class id of the attack target prototype.
            cfg (DictConfig): Config for ADC.
        """
        self.old_model = old_model
        self.alpha = alpha
        self.loader = loader
        self.device = x_min.device
        self.target_cls_id = target_cls_id
        self.epochs = cfg.epochs
        self.target_proto = protos[target_cls_id]
        self.protos = torch.stack(protos)
        self.clamp_x = cfg.clamp_x
        self.x_min = x_min
        self.x_max = x_max

    def perturb(
        self,
        x: torch.Tensor,
        alpha: float,
        grad: torch.Tensor,
        x_min: torch.Tensor,
        x_max: torch.Tensor,
    ) -> torch.Tensor:
        """Perturb the input image data with backpropagated gradient.

        Args:
            x (torch.Tensor): Target (input) image data,
                shape (batchsize, 3, H, W).
            alpha (float): Attack magnitude.
            grad (torch.Tensor): Propagated gradient,
                shape (batchsize, 3, H, W).
            x_min (torch.Tensor): Minimum image data value for clipping.
            x_max (torch.Tensor): Maximum image data value for clipping.

        Returns:
            torch.Tensor: Resulting image data, shape (batchsize, 3, H, W).
        """
        x_prime = x - (alpha * grad / torch.norm(grad, keepdim=True))
        if self.clamp_x:
            x_prime = torch.clamp(
                x_prime, x_min.to(self.device), x_max.to(self.device)
            )
        return x_prime

    def _train_one_epoch(self, data: torch.Tensor) -> torch.Tensor:
        """Perturb the input image data.

        Args:
            data (torch.Tensor): Target (input) image
                data, shape (batchsize, 3, H, W).

        Returns:
            torch.Tensor: Perturbed image data,
                shape (batchsize, 3, H, W).
        """
        data.requires_grad = True
        feats = self.old_model(data)["features"]
        proto_targets = self.target_proto.expand(
            data.shape[0], self.target_proto.shape[0]
        ).to(self.device)
        L = nn.MSELoss()
        loss = L(feats, proto_targets)
        # zero out all existing gradients
        self.old_model.zero_grad()
        # calculate gradients
        loss.backward()
        data_grad = data.grad

        perturbed_data = self.perturb(
            data, self.alpha, data_grad, self.x_min, self.x_max
        )
        return perturbed_data.detach()

    def _evaluate(
        self,
        model: torch.nn.Module,
        perturbed_data: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate whether the perturbed image data
        can be classified as the target class

        Args:
            model (torch.nn.Module): Frozen old-task model
            perturbed_data (torch.Tensor): Perturbed image
                data, shape (batchsize, 3, H, W).
            target (torch.Tensor): Target class labels,
                shape (batchsize,).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                mask: Bool data for each image whether the attack is
                    successful, shape (batchsize,).
                mindist: Distances between image feature and
                    the closest prototype, shape (batchsize,).
        """

        with torch.no_grad():
            adv_output = model(perturbed_data)["features"]
        d = torch.cdist(adv_output, self.protos)
        adv_pred = torch.argmin(d, dim=1)
        mindist = d.min(dim=1).values
        mask = adv_pred == target
        return mask, mindist

    def run(self) -> tuple[torch.Tensor, ...]:
        """Do adversarial attack loop.

        Raises:
            TimeoutError: Attack does not converge with 100 epochs.

        Returns:
            tuple[torch.Tensor, ...]:
                p_data: Resulting image data, shape (N, 3, H, W).
                p_label: Target (label) data, shape (N,).
                p_dist: Distances between p_data and the closest prototype.
                p_mask: Bool data for each image whether the attack is
                    successful or not.
                N is the number of new-task data used for this attack,
                specified by `adc.adc_sample_limit` in config.
        """
        p_data, p_label, p_dist, p_mask = [], [], [], []
        for data, label in self.loader:
            data, label = data.to(self.device), label.to(self.device)
            target = (
                torch.tensor(self.target_cls_id)
                .expand(len(label))
                .to(self.device)
            )
            epoch = 0
            while True:
                data = self._train_one_epoch(data)
                epoch += 1
                if epoch % self.epochs == 0:
                    mask, mindist = self._evaluate(
                        self.old_model,
                        data,
                        target,
                    )
                    success = mask.float().sum()
                    if success > 0:
                        p_data.append(data)
                        p_label.append(target)
                        p_dist.append(mindist)
                        p_mask.append(mask)
                        break
                if epoch > 100:
                    raise TimeoutError(f"{epoch} ADC epochs but unsuccessful")
        return (
            torch.cat(p_data, 0),
            torch.cat(p_label, 0),
            torch.cat(p_dist, 0),
            torch.cat(p_mask, 0),
        )
