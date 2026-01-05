"""
Adapted from:
https://github.com/dipamgoswami/FeCAM/blob/main/models/base.py
"""
import torch
import torch.nn.functional as F


def shrink_cov(
    cov: torch.Tensor, gamma1: float = 1.0, gamma2: float = 1.0
) -> torch.Tensor:
    """Shrink covariance matrix.

    Args:
        cov (torch.Tensor): Input covariance matrix, shape (dim, dim).
        gamma1 (float, optional): Shrinkage paramater for in-diagonal
            values. Defaults to 1.0.
        gamma2 (float, optional): Shrinkage paramater for off-diagonal
            values. Defaults to 1.0.

    Returns:
        torch.Tensor: Output covariance matrix, shape (dim, dim).
    """
    diag_mean = torch.diagonal(cov).mean()
    off_diag = cov.clone()
    off_diag.fill_diagonal_(0.0)
    mask = off_diag != 0.0
    off_diag_mean = (off_diag * mask).sum() / mask.sum()
    iden = torch.eye(cov.shape[0]).to(cov.device)
    cov_ = (
        cov
        + (gamma1 * diag_mean * iden)
        + (gamma2 * off_diag_mean * (1 - iden))
    )
    return cov_


def mahalanobis(
    x: torch.Tensor,
    mu: torch.Tensor,
    cov: torch.Tensor,
    normalize: bool = True,
):
    """Compute Mahalanobis distance between input features and
    class mean feature / covariance matrix.

    Args:
        x (torch.Tensor): Input features, shape (N, dim).
        mu (torch.Tensor): Class mean feature (prototype), shape (dim,).
        cov (torch.Tensor): Class covariance matrix, shape (dim, dim).
        normalize (bool, optional): Whether to normalize covariance.
            Defaults to True.
    """
    x_minus_mu = F.normalize(x, p=2, dim=-1) - F.normalize(mu, p=2, dim=-1)
    # normalize cov
    if normalize:  # cov normalization
        sd = torch.sqrt(torch.diagonal(cov))
        cov = cov / (
            torch.matmul(sd.unsqueeze(1), sd.unsqueeze(0))
        )  # [512, 512]
    inv_covmat = torch.linalg.pinv(cov)
    mahal = x_minus_mu @ inv_covmat @ x_minus_mu.T
    mahal = torch.diagonal(mahal, 0)
    torch.cuda.empty_cache()
    return mahal


def svd_decompose(cov: torch.Tensor, k: int) -> tuple[torch.Tensor, ...]:
    """Decompose covariance matrix and select top-k.

    Args:
        cov (torch.Tensor): Input covariance matrix, shape (dim, dim).
        k (int): top-k dimensions.

    Returns:
        tuple[torch.Tensor, ...]:
            U_k: shape (dim, k), S_k: shape (k, k), VT_k: shape (k, dim).
    """
    U, S, VT = torch.linalg.svd(cov, full_matrices=True)
    U_k = U[:, :k]  # [dim, k]
    S_k = torch.diag(S[:k])  # [k, k]
    VT_k = VT[:k, :]  # [k, dim]
    return U_k, S_k, VT_k


def svd_compose(
    U_k: torch.Tensor, S_k: torch.Tensor, VT_k: torch.Tensor
) -> torch.Tensor:
    """Compose a full covariance matrix from decomposed matrices.

    Args:
        U_k (torch.Tensor): Shape (dim, k).
        S_k (torch.Tensor): Shape (k, k).
        VT_k (torch.Tensor): Shape (k, dim).

    Returns:
        torch.Tensor: Shape (dim, dim).
    """
    A = torch.matmul(S_k, VT_k)
    cov = torch.matmul(U_k, A)  # [dim, dim]
    return cov
