import torch


def cholesky(X):
    """Cholesky decomposition of $X$.

    >>> cholesky(X)
    return L
    torch.mm(L, L.t()) = X
    """

    if torch.__version__ >= '1.0.0':
        return torch.cholesky(X)
    else:
        return torch.potrf(X, upper=False)


def cho_log_det(L):
    r"""Compute tensor log determinant of $C$ from cholesky factor.

    Args:
        L (torch.Tensor): Lower triangular tensor where $L L^T = C$.

    Returns:
        torch.Tensor: $\log |C|$
    """

    return 2.0 * torch.sum(torch.log(torch.diagonal(L)))


def cho_inv(L):
    """Compute tensor $C^{-1}$ from cholesky factor.

    Args:
        L (torch.Tensor): Lower triangular tensor where $L L^T = C$.

    Returns:
        torch.Tensor: $C^{-1}$
    """

    return cho_solve(L, torch.eye(L.size(0), dtype=L.dtype, device=L.device))


def cho_solve(L, b):
    """Compute tensor $C^{-1} b$ from cholesky factor.

    This function supports batched inputs.

    Args:
        L (torch.Tensor): (B x )N x N lower triangular tensor where $L L^T = C$.
        b (torch.Tensor): (B x )N x L tensor.

    Returns:
        torch.Tensor: $C^{-1} b$
    """

    tmp, _ = torch.triangular_solve(b, L, upper=False)
    tmp2, _ = torch.triangular_solve(tmp, torch.transpose(L, -2, -1), upper=True)
    return tmp2


def cho_solve_AXB(a, L, b):
    """Compute tensor $a C^{-1} b$ from cholesky factor.

    Args:
        a (torch.Tensor): M x N tensor.
        L (torch.Tensor): N x N lower triangular tensor where $L L^T = C$.
        b (torch.Tensor): N x L tensor.

    Returns:
        torch.Tensor: $a C^{-1} b$
    """

    left, _ = torch.triangular_solve(a.t(), L, upper=False)
    right, _ = torch.triangular_solve(b, L, upper=False)
    return torch.mm(left.t(), right)
