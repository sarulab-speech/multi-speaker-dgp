from typing import Optional

import numpy as np
import torch

from . import torch_matrix_utils as tmu
from .kernels import Kernel
from .likelihoods import Gaussian
from .parameter_transform import positive
from .type_config import NUMPY_DTYPE

VAR_FLOOR = 1e-4


class SVGP(torch.nn.Module):
    """Stochastic Variational inference of Gaussian Process.
    """

    def __init__(self,
                 kernel: Kernel,
                 input_dim: int,
                 output_dim: int,
                 num_inducings: Optional[int] = None,
                 inducing_variable: Optional[np.ndarray] = None,
                 initial_q_S_value: Optional[float] = 1.0,
                 fix_inducing: Optional[bool] = False):
        """Initialize SVGP object. This method only works with Gaussian likelihood.

        Args:
            kernel: TorchGP Kernel object.
            input_dim: Dimension of inputs, D_in.
            output_dim: Dimension of outputs, D_out.
            num_inducings (int, optional): Number of inducing points, M.
            inducing_variable (numpy.ndarray, optional): matrix of inducing points, size M (inducing points) x D_in (input dimensions).
                By default samples from standard normal distribution.
            initial_q_S_value (float, optional): Initial variance of q(u) (variational distribution of u).
            fix_inducing (bool, optional): When False, optimize inducing inputs.
        """
        super(SVGP, self).__init__()

        self.kernel = kernel
        self.output_dim = output_dim

        if (inducing_variable is None) == (num_inducings is None):
            raise ValueError('BayesianGPLVM needs exactly one of `inducing_variable` and `num_inducings`')

        if inducing_variable is None:
            # By default we initialize by samples from standard normal distribution.
            inducing_variable = np.random.randn(num_inducings, input_dim)
        else:
            num_inducings = len(inducing_variable)
        self.Z = torch.nn.Parameter(torch.tensor(inducing_variable.astype(NUMPY_DTYPE)),
                                    requires_grad=(not fix_inducing))  # (M, D_in)

        self.q_mu = torch.nn.Parameter(torch.zeros(num_inducings, output_dim))  # (M, D_out)
        self.q_diag_S_ = torch.nn.Parameter(positive.backward(initial_q_S_value * torch.ones_like(self.q_mu)))  # (M, D_out)

    @property
    def q_diag_S(self):
        return positive.forward_tensor(self.q_diag_S_)

    def forward(self, x):
        """Forward propagation.

        This function samples N points from predicted distribution using
        reparameterization trick while training.
        Otherwise it returns predicted mean.

        Args:
            x (torch.Tensor): Data matrix, size N (number of points) x D_in (input dimensions).

        Returns:
            torch.Tensor: Output of layer, size N (number of points) x D_out (output dimensions).
        """
        if self.training:
            return self.sample(x)
        else:
            return self.pred_mean_and_var(x)['mean']

    def pred_mean_and_var(self, x):
        r"""
        Predict $q(f|x) = \int q(f|x,u) q(u) du$
        where $q(u) =  N(u; m, S)$.  
        First,

        .. math::
            q(f|x,u) &= N(f; \mu, \Sigma) \\
            \mu &= K_{NM} K_{MM}^{-1} u \\
            \Sigma  &= K_{NN} - K_{NM} K_{MM}^{-1} K_{MN}

        Then,

        .. math::
            q(f|x) &= N(f, \mu_f, \Sigma_f) \\
            \mu_f &= K_{NM} K_{MM}^{-1} m \\
            \Sigma_f &= K_{NN} - K_{NM} K_{MM}^{-1} K_{MN} 
                + K_{NM} K_{MM}^{-1} S K_{MM}^{-1} K_{MN}

        Args:
            x (torch.Tensor): Data matrix, size N (number of points) x D_in (input dimensions).

        Returns:
            dict: mean and variance of predicted distribution.
        """
        Kmn = self.kernel.K(self.Z, x)  # (M, N)
        cho_Kmm = tmu.cholesky(self.kernel.K(self.Z))  # (M, M)
        Knn_diag = self.kernel.K_diag(x)  # (N, )

        pred_mu = tmu.cho_solve_AXB(Kmn.T, cho_Kmm, self.q_mu)  # (N, D_out)

        MM = tmu.cho_solve(cho_Kmm, Kmn)  # K_{MM}^{-1} K_{MN}

        s1 = Knn_diag[:, None].expand(-1, self.output_dim)  # (N, D_out)
        s2 = torch.sum(Kmn * MM, dim=0)[:, None].expand(-1, self.output_dim)
        s3 = torch.mm((MM * MM).T, self.q_diag_S)

        pred_var = torch.clamp(s1 - s2 + s3, min=VAR_FLOOR)  # (N, D_out)

        return {'mean': pred_mu, 'var': pred_var}

    def sample(self, x):
        """Sample N points from predicted distribution.

        Args:
            x (torch.Tensor): Data matrix, size N (number of points) x D_in (input dimensions).

        Returns:
            torch.Tensor: sampled points, size N (number of points) x D_out (output dimensions).
        """
        pred = self.pred_mean_and_var(x)
        return pred['mean'] + torch.randn_like(pred['mean']) * torch.sqrt(pred['var'])

    def kl_divergence(self):
        r"""
        Calc Kullback-Leibler divergence $KL(q(u)||p(u))$
        where

        .. math::
            p(u) &= N(u; 0, K_{MM}) \\
            q(u) &= N(u; m, S)

        Since p and q are both Gaussian, KL divergence can be calculated analytically.

        .. math::
            KL &= \int q(u) (\log(q(u) - \log p(u)) du \\
            &= \int q(u) \left(-\frac{1}{2} \log|S| - \frac{1}{2} (u - m)^T S^{-1} (u - m) \right. \\
                & \qquad\left. +\frac{1}{2} \log|K_{MM}| + \frac{1}{2} u^T K_{MM}^{-1} u \right) du \\
            &= -\frac{1}{2} \log|S| - \frac{1}{2} Tr(I_M)
                + \frac{1}{2} \log|K_{MM}| + \frac{1}{2} (m^T K_{MM}^{-1} m + Tr(K_{MM}^{-1} S)) \\
            &= -\frac{1}{2} \log|S| -\frac{1}{2} M + \frac{1}{2} \log|K_{MM}|
                + \frac{1}{2} m^T K_{MM}^{-1} m + \frac{1}{2} Tr(K_{MM}^{-1} S))

        In multidimentional case, we sum up KLDs of all dims.

        Returns:
            torch.Tensor: Kullback-Leibler divergence.
        """
        M, D = self.q_mu.shape
        cho_Kmm = tmu.cholesky(self.kernel.K(self.Z))

        KL = -0.5 * torch.sum(torch.log(self.q_diag_S))
        KL -= 0.5 * M * D
        KL += 0.5 * D * tmu.cho_log_det(cho_Kmm)
        KL += 0.5 * torch.trace(tmu.cho_solve_AXB(self.q_mu.T, cho_Kmm, self.q_mu))
        KL += 0.5 * torch.sum(self.q_diag_S.T * torch.diag(tmu.cho_inv(cho_Kmm)))

        return KL
