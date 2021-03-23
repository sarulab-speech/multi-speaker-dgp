import abc
from typing import Optional

import numpy as np
import torch

from .parameter_transform import positive
from .type_config import NUMPY_DTYPE

INITIAL_FLOOR = 1e-4
JITTER = 1e-6
ARCCOS_EPS = 1e-4


class Kernel(torch.nn.Module, metaclass=abc.ABCMeta):
    """
    The basic kernel class.
    """

    @abc.abstractmethod
    def K(self, x1, x2=None):
        raise NotImplementedError

    @abc.abstractmethod
    def K_diag(self, x1):
        raise NotImplementedError


class RBF(Kernel):
    r"""RBF kernel.

    .. math::
        k(x, x') &= m \exp[(x - x')^T L^{-1} (x - x')] + f \\
        L &= {\rm diag}(l) = {\rm diag}[l_1^2, l_2^2, ... l_D^2]
    where
        - $l$: lengthscale
        - $m$: magnitude
        - $f$: flooring value

    Args:
        input_dim (int): Dimension of input.
        initial_lengthscale (float, optional): Initial lengthscale common among all dims.
    """

    def __init__(self,
                 input_dim: int,
                 initial_lengthscale: Optional[float] = 1.0):

        super().__init__()

        lengthscale_init_val = positive.backward(initial_lengthscale) * np.ones(input_dim)
        magnitude_init_val = positive.backward(np.array([1.0]))
        floor_init_val = positive.backward(np.array([INITIAL_FLOOR]))

        self.lengthscale_ = torch.nn.Parameter(torch.from_numpy(lengthscale_init_val.astype(NUMPY_DTYPE)))
        self.magnitude_ = torch.nn.Parameter(torch.from_numpy(magnitude_init_val.astype(NUMPY_DTYPE)))
        self.floor_ = torch.nn.Parameter(torch.from_numpy(floor_init_val.astype(NUMPY_DTYPE)))

        self._jitter = JITTER

    @property
    def lengthscale(self):
        return positive.forward_tensor(self.lengthscale_)

    @property
    def magnitude(self):
        return positive.forward_tensor(self.magnitude_)

    def K(self, x1, x2=None):
        lengthscale = positive.forward_tensor(self.lengthscale_)
        magnitude = positive.forward_tensor(self.magnitude_)
        floor = positive.forward_tensor(self.floor_)

        if x2 is None:
            x2 = x1
            sym = True
        else:
            sym = False

        beta = 1.0 / lengthscale ** 2.0
        xx1 = torch.sum((beta * x1 ** 2.0), dim=1, keepdim=True)
        xx2 = torch.sum((beta * x2 ** 2.0), dim=1, keepdim=True).t()
        x1x2 = torch.mm(beta * x1, x2.t())

        D = xx1 + xx2 - 2.0 * x1x2

        if sym:
            K = magnitude * torch.exp(-0.5 * D) + (floor + self._jitter) * torch.eye(D.size(0), dtype=D.dtype, device=D.device)
        else:
            K = magnitude * torch.exp(-0.5 * D)

        return K

    def K_diag(self, x1):
        magnitude = positive.forward_tensor(self.magnitude_)
        floor = positive.forward_tensor(self.floor_)

        return magnitude * torch.ones_like(x1[:, 0]) + floor + self._jitter


class ArcCos(Kernel):
    r"""
    A kernel function derived from neural network with an infinite number
    of hidden units and ReLU activation function.

    .. math::
        k_0(x, x') &= \sigma_{b,0}^2  + \sigma_{w,0}^2 x^T L x' \\
        k_{p+1}(x, x') &= \sigma_{b, p+1}^2  + \sigma_{w, p+1}^2 \sqrt{k_p(x, x)} \sqrt{k_p(x', x')} (\sin \theta_p + (\pi - \theta_p) \cos \theta_p)  \\
        \theta_p &= \arccos \frac{k_p(x, x')}{\sqrt{k_p(x, x)} \sqrt{k_p(x', x')}} \\
        L &= diag(l) = diag[l_1^2, l_2^2, ... l_D^2]

    where
        - $L$: relevance
        - $\sigma_w$: variance of NN weight prior
        - $\sigma_b$: variance of NN bias prior

    Args:
        input_dim (int): Dimension of input.
        num_layers (int, optional): Number of layers of NN, default to 1.
        initial_relevance (float, optional): This value is devided by input_dim, default to 1.0.
    """

    def __init__(self,
                 input_dim: int,
                 num_layers: int = 1,
                 initial_relevance: float = 1.0,
                 initial_sigma_w: list = None,
                 initial_sigma_b: float = 0.01,
                 normalize: bool = True,
                 fix_relevance: bool = False,
                 fix_sw: bool = False,
                 fix_sb: bool = False,
                 fix_magnitude: bool = False,
                 fix_floor: bool = False):
        super().__init__()

        assert num_layers > 0, "Num layers mush be positive integer."

        if initial_sigma_w is None:
            # use He initialization
            initial_sigma_w = [2.0 / input_dim] + [1.0] * num_layers

        relevance_init_val = positive.backward(initial_relevance) * np.ones(input_dim)
        sigam_w_init_val = positive.backward(initial_sigma_w * np.ones(num_layers + 1))
        sigam_b_init_val = positive.backward(initial_sigma_b * np.ones(num_layers + 1))

        magnitude_init_val = positive.backward(np.array([1.0]))
        floor_init_val = positive.backward(np.array([INITIAL_FLOOR]))

        self.relevance_ = torch.nn.Parameter(torch.from_numpy(relevance_init_val.astype(NUMPY_DTYPE)), requires_grad=(not fix_relevance))
        self.sigma_w_ = torch.nn.Parameter(torch.from_numpy(sigam_w_init_val.astype(NUMPY_DTYPE)), requires_grad=(not fix_sw))
        self.sigma_b_ = torch.nn.Parameter(torch.from_numpy(sigam_b_init_val.astype(NUMPY_DTYPE)), requires_grad=(not fix_sb))

        self.magnitude_ = torch.nn.Parameter(torch.from_numpy(magnitude_init_val.astype(NUMPY_DTYPE)), requires_grad=(not fix_magnitude))
        self.floor_ = torch.nn.Parameter(torch.from_numpy(floor_init_val.astype(NUMPY_DTYPE)), requires_grad=(not fix_floor))

        self._num_layers = num_layers
        self._normalize = normalize

        self._jitter = JITTER

    def K(self, x1, x2=None):
        relevance = positive.forward_tensor(self.relevance_)
        sigma_w = positive.forward_tensor(self.sigma_w_)
        sigma_b = positive.forward_tensor(self.sigma_b_)

        magnitude = positive.forward_tensor(self.magnitude_)
        floor = positive.forward_tensor(self.floor_)

        if x2 is None:
            x2 = x1
            sym = True
        else:
            sym = False

        x1x2 = torch.mm(relevance * x1, x2.t())
        xx1 = torch.sum(relevance * (x1 ** 2.0), dim=1, keepdim=True)
        xx2 = torch.sum(relevance * (x2 ** 2.0), dim=1, keepdim=True).t()

        Ki0 = sigma_b[0] + sigma_w[0] * x1x2
        k1diag_sqrt = torch.sqrt(sigma_b[0] + sigma_w[0] * xx1)
        k2diag_sqrt = torch.sqrt(sigma_b[0] + sigma_w[0] * xx2)

        Ki = Ki0

        for i in range(self._num_layers):
            sb = sigma_b[i + 1]
            sw = sigma_w[i + 1]

            dd = torch.mm(k1diag_sqrt, k2diag_sqrt)
            div = Ki / dd
            div = ARCCOS_EPS + (1 - 2.0 * ARCCOS_EPS) * div  # to restrict "div" in [-1, 1]
            theta = torch.acos(div)
            Ki = sb + sw / (2.0 * np.pi) * dd * (torch.sin(theta) + (np.pi - theta) * torch.cos(theta))

            k1diag_sqrt = torch.sqrt(sb + sw / 2.0 * k1diag_sqrt ** 2.0)
            k2diag_sqrt = torch.sqrt(sb + sw / 2.0 * k2diag_sqrt ** 2.0)

        if self._normalize:
            Ki = Ki / torch.mm(k1diag_sqrt, k2diag_sqrt)

        if sym:
            K = magnitude * Ki + (floor + self._jitter) * torch.eye(Ki.size(0), dtype=Ki.dtype, device=Ki.device)
        else:
            K = magnitude * Ki

        return K

    def K_diag(self, x1):

        magnitude = positive.forward_tensor(self.magnitude_)
        floor = positive.forward_tensor(self.floor_)

        if self._normalize:
            return magnitude * torch.ones_like(x1[:, 0]) + floor + self._jitter

        else:
            relevance = positive.forward_tensor(self.relevance_)
            sigma_w = positive.forward_tensor(self.sigma_w_)
            sigma_b = positive.forward_tensor(self.sigma_b_)

            xx1 = torch.sum(relevance * (x1 ** 2.0), dim=1)
            k1diag = sigma_b[0] + sigma_w[0] * xx1
            for i in range(self._num_layers):
                sb = sigma_b[i + 1]
                sw = sigma_w[i + 1]

                k1diag = sb + sw / 2.0 * k1diag

            return magnitude * k1diag + floor + self._jitter
