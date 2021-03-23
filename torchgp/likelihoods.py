import math

import numpy as np
import torch

from .parameter_transform import positive
from .type_config import NUMPY_DTYPE


class Gaussian(torch.nn.Module):
    r"""
    Gaussian type likelihood.
    The relationship between output variable $y$ and latent function $f$ is given by
        $p(y|f) = N(y; f, \sigma^2)$
    where parameter $\sigma^2$ is the variance of noise.

    Args:
        output_dim (int): Dimension of outputs.
        initial_variance (float, optional): Initial variance common among all dimensions.
        share_dimension (bool, optional): If true, use the same parameter for all dimensions.
    """

    def __init__(self, output_dim, initial_variance=1e-3, share_parameter=False):
        super().__init__()

        self.share_parameter = share_parameter
        self.output_dim = output_dim

        if self.share_parameter:
            variance_init_val = positive.backward(initial_variance) * np.ones(1)
        else:
            variance_init_val = positive.backward(initial_variance) * np.ones(output_dim)

        self.variance_ = torch.nn.Parameter(torch.from_numpy(variance_init_val.astype(NUMPY_DTYPE)))

    @property
    def variance(self):
        return positive.forward_tensor(self.variance_)

    def predictive_expectation(self, y, f_stats):
        r"""Calculate predictive expectation of log likelihoods.

        With

        .. math::
            p(y|f) &= N(y; f, \sigma^2) \\
            q(f) &= N(f; \mu_f, \Sigma_f)

        , predictive expectation is given by

        .. math::
            & \int q(f) \log p(y|f) df \\
            &= \int q(f) \left(-\frac{N}{2} \log(2 \pi) - \frac{N}{2} \log(v)
                - \frac{(y-f)^T (y-f)}{2\sigma^2} \right) df \\
            &= -\frac{N}{2} \log(2 \pi) - \frac{N}{2} \log(\sigma^2)
               - \frac{(y - \mu_f)^T (y - \mu_f)}{2\sigma^2}
               - \frac{Tr[\Sigma_f]}{2\sigma^2}

        In multidimentional cases, we sum up log likelihoods of all dims.
        """

        if self.share_parameter:
            variance = positive.forward_tensor(self.variance_).expand(self.output_dim)
        else:
            variance = positive.forward_tensor(self.variance_)

        size_B = y.size(0)
        size_D = y.size(1)

        diff = y - f_stats['mean']
        pe1 = -0.5 * size_B * size_D * np.log(2.0 * np.pi)
        pe2 = -0.5 * size_B * torch.sum(torch.log(variance))
        pe3 = -0.5 * torch.sum(diff * diff / variance)
        pe4 = -0.5 * torch.sum(f_stats['var'] / variance)

        return pe1 + pe2 + pe3 + pe4

    def predict(self, f_stats):
        r"""Calculate predictive distribution of output variable $y$

        .. math::
            q(y) = \int q(f) p(y | f) df
        """

        variance = positive.forward_tensor(self.variance_)

        y_var = f_stats['var'] + variance.expand_as(f_stats['var'])
        return {'mean': f_stats['mean'], 'var': y_var}
