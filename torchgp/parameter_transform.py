import torch
import numpy as np


class Logistic:
    """
    Use for parameter which is in range (0, 1).
    """

    def __init__(self):
        pass

    def forward(self, x):
        """
        Convert real numpy number to (-1, 1) using sigmoid.
        """
        return 1.0 / (1.0 + np.exp(-x))

    def forward_tensor(self, x):
        """
        Convert real number tensor to positive.
        """
        return torch.sigmoid(x)

    def backward(self, y):
        """
        Convert positive numpy number to real.
        """

        return -np.log(1.0 / y - 1.0)


logistic = Logistic()


class Positive:
    """
    Use for parameter which is always positive.
    """

    def __init__(self, lower=1e-6):
        self._lower = lower

    def forward(self, x):
        """
        Convert real numpy number to positive.
        """
        return np.logaddexp(0, x) + self._lower

    def forward_tensor(self, x):
        """
        Convert real number tensor to positive.
        """
        return torch.nn.functional.softplus(x) + self._lower

    def backward(self, y):
        """
        Convert positive numpy number to real.
        """
        ys = np.maximum(y - self._lower, np.finfo(np.float32).eps)
        return ys + np.log(-np.expm1(-ys))


positive = Positive()
