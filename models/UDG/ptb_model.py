from __future__ import absolute_import

import numpy as np
import torch
from torch import nn


class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].
    """

    def __init__(self, p=0.5, eps=1e-6):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0

    def reparameterize(self, mu, std):
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var() + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean) * self.factor
        sqrtvar_std = self.sqrtvar(std) * self.factor

        beta = self.reparameterize(mean, sqrtvar_mu)
        gamma = self.reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(
            x.shape[0], x.shape[1], 1, 1
        )
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(
            x.shape[0], x.shape[1], 1, 1
        )

        return x
