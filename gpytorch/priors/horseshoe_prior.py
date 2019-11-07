#!/usr/bin/env python3

import math
from numbers import Number

import torch
from gpytorch.priors.prior import Prior
from torch.distributions import constraints
from torch.nn import Module as TModule


class HorseshoePrior(Prior):
    """Horseshoe prior.

    There is no analytical form for the horeshoe prior's pdf, but it
    satisfies a tight bound of the form `lb(x) <= pdf(x) <= ub(x)`, where

        lb(x) = K/2 * log(1 + 4 * (scale / x) ** 2)
        ub(x) = K * log(1 + 2 * (scale / x) ** 2)

    with `K = 1 / sqrt(2 pi^3)`. Here, we simply use

        pdf(x) ~ (lb(x) + ub(x)) / 2

    Reference: C. M. Carvalho, N. G. Polson, and J. G. Scott.
        The horseshoe estimator for sparse signals. Biometrika, 2010.
    """

    arg_constraints = {"scale": constraints.positive}
    support = constraints.real
    _validate_args = True

    def __init__(self, scale, validate_args=False, transform=None):
        TModule.__init__(self)
        if isinstance(scale, Number):
            scale = torch.tensor(float(scale))
        self.K = 1 / math.sqrt(2 * math.pi ** 3)
        self.scale = scale
        super().__init__(scale.shape, validate_args=validate_args)
        # now need to delete to be able to register buffer
        del self.scale
        self.register_buffer("scale", scale)
        self._transform = transform

    def log_prob(self, X):
        A = (self.scale / self.transform(X)) ** 2
        lb = self.K / 2 * torch.log(1 + 4 * A)
        ub = self.K * torch.log(1 + 2 * A)
        return torch.log((lb + ub) / 2)
