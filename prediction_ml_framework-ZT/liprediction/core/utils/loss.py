# Copyright (c) 2022 Li Auto Company. All rights reserved.

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def two_dimension_gaussian_nll_loss(
    input: Tensor,
    target: Tensor,
    sigma: Tensor,
    rho: Tensor,
    full: bool = False,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> Tensor:
    r"""2D Gaussian negative log likelihood loss.

    Args:
        input: expectation of the Gaussian distribution, x, y.
        target: sample from the Gaussian distribution, gt_x, gt_y.
        sigma: tensor of positive variance(s), sigma_x, sigma_y.
        rho: tensor of xy rho, [-1, 1]
        full (bool, optional): include the constant term in the loss calculation. Default: ``False``.
        eps (float, optional): value added to var, for stability. Default: 1e-6.
        reduction (string, optional): specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output is the average of all batch member losses,
            ``'sum'``: the output is the sum of all batch member losses.
            Default: ``'mean'``.
    """

    # Check var size
    if input.size()[:-1] == sigma.size()[:-1] and sigma.size(-1) == 2 and rho.size(-1) == 1:
        pass
    else:
        # the size of var is incorrect.
        raise ValueError("var is of incorrect size")

    # Check validity of reduction mode
    if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
        raise ValueError(reduction + " is not valid")

    # Entries of var must be non-negative
    if torch.any(sigma < 0):
        raise ValueError("var has negative entry/entries")

    # Calculate the loss, -log(f(x, y)) --> https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    # For loss stability, add some epsilon
    # --> https://stackoverflow.com/questions/43031731/negative-values-in-log-likelihood-of-a-bivariate-gaussian

    norm1 = torch.log(1 + torch.abs(target[..., [0]] - input[..., [0]]))
    norm2 = torch.log(1 + torch.abs(target[..., [1]] - input[..., [1]]))

    variance_x = F.softplus(torch.square(sigma[..., [0]]))
    variance_y = F.softplus(torch.square(sigma[..., [1]]))
    s1s2 = F.softplus(sigma[..., [0]] * sigma[..., [1]])  # very large if sigma_x and/or sigma_y are very large

    z = F.softplus((torch.square(norm1) / variance_x) + (torch.square(norm2) / variance_y) -
                   (2 * rho * norm1 * norm2 / s1s2))
    neg_rho = 1 - torch.square(0.99 * rho)
    numerator = torch.exp(-z / (2 * neg_rho))
    denominator = (s1s2 * torch.sqrt(neg_rho)) + eps
    pdf = numerator / denominator

    loss = -torch.log(pdf + eps)

    if full:
        loss += math.log(2 * math.pi)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
