"""Photometric loss as in 3DGS: (1 - w) * L1 + w * (1 - SSIM)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _kernel(size: int, sigma: float, device) -> Tensor:
    x = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    return g / g.sum()


def ssim(a: Tensor, b: Tensor, size: int = 11, sigma: float = 1.5) -> Tensor:
    """Mean SSIM of two (3, H, W) images in [0, 1], with a Gaussian window.

    The window is separable, so it is applied as two 1-D passes over all five
    statistics' channels at once.
    """
    g = _kernel(size, sigma, a.device)
    x = torch.cat([a, b, a * a, b * b, a * b])[None]  # (1, 15, H, W)
    x = F.conv2d(x, g.view(1, 1, 1, size).expand(15, 1, 1, size), padding=(0, size // 2), groups=15)
    x = F.conv2d(x, g.view(1, 1, size, 1).expand(15, 1, size, 1), padding=(size // 2, 0), groups=15)
    mu_a, mu_b, aa, bb, ab = x[0].split(3)
    var_a = aa - mu_a**2
    var_b = bb - mu_b**2
    cov = ab - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
    return s.mean()


def photometric(img: Tensor, target: Tensor, ssim_weight: float = 0.2) -> Tensor:
    l1 = (img - target).abs().mean()
    if ssim_weight == 0:
        return l1
    return (1 - ssim_weight) * l1 + ssim_weight * (1 - ssim(img, target))
