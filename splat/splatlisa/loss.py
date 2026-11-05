"""Photometric loss as in 3DGS: (1 - w) * L1 + w * (1 - SSIM)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _window(size: int, sigma: float, device) -> Tensor:
    x = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(3, 1, size, size).contiguous()


def ssim(a: Tensor, b: Tensor, size: int = 11, sigma: float = 1.5) -> Tensor:
    """Mean SSIM of two (3, H, W) images in [0, 1]."""
    win = _window(size, sigma, a.device)
    blur = lambda x: F.conv2d(x[None], win, padding=size // 2, groups=3)[0]
    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a**2
    var_b = blur(b * b) - mu_b**2
    cov = blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
    return s.mean()


def photometric(img: Tensor, target: Tensor, ssim_weight: float = 0.2) -> Tensor:
    l1 = (img - target).abs().mean()
    if ssim_weight == 0:
        return l1
    return (1 - ssim_weight) * l1 + ssim_weight * (1 - ssim(img, target))
