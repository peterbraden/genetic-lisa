import torch
import torch.nn.functional as F

from splatlisa.loss import _kernel, ssim


def reference_ssim(a, b, size=11, sigma=1.5):
    """Direct 2-D window SSIM, for comparison with the separable version."""
    g = _kernel(size, sigma, a.device)
    win = (g[:, None] * g[None, :]).expand(3, 1, size, size)
    blur = lambda x: F.conv2d(x[None], win, padding=size // 2, groups=3)[0]
    mu_a, mu_b = blur(a), blur(b)
    var_a, var_b, cov = blur(a * a) - mu_a**2, blur(b * b) - mu_b**2, blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    return (((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))).mean()


def test_separable_ssim_matches_2d_window():
    g = torch.Generator().manual_seed(0)
    a, b = torch.rand(3, 30, 25, generator=g), torch.rand(3, 30, 25, generator=g)
    assert torch.allclose(ssim(a, b), reference_ssim(a, b), atol=1e-5)
    assert torch.allclose(ssim(a, a), torch.tensor(1.0), atol=1e-5)
