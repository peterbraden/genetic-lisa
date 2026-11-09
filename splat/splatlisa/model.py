"""Trainable ellipse set with an optimiser that survives structural edits."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .render import FIELDS

DEFAULT_LR = {"mu": 0.5, "log_s": 0.02, "theta": 0.02, "col": 0.03, "op": 0.03, "bg": 0.01}


class Model:
    def __init__(self, params: dict[str, Tensor], bg_logit: Tensor, lr: dict[str, float] | None = None):
        self.lr = {**DEFAULT_LR, **(lr or {})}
        self.p = {k: params[k].detach().clone().requires_grad_(True) for k in FIELDS}
        self.bg = bg_logit.detach().clone().requires_grad_(True)
        groups = [{"params": [self.p[k]], "lr": self.lr[k], "name": k} for k in FIELDS]
        groups.append({"params": [self.bg], "lr": self.lr["bg"], "name": "bg"})
        self.opt = torch.optim.Adam(groups, eps=1e-15)
        self.reset_stats()

    @property
    def n(self) -> int:
        return self.p["mu"].shape[0]

    @property
    def device(self):
        return self.bg.device

    def reset_stats(self):
        """Positional gradient accumulators used by clone/split."""
        self.grad_norm = torch.zeros(self.n, device=self.device)
        self.grad_vec = torch.zeros(self.n, 2, device=self.device)
        self.grad_count = 0

    def accumulate(self):
        g = self.p["mu"].grad
        if g is None:
            return
        self.grad_norm += g.norm(dim=1)
        self.grad_vec += g
        self.grad_count += 1

    def mean_grad(self) -> Tensor:
        return self.grad_norm / max(1, self.grad_count)

    def detached(self) -> dict[str, Tensor]:
        return {k: v.detach() for k, v in self.p.items()}

    def restructure(self, order: Tensor, new_rows: dict[str, Tensor] | None = None):
        """Rebuild the shape list as `cat(old, new_rows)[order]`.

        Shapes omitted from `order` are removed. Optimiser moments are carried
        over for surviving shapes and zeroed for new ones, as in 3DGS.
        """
        m = 0 if new_rows is None else new_rows["mu"].shape[0]
        order = order.to(self.device)
        stats_norm = torch.cat([self.grad_norm, torch.zeros(m, device=self.device)])[order]
        stats_vec = torch.cat([self.grad_vec, torch.zeros(m, 2, device=self.device)])[order]
        for group in self.opt.param_groups:
            name = group["name"]
            if name == "bg":
                continue
            old = group["params"][0]
            extra = new_rows[name].to(old) if m else old[:0]
            new = torch.cat([old.detach(), extra])[order].clone().requires_grad_(True)
            state = self.opt.state.pop(old, None)
            if state:
                for key in ("exp_avg", "exp_avg_sq"):
                    s = state[key]
                    state[key] = torch.cat([s, torch.zeros_like(extra)])[order].clone()
                self.opt.state[new] = state
            group["params"][0] = new
            self.p[name] = new
        self.grad_norm, self.grad_vec = stats_norm, stats_vec

    @torch.no_grad()
    def rescale(self, f: float):
        """Scale the image plane by `f`: centres and semi-axes, and the position
        moments so that Adam's steps scale with it (the position gradient
        scales by 1/f; the caller multiplies the position learning rate by f).
        Gradient statistics are reset."""
        self.p["mu"].mul_(f)
        self.p["log_s"].add_(math.log(f))
        state = self.opt.state.get(self.p["mu"])
        if state:
            state["exp_avg"].div_(f)
            state["exp_avg_sq"].div_(f * f)
        self.reset_stats()

    def set_lr(self, name: str, lr: float):
        for group in self.opt.param_groups:
            if group["name"] == name:
                group["lr"] = lr
