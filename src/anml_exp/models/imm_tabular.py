from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base import ArrayLike, BaseAnomalyModel, NDArray

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def rbf_mmd(x: torch.Tensor, y: torch.Tensor, bandwidth: float = 1.0) -> torch.Tensor:
    """Unbiased MMD^2 between two batches using an RBF kernel."""
    xx = torch.exp(-torch.cdist(x, x).pow(2) / (2 * bandwidth**2))
    yy = torch.exp(-torch.cdist(y, y).pow(2) / (2 * bandwidth**2))
    xy = torch.exp(-torch.cdist(x, y).pow(2) / (2 * bandwidth**2))
    return xx.mean() + yy.mean() - 2 * xy.mean()


class TimeEmbed(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # (B,)
        return cast(torch.Tensor, self.net(t[:, None]))


class IMMBackbone(nn.Module):
    def __init__(self, d_in: int, hidden: int, t_dim: int) -> None:
        super().__init__()
        self.t_embed = TimeEmbed(t_dim)
        self.net = nn.Sequential(
            nn.Linear(d_in + t_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_in),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x_t, self.t_embed(t)], dim=-1)
        return cast(torch.Tensor, self.net(h))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

@dataclass
class _IMMConfig:
    hidden: int = 256
    t_dim: int = 64
    n_steps: int = 8
    beta: float = 2.0
    mmd_bw: float = 1.0
    lr: float = 3e-4
    batch_size: int = 256
    epochs: int = 150
    device: str = "cpu"
    grad_clip: float = 1.0


class IMMTabularModel(BaseAnomalyModel):
    """Inductive Moment Matching for continuous tabular data."""
    alpha: torch.Tensor
    sigma: torch.Tensor

    def __init__(
        self,
        n_features: int,
        *,
        hidden: int = 256,
        t_dim: int = 64,
        n_steps: int = 8,
        beta: float = 2.0,
        mmd_bw: float = 1.0,
        lr: float = 3e-4,
        batch_size: int = 256,
        epochs: int = 150,
        device: str = "cpu",
        grad_clip: float = 1.0,
    ) -> None:
        super().__init__()
        self.cfg = _IMMConfig(
            hidden=hidden,
            t_dim=t_dim,
            n_steps=n_steps,
            beta=beta,
            mmd_bw=mmd_bw,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            grad_clip=grad_clip,
        )
        self.dim = n_features
        self.backbone = IMMBackbone(self.dim, hidden, t_dim).to(device)
        k = torch.arange(n_steps + 1, dtype=torch.float32) / n_steps
        alpha = torch.exp(-0.5 * beta * k ** 2)
        self.alpha = alpha.to(device)
        self.sigma = torch.sqrt(1 - alpha**2).to(device)
        self.opt = torch.optim.AdamW(self.backbone.parameters(), lr, weight_decay=0.0)
        self._mu: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    # ------------------------------------------------------------------
    def _interpolate(self, x0: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        a = self.alpha[idx]
        s = self.sigma[idx]
        eps = torch.randn_like(x0)
        return a[:, None] * x0 + s[:, None] * eps

    def _prep(self, X: ArrayLike) -> torch.Tensor:
        tensor_X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        if self._mu is None or self._std is None:
            raise RuntimeError("Model not fitted")
        return ((tensor_X - self._mu) / self._std).to(self.alpha.device)

    # ------------------------------------------------------------------
    def fit(
        self, X: ArrayLike, y: Optional[ArrayLike] | None = None
    ) -> "IMMTabularModel":
        del y
        tensor_X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self._mu = tensor_X.mean(dim=0, keepdim=True)
        self._std = tensor_X.std(dim=0, unbiased=False, keepdim=True) + 1e-6
        tensor_X = ((tensor_X - self._mu) / self._std).to(self.alpha.device)

        loader = DataLoader(
            TensorDataset(tensor_X), batch_size=self.cfg.batch_size, shuffle=True
        )
        self.backbone.train()
        for _ in range(self.cfg.epochs):
            for (batch,) in loader:
                B = batch.size(0)
                k = torch.randint(1, self.cfg.n_steps + 1, (B,), device=batch.device)
                x_t = self._interpolate(batch, k)
                z = torch.randn_like(batch)
                k_plus = torch.clamp(k + 1, max=self.cfg.n_steps)
                z_tplus = self._interpolate(z, k_plus)
                delta = self.backbone(z_tplus, k_plus.float() / self.cfg.n_steps)
                z_t = z_tplus + (self.alpha[k] - self.alpha[k_plus])[:, None] * delta
                loss = rbf_mmd(x_t, z_t, self.cfg.mmd_bw)
                self.opt.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(
                    self.backbone.parameters(), self.cfg.grad_clip
                )
                self.opt.step()
        self.backbone.eval()
        return self

    # ------------------------------------------------------------------
    def score_samples(self, X: ArrayLike) -> NDArray:
        if self._mu is None or self._std is None:
            raise RuntimeError("Model not fitted")
        Xn = self._prep(X)
        t_idx = torch.full(
            (Xn.size(0),), self.cfg.n_steps, dtype=torch.long, device=Xn.device
        )
        x_noisy = self._interpolate(Xn, t_idx)
        delta = self.backbone(x_noisy, torch.ones_like(t_idx, dtype=torch.float32))
        x_rec = x_noisy + (-self.alpha[-1]) * delta
        scores = (Xn - x_rec).abs().mean(dim=1)
        return cast(NDArray, scores.detach().cpu().numpy())
