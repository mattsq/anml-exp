from __future__ import annotations

from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import ArrayLike, BaseAnomalyModel

# ──────────────────────────────── utils ──────────────────────────────────── 

def make_beta_schedule(T: int, schedule: str = "cosine") -> torch.Tensor:
    if schedule == 'linear':
        betas = torch.linspace(1e-4, 0.02, T)
    elif schedule == 'cosine':
        s = 0.008
        steps = torch.arange(T + 1, dtype=torch.float64)
        alphas_cumprod = torch.cos(((steps / T + s) / (1. + s)) * np.pi * 0.5) ** 2
        alphas_cumprod /= alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = betas.clamp(max=0.999)
    return betas.float()

def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_acp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(x0)
    return (
        sqrt_alphas_cumprod[t][:, None] * x0 +
        sqrt_one_minus_acp[t][:, None] * noise,
        noise
    )

# ──────────── conditioned MLP ε-predictor ────────────
class _CondMLP(nn.Module):
    def __init__(self, dim_in: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in * 2 + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim_in)  # predict ε
        )

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, x_cond: torch.Tensor
    ) -> torch.Tensor:
        log_t = torch.log(t.float() + 1e-5)[:, None]
        h = torch.cat([x_t, x_cond, log_t.expand_as(x_t[:, :1])], dim=1)
        return cast(torch.Tensor, self.net(h))

# ─────────────────────── main model class ───────────────────────
class DDADTabularModel(BaseAnomalyModel):
    """DDAD refactored for simple continuous tabular data."""
    def __init__(
        self,
        n_features: int,
        T: int = 1000,
        K: int = 25,
        beta_schedule: str = "cosine",
        hidden: int = 256,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 512,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.T, self.K = T, K
        self.device = torch.device(device)
        self.epochs, self.lr, self.bs = epochs, lr, batch_size

        self.betas = make_beta_schedule(T, beta_schedule).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_acp = torch.sqrt(self.alpha_cumprod)
        self.sqrt_om_acp = torch.sqrt(1.0 - self.alpha_cumprod)

        self.model = _CondMLP(n_features, hidden).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr)

        self._mu: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "DDADTabularModel":
        tensor_X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self._mu = tensor_X.mean(dim=0, keepdim=True)
        self._std = tensor_X.std(dim=0, unbiased=False, keepdim=True) + 1e-6
        tensor_X = ((tensor_X - self._mu) / self._std).to(self.device)

        steps = tensor_X.shape[0] // self.bs + 1
        for _ in range(self.epochs):
            idx = torch.randperm(tensor_X.shape[0])
            for i in range(steps):
                batch = tensor_X[idx[i * self.bs:(i + 1) * self.bs]]
                t = torch.randint(0, self.T, (batch.size(0),),
                                  device=self.device)
                x_t, noise = q_sample(batch, t,
                                      self.sqrt_acp, self.sqrt_om_acp)
                eps_hat = self.model(x_t, t, batch)
                loss = F.mse_loss(eps_hat, noise)
                self.opt.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                self.opt.step()
        return self

    @torch.no_grad()
    def _denoise(self, x_cond: torch.Tensor) -> torch.Tensor:
        x_t = torch.randn_like(x_cond)
        stride = self.T // self.K
        for t in range(self.T - 1, -1, -stride):
            t_cur = torch.full((x_cond.size(0),), t, device=self.device)
            eps = self.model(x_t, t_cur, x_cond)
            alpha_t = self.alphas[t]
            sqrt_one_minus_acp_t = self.sqrt_om_acp[t]
            sqrt_acp_t = self.sqrt_acp[t]

            x0_hat = (x_t - sqrt_one_minus_acp_t * eps) / sqrt_acp_t
            coeff = torch.sqrt(alpha_t)
            x_t = coeff * x0_hat + torch.sqrt(1 - alpha_t) * eps
        return x_t

    def _prep(self, X: ArrayLike) -> torch.Tensor:
        tensor_X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        if self._mu is None or self._std is None:
            raise ValueError("Model must be fitted first.")
        return ((tensor_X - self._mu) / self._std).to(self.device)

    def score_samples(self, X: ArrayLike) -> np.ndarray:
        Xn = self._prep(X)
        with torch.no_grad():
            x_hat = self._denoise(Xn)
            resid = (Xn - x_hat).pow(2).sum(dim=1).sqrt()
        return resid.cpu().numpy()

    @torch.no_grad()
    def reconstruct(self, X: ArrayLike) -> torch.Tensor:
        Xn = self._prep(X)
        x_hat = self._denoise(Xn)
        assert self._mu is not None and self._std is not None
        return x_hat * self._std + self._mu
