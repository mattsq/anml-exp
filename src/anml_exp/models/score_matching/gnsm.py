"""Gumbel-Noise Score Matching model."""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture  # type: ignore[import-untyped]

from ..base import ArrayLike, BaseAnomalyModel, NDArray


class _ScoreNet(nn.Module):
    """Simple MLP predicting epsilons."""

    def __init__(
        self,
        dim_in: int,
        dim_hidden: int = 256,
        dim_out: int | None = None,
    ) -> None:
        super().__init__()
        dim_out = dim_out or dim_in
        self.net = nn.Sequential(
            nn.Linear(dim_in + 1, dim_hidden),
            nn.SiLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.SiLU(),
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, x: torch.Tensor, log_lambda: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x, log_lambda.expand_as(x[..., :1])], dim=-1)
        return cast(torch.Tensor, self.net(h))


class GNSMModel(BaseAnomalyModel):
    """Gumbel-Noise Score Matching for categorical data."""

    def __init__(
        self,
        cardinals: list[int],
        lambdas: tuple[float, ...] = (0.1, 0.3, 0.7, 1.5),
        *,
        hidden: int = 256,
        epochs: int = 100,
        lr: float = 1e-3,
        gmm_components: int = 5,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.cardinals = cardinals
        self.lambdas = torch.tensor(lambdas, dtype=torch.float32)
        self.dim = int(sum(cardinals))
        self.net = _ScoreNet(self.dim, hidden).to(device)
        self.gmm = GaussianMixture(gmm_components)
        self.epochs = epochs
        self.lr = lr
        self.device = device

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "GNSMModel":
        X_oh = self._one_hot(X)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            idx = torch.randint(len(X_oh), (256,))
            x = X_oh[idx].to(self.device)
            loss = torch.tensor(0.0, device=self.device)
            for lam in self.lambdas.to(self.device):
                x_tilde, target = self._gumbel_noise(x, lam)
                pred = self.net(x_tilde, torch.log(lam)[None])
                loss = loss + F.mse_loss(torch.sigmoid(pred), torch.sigmoid(target))
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()

        with torch.no_grad():
            eta = self._embed(torch.as_tensor(X_oh, device=self.device))
        self.gmm.fit(eta.cpu().numpy())
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def score_samples(self, X: ArrayLike) -> NDArray:
        X_oh = self._one_hot(X)
        with torch.no_grad():
            eta = self._embed(torch.as_tensor(X_oh, device=self.device))
        ll: np.ndarray = self.gmm.score_samples(eta.cpu().numpy())
        return (-ll).astype(float)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _one_hot(self, X: ArrayLike) -> torch.Tensor:
        blocks = [
            F.one_hot(torch.as_tensor(X[:, d]), K) for d, K in enumerate(self.cardinals)
        ]
        return torch.cat(blocks, dim=-1).float()

    def _gumbel_noise(
        self, x: torch.Tensor, lam: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g = -torch.empty_like(x).exponential_().log()
        logits = torch.log(x + 1e-20) + g
        x_tilde = F.softmax(logits / lam, dim=-1)
        eps_target = torch.log(x + 1e-20) - lam * x_tilde
        return x_tilde, eps_target

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        norms = []
        for lam in self.lambdas.to(self.device):
            s = self._score(x, lam)
            norms.append(s.pow(2).sum(dim=-1))
        return torch.stack(norms, dim=-1).cpu()

    def _score(self, x: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        eps_hat = self.net(x, torch.log(lam)[None])
        return -lam + lam * x.size(-1) * torch.softmax(eps_hat, dim=-1)

