"""Minimal USAD (UnSupervised Anomaly Detection) model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base import ArrayLike, BaseAnomalyModel, NDArray


@dataclass
class _USADConfig:
    hidden_dim: int = 8
    lr: float = 1e-3
    n_epochs: int = 10
    batch_size: int = 32
    device: str | None = None
    alpha: float = 0.5


class _USADNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x1 = self.decoder1(z)
        x2 = self.decoder2(x1)
        return x1, x2


class USADModel(BaseAnomalyModel):
    """Simplified USAD training scheme with two decoders."""

    def __init__(
        self,
        *,
        hidden_dim: int = 8,
        lr: float = 1e-3,
        n_epochs: int = 10,
        batch_size: int = 32,
        device: str | None = None,
        alpha: float = 0.5,
    ) -> None:
        self.cfg = _USADConfig(
            hidden_dim=hidden_dim,
            lr=lr,
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=device,
            alpha=alpha,
        )
        self.net: _USADNet | None = None
        self._threshold: float | None = None
        self.input_dim: int | None = None

    def fit(
        self, X: ArrayLike, y: Optional[ArrayLike] | None = None
    ) -> "USADModel":
        self.input_dim = int(X.shape[1])
        device = self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = _USADNet(self.input_dim, self.cfg.hidden_dim).to(device)

        opt1 = torch.optim.Adam(
            list(self.net.encoder.parameters()) + list(self.net.decoder1.parameters()),
            lr=self.cfg.lr,
        )
        opt2 = torch.optim.Adam(self.net.decoder2.parameters(), lr=self.cfg.lr)

        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)

        for epoch in range(self.cfg.n_epochs):
            factor = epoch / (epoch + 1)
            for (batch,) in loader:
                batch = batch.to(device)

                # Update encoder + decoder1
                opt1.zero_grad()
                x1, x2 = self.net(batch)
                loss1 = (
                    (1 - factor) * nn.functional.mse_loss(x1, batch)
                    + factor * nn.functional.mse_loss(x2, batch)
                )
                loss1.backward()  # type: ignore[no-untyped-call]
                opt1.step()

                # Update decoder2
                opt2.zero_grad()
                with torch.no_grad():
                    z = self.net.encoder(batch)
                    x1_detached = self.net.decoder1(z)
                x2 = self.net.decoder2(x1_detached)
                loss2 = (
                    factor * nn.functional.mse_loss(x1_detached, batch)
                    + (1 - factor) * nn.functional.mse_loss(x2, batch)
                )
                loss2.backward()  # type: ignore[no-untyped-call]
                opt2.step()
        self.net.eval()
        return self

    def score_samples(self, X: ArrayLike) -> NDArray:
        if self.net is None:
            raise RuntimeError("Model not fitted")
        device = next(self.net.parameters()).device
        tensor_X = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            x1, x2 = self.net(tensor_X)
            recon_loss1 = torch.mean((tensor_X - x1) ** 2, dim=1)
            recon_loss2 = torch.mean((tensor_X - x2) ** 2, dim=1)
            scores = self.cfg.alpha * recon_loss1 + (1 - self.cfg.alpha) * recon_loss2
        return np.asarray(scores.cpu().numpy(), dtype=np.float64)

    @property
    def decision_threshold(self) -> float:
        if self._threshold is None:
            raise RuntimeError("Model has no threshold set")
        return self._threshold

    def set_threshold(self, value: float) -> None:
        self._threshold = value
