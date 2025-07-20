"""Simple feed-forward autoencoder model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base import ArrayLike, BaseAnomalyModel, NDArray


@dataclass
class _AEConfig:
    hidden_dim: int = 8
    lr: float = 1e-3
    n_epochs: int = 10
    batch_size: int = 32
    device: str | None = None


class AutoEncoderModel(BaseAnomalyModel):
    """Basic autoencoder trained with mean squared error."""

    def __init__(
        self,
        *,
        hidden_dim: int = 8,
        lr: float = 1e-3,
        n_epochs: int = 10,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.cfg = _AEConfig(
            hidden_dim=hidden_dim,
            lr=lr,
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=device,
        )
        self._threshold: float | None = None
        self.model: nn.Module | None = None
        self.input_dim: int | None = None

    def _build_model(self, input_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, self.cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.cfg.hidden_dim, input_dim),
        )

    def fit(
        self, X: ArrayLike, y: Optional[ArrayLike] | None = None
    ) -> "AutoEncoderModel":
        self.input_dim = int(X.shape[1])
        device = self.cfg.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self._build_model(self.input_dim).to(device)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)
        self.model.train()
        for _ in range(self.cfg.n_epochs):
            for (batch,) in loader:
                batch = batch.to(device)
                optimiser.zero_grad()
                recon = self.model(batch)
                loss = nn.functional.mse_loss(recon, batch)
                loss.backward()  # type: ignore[no-untyped-call]
                optimiser.step()
        self.model.eval()
        return self

    def score_samples(self, X: ArrayLike) -> NDArray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        device = next(self.model.parameters()).device
        tensor_X = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            recon = self.model(tensor_X)
            errors = torch.mean((recon - tensor_X) ** 2, dim=1)
        return np.asarray(errors.cpu().numpy(), dtype=np.float64)



