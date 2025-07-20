"""Simple Deep SVDD implementation with a single hidden layer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base import ArrayLike, BaseAnomalyModel, NDArray


@dataclass
class _SVDDConfig:
    hidden_dim: int = 8
    lr: float = 1e-3
    n_epochs: int = 10
    batch_size: int = 32
    device: str | None = None


class DeepSVDDModel(BaseAnomalyModel):
    """Deep Support Vector Data Description.

    A lightweight implementation using a single fully-connected layer.
    The network is trained to map normal data close to a fixed centre
    computed from the initial network output.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 8,
        lr: float = 1e-3,
        n_epochs: int = 10,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.cfg = _SVDDConfig(
            hidden_dim=hidden_dim,
            lr=lr,
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=device,
        )
        self.model: nn.Module | None = None
        self.center: torch.Tensor | None = None
        self._threshold: float | None = None
        self.input_dim: int | None = None

    def _build_model(self, input_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, self.cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
        )

    def fit(
        self, X: ArrayLike, y: Optional[ArrayLike] | None = None
    ) -> "DeepSVDDModel":
        self.input_dim = int(X.shape[1])
        device = self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(self.input_dim).to(device)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)

        # Compute centre of initial network output
        self.model.eval()
        with torch.no_grad():
            outputs = []
            for (batch,) in loader:
                batch = batch.to(device)
                outputs.append(self.model(batch))
            self.center = torch.mean(torch.cat(outputs, dim=0), dim=0)

        self.model.train()
        for _ in range(self.cfg.n_epochs):
            for (batch,) in loader:
                batch = batch.to(device)
                optimiser.zero_grad()
                feats = self.model(batch)
                loss = torch.mean((feats - self.center) ** 2)
                loss.backward()  # type: ignore[no-untyped-call]
                optimiser.step()
        self.model.eval()
        return self

    def score_samples(self, X: ArrayLike) -> NDArray:
        if self.model is None or self.center is None:
            raise RuntimeError("Model not fitted")
        device = next(self.model.parameters()).device
        tensor_X = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            feats = self.model(tensor_X)
            dist = torch.mean((feats - self.center) ** 2, dim=1)
        return dist.cpu().numpy()

    @property
    def decision_threshold(self) -> float:
        if self._threshold is None:
            raise RuntimeError("Model has no threshold set")
        return self._threshold

    def set_threshold(self, value: float) -> None:
        self._threshold = value
