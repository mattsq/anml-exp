"""Dataset registry and loaders."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.datasets import make_blobs  # type: ignore[import-untyped]

NDArray = np.ndarray
Dataset = Tuple[NDArray, Optional[NDArray]]


def _toy_blobs(split: str, *, seed: int) -> Dataset:
    """Synthetic Gaussian blobs for quick experiments.

    The training split contains a single Gaussian cluster representing
    "normal" data. The test split mixes the same cluster with an equal
    number of points from a second, well separated cluster which are
    labelled as anomalies.

    Data are generated via :func:`sklearn.datasets.make_blobs` and are
    fully deterministic given ``seed``.

    References
    ----------
    * F. Pedregosa et al., "Scikit-learn: Machine Learning in Python",
      Journal of Machine Learning Research, 2011.
    """

    rng = np.random.default_rng(seed)

    if split == "train":
        X, _ = make_blobs(
            n_samples=100,
            centers=[[0.0, 0.0]],
            cluster_std=0.5,
            random_state=rng.integers(0, 2**32 - 1),
        )
        return X.astype(np.float64), None

    if split == "test":
        X_norm, _ = make_blobs(
            n_samples=50,
            centers=[[0.0, 0.0]],
            cluster_std=0.5,
            random_state=rng.integers(0, 2**32 - 1),
        )
        X_anom, _ = make_blobs(
            n_samples=50,
            centers=[[4.0, 4.0]],
            cluster_std=0.5,
            random_state=rng.integers(0, 2**32 - 1),
        )
        X = np.vstack([X_norm, X_anom]).astype(np.float64)
        y = np.hstack([
            np.zeros(len(X_norm), dtype=int),
            np.ones(len(X_anom), dtype=int),
        ])
        return X, y

    raise KeyError(f"Unknown split: {split}")


_REGISTRY = {"toy-blobs": _toy_blobs}


def load_dataset(name: str, split: str = "train", *, seed: int = 42) -> Dataset:
    """Load a dataset by name.

    Parameters
    ----------
    name:
        Dataset identifier.
    split:
        ``"train"`` or ``"test"``.
    seed:
        Random seed controlling synthetic data generation.
    """

    try:
        loader = _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset: {name!r}") from exc
    return loader(split, seed=seed)
