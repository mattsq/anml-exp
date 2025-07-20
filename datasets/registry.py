"""Dataset registry and loaders.

This module provides a minimal :func:`load_dataset` interface used across
the repository.  Datasets are returned as ``(X, y)`` tuples where ``y`` may be
``None`` for unsupervised training splits.  Splits are deterministic given a
``seed`` and currently include a couple of small examples for testing and
demonstration purposes.
"""
from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
from sklearn.datasets import load_iris, make_blobs  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

NDArray = np.ndarray
Dataset = Tuple[NDArray, Optional[NDArray]]

def _load_toy_blobs(seed: int = 42) -> Dataset:
    """Synthetic Gaussian blobs.

    This dataset consists of a large cluster of ``normal`` points centred at the
    origin and a smaller cluster of ``anomalies`` a few units away.  It is often
    used as a minimal sanity check for models.

    Parameters
    ----------
    seed:
        Random seed controlling sample generation.

    Returns
    -------
    Dataset
        Features ``X`` with corresponding binary labels ``y``.
    """

    X, y = make_blobs(
        n_samples=[900, 100],
        centers=[(0.0, 0.0), (4.0, 4.0)],
        cluster_std=0.5,
        random_state=seed,
    )
    return X.astype(float), y.astype(int)


def _load_iris(seed: int = 42) -> Dataset:
    """Binary anomaly version of Fisher's Iris dataset.

    Following common practice in anomaly-detection literature, the ``setosa``
    class is treated as ``normal`` while the remaining classes are labelled as
    ``anomalous``.

    Parameters
    ----------
    seed:
        Included for API symmetry; the dataset itself is deterministic.

    Returns
    -------
    Dataset
        Features ``X`` with corresponding binary labels ``y``.
    """

    data = load_iris()
    X = data.data.astype(float)
    y = (data.target != 0).astype(int)
    # shuffle deterministically to avoid ordered labels
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    return X[idx], y[idx]


_REGISTRY = {
    "toy-blobs": _load_toy_blobs,
    "iris": _load_iris,
}


def load_dataset(name: str, split: str = "train", *, seed: int = 42) -> Dataset:
    """Load a dataset by name.

    Parameters
    ----------
    name:
        Dataset identifier.  See ``_REGISTRY`` for available names.
    split:
        ``"train"`` or ``"test"`` split.
    seed:
        Random seed controlling the train/test split and, when applicable,
        synthetic data generation.

    Returns
    -------
    Dataset
        Tuple of ``(X, y)`` where ``y`` is ``None`` for unsupervised training
        data.
    """

    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset: {name!r}")

    X, y = _REGISTRY[name](seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=seed,
        stratify=y,
    )

    if split == "train":
        return X_train, None
    if split == "test":
        return X_test, y_test

    raise ValueError("split must be 'train' or 'test'")
