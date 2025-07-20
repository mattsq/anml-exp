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



def _toy_circles(split: str, *, seed: int) -> Dataset:
    """Synthetic concentric circles classification toy dataset.

    The training split consists solely of the inner circle, representing
    normal observations. The test split contains an equal number of points
    from both the inner circle and the outer ring. Points on the outer ring
    are labelled as anomalies.

    Data are generated via :func:`sklearn.datasets.make_circles` and are fully
    deterministic given ``seed``.

    References
    ----------
    * F. Pedregosa et al., "Scikit-learn: Machine Learning in Python",
      Journal of Machine Learning Research, 2011.
    """

    from sklearn.datasets import make_circles

    rng = np.random.default_rng(seed)

    if split == "train":
        X, y = make_circles(
            n_samples=200,
            noise=0.05,
            factor=0.3,
            random_state=rng.integers(0, 2**32 - 1),
        )
        X_inner = X[y == 1]
        return X_inner.astype(np.float64), None

    if split == "test":
        X, y = make_circles(
            n_samples=200,
            noise=0.05,
            factor=0.3,
            random_state=rng.integers(0, 2**32 - 1),
        )
        # points on the outer ring are anomalies
        y = (y == 0).astype(int)
        return X.astype(np.float64), y

    raise KeyError(f"Unknown split: {split}")


def _breast_cancer(split: str, *, seed: int) -> Dataset:
    """Breast Cancer Wisconsin (Diagnostic) dataset.

    Benign samples are treated as normal data and malignant samples as
    anomalies. The training split contains only benign cases. The test
    split is a stratified 20% hold-out containing both classes.

    Data are loaded via :func:`sklearn.datasets.load_breast_cancer` and the
    split is performed deterministically using
    :func:`sklearn.model_selection.train_test_split` with ``seed``.

    References
    ----------
    * W. N. Street, W. H. Wolberg, and O. L. Mangasarian,
      "Nuclear feature extraction for breast tumor diagnosis," IS&T/SPIE,
      1993.
    * D. Dua and C. Graff, "UCI Machine Learning Repository", 2019.
    """

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

    rng = np.random.default_rng(seed)

    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=rng.integers(0, 2**32 - 1),
        stratify=y,
    )

    if split == "train":
        X_norm = X_train[y_train == 1]
        return X_norm.astype(np.float64), None

    if split == "test":
        y_test_anom = (y_test == 0).astype(int)
        return X_test.astype(np.float64), y_test_anom

    raise KeyError(f"Unknown split: {split}")


def _wine(split: str, *, seed: int) -> Dataset:
    """UCI Wine recognition dataset.

    Class ``0`` (Cultivar 1) is considered normal. The training split
    contains only this class. The test split is a stratified 20% hold-out
    containing all three classes with non-normal samples labelled as
    anomalies.

    Data are loaded via :func:`sklearn.datasets.load_wine` and the split is
    performed deterministically using
    :func:`sklearn.model_selection.train_test_split` with ``seed``.

    References
    ----------
    * M. Forina et al., "Parvus" dataset, 1991.
    * D. Dua and C. Graff, "UCI Machine Learning Repository", 2019.
    """

    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)

    X, y = load_wine(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=rng.integers(0, 2**32 - 1),
        stratify=y,
    )

    if split == "train":
        X_norm = X_train[y_train == 0]
        return X_norm.astype(np.float64), None

    if split == "test":
        y_test_anom = (y_test != 0).astype(int)
        return X_test.astype(np.float64), y_test_anom

    raise KeyError(f"Unknown split: {split}")


def _digits(split: str, *, seed: int) -> Dataset:
    """Scikit-learn digits dataset.

    Digit ``0`` is treated as the normal class. The training split contains
    only zeros. The test split is a stratified 20% hold-out containing all
    digits with non-zero digits labelled as anomalies.

    Data are loaded via :func:`sklearn.datasets.load_digits` and the split is
    performed deterministically using
    :func:`sklearn.model_selection.train_test_split` with ``seed``.

    References
    ----------
    * K. Bache and M. Lichman, "UCI Machine Learning Repository", 2013.
    """

    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)

    X, y = load_digits(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=rng.integers(0, 2**32 - 1),
        stratify=y,
    )

    if split == "train":
        X_norm = X_train[y_train == 0]
        return X_norm.astype(np.float64), None

    if split == "test":
        y_test_anom = (y_test != 0).astype(int)
        return X_test.astype(np.float64), y_test_anom

    raise KeyError(f"Unknown split: {split}")


_REGISTRY = {
    "toy-blobs": _toy_blobs,
    "toy-circles": _toy_circles,
    "breast-cancer": _breast_cancer,
    "wine": _wine,
    "digits": _digits,
}


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
