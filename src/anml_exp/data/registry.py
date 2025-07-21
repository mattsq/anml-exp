"""Dataset registry and loading utilities with hash verification."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray as _NDArray
from sklearn.datasets import make_blobs  # type: ignore[import-untyped]

NDArray = _NDArray[np.float64]
Dataset = Tuple[NDArray, Optional[NDArray]]


class HashError(RuntimeError):
    """Raised when a dataset file fails its hash check."""


# Pre-computed SHA-256 digests for files under ``datasets``
_FILE_HASHES: Dict[str, str] = {
    "breast_cancer.npz": (
        "2402285ebb2134863dfb40ffce13904d3f0bae1a81ba53c27339086f80b2490e"
    ),
    "wine.npz": (
        "adc597add63eb9e88a4fc555a9829d01577ac6f72815701a447bb20d8b6675a6"
    ),
    "digits.npz": (
        "6952bf8d305ecf99f1e65db39af0ba46ce6b72df49a5f695bc1c935c5e7e074b"
    ),
    "Twitter_volume_AAPL.csv": (
        "826f5cf404c2890784a7824f7102fd00cb134a4948e12e44ec320d095cbbc217"
    ),
    "nab_labels.json": (
        "2dc6fbccfae40e45e066badad2b8ff365f16a907a8f4be8fdfcd3ada955a99b9"
    ),
}

_DATA_URLS: Dict[str, str] = {
    "Twitter_volume_AAPL.csv": (
        "https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/"
        "Twitter_volume_AAPL.csv"
    ),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


_DATA_DIR = Path(__file__).resolve().parent / "datasets"


def _dataset_path(filename: str) -> Path:
    return _DATA_DIR / filename


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, dst.open("wb") as f:
        f.write(r.read())


def _make_breast_cancer(path: Path) -> None:
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    np.savez_compressed(path, X=data.data.astype(np.float64), y=data.target)


def _make_wine(path: Path) -> None:
    from sklearn.datasets import load_wine

    data = load_wine()
    np.savez_compressed(path, X=data.data.astype(np.float64), y=data.target)


def _make_digits(path: Path) -> None:
    from sklearn.datasets import load_digits

    data = load_digits()
    np.savez_compressed(path, X=data.data.astype(np.float64), y=data.target)


def _make_nab_labels(path: Path) -> None:
    url = (
        "https://raw.githubusercontent.com/numenta/NAB/master/labels/"
        "combined_labels.json"
    )
    with urllib.request.urlopen(url) as r:
        data = json.load(r)
    text = json.dumps(data, separators=(", ", ": "))
    path.write_text(text)


def _ensure_file(filename: str) -> Path:
    path = _dataset_path(filename)
    if path.exists():
        return path

    if filename == "breast_cancer.npz":
        _make_breast_cancer(path)
    elif filename == "wine.npz":
        _make_wine(path)
    elif filename == "digits.npz":
        _make_digits(path)
    elif filename == "nab_labels.json":
        _make_nab_labels(path)
    elif filename in _DATA_URLS:
        _download(_DATA_URLS[filename], path)
    else:  # pragma: no cover - defensive
        raise KeyError(f"Unknown dataset file: {filename}")
    return path


def _verify_file(filename: str) -> Path:
    path = _ensure_file(filename)
    digest = _sha256(path)
    expected = _FILE_HASHES[filename]
    if digest != expected:
        raise HashError(
            f"Hash mismatch for {filename}: {digest} != {expected}"
        )
    return path


# ------------------------------ Synthetic datasets ---------------------------

def _toy_blobs(split: str, *, seed: int) -> Dataset:
    """Synthetic Gaussian blobs for quick experiments."""
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
    """Synthetic concentric circles classification toy dataset."""
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
        y = (y == 0).astype(int)
        return X.astype(np.float64), y

    raise KeyError(f"Unknown split: {split}")


# ------------------------------ Real datasets -------------------------------

def _breast_cancer(split: str, *, seed: int) -> Dataset:
    """Breast Cancer Wisconsin (Diagnostic) dataset."""
    from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

    path = _verify_file("breast_cancer.npz")
    with np.load(path) as data:
        X = data["X"]
        y = data["y"]

    rng = np.random.default_rng(seed)
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
    """UCI Wine recognition dataset."""
    from sklearn.model_selection import train_test_split

    path = _verify_file("wine.npz")
    with np.load(path) as data:
        X = data["X"]
        y = data["y"]

    rng = np.random.default_rng(seed)
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
    """Scikit-learn digits dataset."""
    from sklearn.model_selection import train_test_split

    path = _verify_file("digits.npz")
    with np.load(path) as data:
        X = data["X"]
        y = data["y"]

    rng = np.random.default_rng(seed)
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


def _nab_twitter_aapl(split: str, *, seed: int) -> Dataset:  # noqa: ARG001
    """Twitter volume for AAPL from the NAB dataset."""
    from numpy.lib.stride_tricks import sliding_window_view

    data_path = _verify_file("Twitter_volume_AAPL.csv")
    label_path = _verify_file("nab_labels.json")

    arr = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype="str")
    times = np.array(arr[:, 0], dtype="datetime64[s]")
    values = arr[:, 1].astype(np.float64)

    with open(label_path) as f:
        labels = json.load(f)["realTweets/Twitter_volume_AAPL.csv"]
    anomaly_times = np.array([np.datetime64(t) for t in labels])

    flags = np.isin(times, anomaly_times).astype(int)
    window = 24
    windows = sliding_window_view(values, window)
    labels_window = np.array(
        [flags[i : i + window].max() for i in range(len(windows))], dtype=int
    )

    split_idx = int(0.8 * len(windows))

    if split == "train":
        return windows[:split_idx], None

    if split == "test":
        return windows[split_idx:], labels_window[split_idx:]

    raise KeyError(f"Unknown split: {split}")


_REGISTRY = {
    "toy-blobs": _toy_blobs,
    "toy-circles": _toy_circles,
    "breast-cancer": _breast_cancer,
    "wine": _wine,
    "digits": _digits,
    "nab-twitter-aapl": _nab_twitter_aapl,
}


def load_dataset(name: str, split: str = "train", *, seed: int = 42) -> Dataset:
    """Load a dataset by name verifying file hashes."""
    try:
        loader = _REGISTRY[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unknown dataset: {name!r}") from exc
    return loader(split, seed=seed)


def prepare_datasets() -> None:
    """Ensure all dataset files are present and hashed."""
    for fname in _FILE_HASHES:
        _verify_file(fname)


if __name__ == "__main__":  # pragma: no cover - CLI helper
    import argparse

    parser = argparse.ArgumentParser(description="Dataset utilities")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Download and verify all dataset files",
    )
    args = parser.parse_args()
    if args.fetch:
        prepare_datasets()
