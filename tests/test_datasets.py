from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import numpy.testing as npt
import pytest

from anml_exp.data import HashError, load_dataset
from anml_exp.data.registry import _FILE_HASHES, _sha256


def test_toy_blobs_deterministic() -> None:
    X_train1, y_train1 = load_dataset("toy-blobs", split="train", seed=0)
    X_train2, y_train2 = load_dataset("toy-blobs", split="train", seed=0)
    X_test1, y_test1 = load_dataset("toy-blobs", split="test", seed=0)
    X_test2, y_test2 = load_dataset("toy-blobs", split="test", seed=0)

    assert y_train1 is None
    assert y_train2 is None

    npt.assert_array_equal(X_train1, X_train2)
    npt.assert_array_equal(X_test1, X_test2)
    npt.assert_array_equal(y_test1, y_test2)


def test_toy_blobs_shapes() -> None:
    X_train, y_train = load_dataset("toy-blobs", split="train")
    X_test, y_test = load_dataset("toy-blobs", split="test")

    assert y_train is None
    assert X_train.shape == (100, 2)
    assert X_test.shape == (100, 2)
    assert y_test is not None
    assert y_test.shape == (100,)


def test_toy_circles_deterministic() -> None:
    X_train1, y_train1 = load_dataset("toy-circles", split="train", seed=0)
    X_train2, y_train2 = load_dataset("toy-circles", split="train", seed=0)
    X_test1, y_test1 = load_dataset("toy-circles", split="test", seed=0)
    X_test2, y_test2 = load_dataset("toy-circles", split="test", seed=0)

    assert y_train1 is None
    assert y_train2 is None

    npt.assert_array_equal(X_train1, X_train2)
    npt.assert_array_equal(X_test1, X_test2)
    npt.assert_array_equal(y_test1, y_test2)


def test_toy_circles_shapes() -> None:
    X_train, y_train = load_dataset("toy-circles", split="train")
    X_test, y_test = load_dataset("toy-circles", split="test")

    assert y_train is None
    assert X_train.shape == (100, 2)
    assert X_test.shape == (200, 2)
    assert y_test is not None
    assert y_test.shape == (200,)


def test_breast_cancer_deterministic() -> None:
    X_train1, y_train1 = load_dataset("breast-cancer", split="train", seed=0)
    X_train2, y_train2 = load_dataset("breast-cancer", split="train", seed=0)
    X_test1, y_test1 = load_dataset("breast-cancer", split="test", seed=0)
    X_test2, y_test2 = load_dataset("breast-cancer", split="test", seed=0)

    assert y_train1 is None
    assert y_train2 is None

    npt.assert_array_equal(X_train1, X_train2)
    npt.assert_array_equal(X_test1, X_test2)
    npt.assert_array_equal(y_test1, y_test2)


def test_breast_cancer_shapes() -> None:
    X_train, y_train = load_dataset("breast-cancer", split="train")
    X_test, y_test = load_dataset("breast-cancer", split="test")

    assert y_train is None
    # number of benign samples in training may vary due to split, check shape
    assert X_train.shape[1] == X_test.shape[1] == 30
    assert y_test is not None
    assert y_test.ndim == 1


def test_wine_deterministic() -> None:
    X_train1, y_train1 = load_dataset("wine", split="train", seed=0)
    X_train2, y_train2 = load_dataset("wine", split="train", seed=0)
    X_test1, y_test1 = load_dataset("wine", split="test", seed=0)
    X_test2, y_test2 = load_dataset("wine", split="test", seed=0)

    assert y_train1 is None
    assert y_train2 is None

    npt.assert_array_equal(X_train1, X_train2)
    npt.assert_array_equal(X_test1, X_test2)
    npt.assert_array_equal(y_test1, y_test2)


def test_wine_shapes() -> None:
    X_train, y_train = load_dataset("wine", split="train")
    X_test, y_test = load_dataset("wine", split="test")

    assert y_train is None
    assert X_train.shape[1] == X_test.shape[1] == 13
    assert y_test is not None
    assert y_test.ndim == 1


def test_digits_deterministic() -> None:
    X_train1, y_train1 = load_dataset("digits", split="train", seed=0)
    X_train2, y_train2 = load_dataset("digits", split="train", seed=0)
    X_test1, y_test1 = load_dataset("digits", split="test", seed=0)
    X_test2, y_test2 = load_dataset("digits", split="test", seed=0)

    assert y_train1 is None
    assert y_train2 is None

    npt.assert_array_equal(X_train1, X_train2)
    npt.assert_array_equal(X_test1, X_test2)
    npt.assert_array_equal(y_test1, y_test2)


def test_digits_shapes() -> None:
    X_train, y_train = load_dataset("digits", split="train")
    X_test, y_test = load_dataset("digits", split="test")

    assert y_train is None
    assert X_train.shape[1] == X_test.shape[1] == 64
    assert y_test is not None
    assert y_test.ndim == 1


def test_nab_twitter_aapl_shapes() -> None:
    X_train, y_train = load_dataset("nab-twitter-aapl", split="train")
    X_test, y_test = load_dataset("nab-twitter-aapl", split="test")

    assert y_train is None
    assert X_train.shape[1] == 24
    assert X_test.shape[1] == 24
    assert y_test is not None
    assert y_test.shape == (X_test.shape[0],)


def test_dataset_hashes() -> None:
    for fname, expected in _FILE_HASHES.items():
        path = files("anml_exp.data").joinpath("datasets", fname)
        assert _sha256(Path(str(path))) == expected


def test_hash_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(_FILE_HASHES, "wine.npz", "0" * 64)
    with pytest.raises(HashError):
        load_dataset("wine", split="train")

