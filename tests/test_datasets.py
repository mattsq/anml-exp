from __future__ import annotations

import numpy.testing as npt

from datasets.registry import load_dataset


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


def test_nab_art_daily_deterministic() -> None:
    X_train1, y_train1 = load_dataset("nab-art-daily", split="train")
    X_train2, y_train2 = load_dataset("nab-art-daily", split="train")
    X_test1, y_test1 = load_dataset("nab-art-daily", split="test")
    X_test2, y_test2 = load_dataset("nab-art-daily", split="test")

    assert y_train1 is None
    assert y_train2 is None

    npt.assert_array_equal(X_train1, X_train2)
    npt.assert_array_equal(X_test1, X_test2)
    npt.assert_array_equal(y_test1, y_test2)


def test_nab_art_daily_shapes() -> None:
    X_train, y_train = load_dataset("nab-art-daily", split="train")
    X_test, y_test = load_dataset("nab-art-daily", split="test")

    assert y_train is None
    assert X_train.ndim == 2
    assert X_test.ndim == 2
    assert y_test is not None
    assert X_test.shape[0] == y_test.shape[0]

