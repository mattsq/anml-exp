from datasets.registry import load_dataset
import numpy as np


def test_toy_blobs_shapes_and_labels() -> None:
    X_train, y_train = load_dataset("toy-blobs", split="train", seed=123)
    X_test, y_test = load_dataset("toy-blobs", split="test", seed=123)

    assert y_train is None
    assert X_train.ndim == 2
    assert X_test.shape[0] + X_train.shape[0] == 1000
    assert y_test is not None
    assert set(np.unique(y_test)) <= {0, 1}


def test_deterministic_split() -> None:
    X_train1, _ = load_dataset("toy-blobs", split="train", seed=999)
    X_train2, _ = load_dataset("toy-blobs", split="train", seed=999)
    assert np.array_equal(X_train1, X_train2)

