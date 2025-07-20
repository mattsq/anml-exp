"""Dataset registry and loaders."""
from __future__ import annotations

from typing import Tuple, Optional
import numpy as np

NDArray = np.ndarray
Dataset = Tuple[NDArray, Optional[NDArray]]


def load_dataset(name: str, split: str = "train") -> Dataset:
    """Load a dataset by name.

    This is a placeholder implementation.
    """
    raise NotImplementedError
