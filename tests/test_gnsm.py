from __future__ import annotations

import numpy as np

from anml_exp.data import load_dataset
from anml_exp.models.score_matching.gnsm import GNSMModel


def test_gnsm_smoke() -> None:
    X_train, _ = load_dataset("toy-blobs", split="train")
    # Discretise blobs into two categories per dimension
    X_cat = (X_train > 0).astype(int)
    model = GNSMModel(cardinals=[2, 2], epochs=2)
    model.fit(X_cat)
    scores = model.score_samples(X_cat)
    assert scores.shape == (X_cat.shape[0],)
    assert np.all(np.isfinite(scores))

