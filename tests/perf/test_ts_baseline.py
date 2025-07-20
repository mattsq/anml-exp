from __future__ import annotations

import os
import time

import pytest

from anomaly_models import MatrixProfileModel
from datasets.registry import load_dataset

RUN_PERF = os.getenv("RUN_PERF") == "1"

@pytest.mark.skipif(not RUN_PERF, reason="Performance test disabled")
def test_matrix_profile_perf() -> None:
    X, _ = load_dataset("nab-art-daily", split="train")
    model = MatrixProfileModel(window_size=24).fit(X)
    start = time.perf_counter()
    model.score_samples(X)
    duration = time.perf_counter() - start
    assert duration < 5.0
