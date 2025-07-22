import numpy as np

from anml_exp.models.imm_tabular import IMMTabularModel


def test_smoke() -> None:
    model = IMMTabularModel(n_features=4, epochs=1, batch_size=8)
    X = np.random.randn(16, 4).astype(np.float32)
    model.fit(X)
    scores = model.score_samples(X)
    assert scores.shape == (16,)
    assert np.all(np.isfinite(scores))
