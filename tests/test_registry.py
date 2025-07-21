from __future__ import annotations

from pathlib import Path

import pytest

from anml_exp.registry import Registry


def test_registry_roundtrip(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    obj = {"a": 1}
    digest = registry.save(obj, "dummy", "0.1.0")
    assert digest.startswith("sha256:")
    loaded = registry.load("dummy", "0.1.0")
    assert loaded == obj


def test_registry_digest_mismatch(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    obj = {"a": 1}
    registry.save(obj, "dummy", "0.1.0")
    artefact = tmp_path / "dummy" / "0.1.0" / "artefact.pkl"
    artefact.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError):
        registry.load("dummy", "0.1.0")


def test_registry_bad_metadata(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    obj = [1, 2, 3]
    registry.save(obj, "dummy", "0.2.0")
    meta = tmp_path / "dummy" / "0.2.0" / "metadata.json"
    meta.write_text("{}")
    with pytest.raises(RuntimeError):
        registry.load("dummy", "0.2.0")
