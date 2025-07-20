from __future__ import annotations

from anml_exp.registry import Registry


def test_registry_roundtrip(tmp_path):
    registry = Registry(tmp_path)
    obj = {"a": 1}
    digest = registry.save(obj, "dummy", "0.1.0")
    assert digest.startswith("sha256:")
    loaded = registry.load("dummy", "0.1.0")
    assert loaded == obj
