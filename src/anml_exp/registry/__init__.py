"""Simple artefact registry with versioned storage."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

__all__ = ["Registry"]


class Registry:
    """Versioned artefact registry using SHA-256 digests."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # Internal helper to compute sha256 digest of a file
    @staticmethod
    def _digest_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _artefact_dir(self, name: str, version: str) -> Path:
        return self.root / name / version

    def save(self, obj: Any, name: str, version: str) -> str:
        """Save ``obj`` and return its ``sha256`` digest."""
        path = self._artefact_dir(name, version)
        path.mkdir(parents=True, exist_ok=True)
        artefact = path / "artefact.pkl"
        with artefact.open("wb") as f:
            pickle.dump(obj, f)
        digest = self._digest_file(artefact)
        meta = {"digest": f"sha256:{digest}"}
        with (path / "metadata.json").open("w") as f:
            json.dump(meta, f)
        return meta["digest"]

    def load(self, name: str, version: str) -> Any:
        """Load an artefact verifying its digest."""
        path = self._artefact_dir(name, version)
        artefact = path / "artefact.pkl"
        meta_path = path / "metadata.json"
        with meta_path.open() as f:
            meta = json.load(f)
        expected = meta.get("digest")
        if not expected or not expected.startswith("sha256:"):
            raise RuntimeError("Invalid metadata")
        digest = self._digest_file(artefact)
        if digest != expected.split(":", 1)[1]:
            raise RuntimeError("Digest mismatch")
        with artefact.open("rb") as f:
            return pickle.load(f)
