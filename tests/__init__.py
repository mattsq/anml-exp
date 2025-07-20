from __future__ import annotations

import sys
from pathlib import Path

# Ensure package under src/ is importable without installation
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
