"""Atomic JSON checkpoint save/restore for the evaluation protocol."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Atomically write state dict to path as JSON (temp-file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=_json_default)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_checkpoint(path: Path) -> Optional[dict[str, Any]]:
    """Load and return checkpoint dict, or None if the file does not exist."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(obj: Any) -> Any:
    """Fallback serializer for non-standard types."""
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
