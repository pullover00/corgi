from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def load_rgb(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    """Load an image as HWC uint8 RGB, optionally resizing to (width, height)."""
    image = Image.open(path).convert("RGB")
    if size and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image)


def save_image(path: str | Path, array: np.ndarray) -> None:
    """Save uint8, label, boolean, or normalized floating-point image data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(array)
    if data.dtype != np.uint8:
        data = np.clip(data * 255 if data.max(initial=0) <= 1 else data, 0, 255).astype(np.uint8)
    Image.fromarray(data).save(path)


def save_json(path: str | Path, value: Any) -> None:
    """Atomically replace JSON so Ctrl-C cannot leave a half-written cache marker."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def pair_key(path0: str | Path, path1: str | Path, config: dict[str, Any]) -> str:
    """Create a content-addressed cache key from inputs and configuration."""
    digest = hashlib.sha256()
    for path in (Path(path0), Path(path1)):
        resolved = path.resolve()
        digest.update(str(resolved).encode())
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    digest.update(json.dumps(config, sort_keys=True).encode())
    return digest.hexdigest()[:20]
