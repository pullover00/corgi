from __future__ import annotations

from pathlib import Path
import os
import tempfile

import numpy as np

from .types import Reconstruction


def save_reconstruction(path: str | Path, reconstruction: Reconstruction) -> None:
    """Persist reconstruction atomically so an interrupted run cannot corrupt it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The temporary file lives beside the final cache, which makes os.replace
    # atomic on the same filesystem. A Ctrl-C can therefore lose only the pair
    # currently being computed, never a previously complete reconstruction.
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                image0=reconstruction.images[0],
                image1=reconstruction.images[1],
                points0=reconstruction.points[0],
                points1=reconstruction.points[1],
                depth0=reconstruction.depths[0],
                depth1=reconstruction.depths[1],
                intrinsics0=reconstruction.intrinsics[0],
                intrinsics1=reconstruction.intrinsics[1],
                world_to_camera0=reconstruction.world_to_camera[0],
                world_to_camera1=reconstruction.world_to_camera[1],
                confidence0=reconstruction.confidence[0],
                confidence1=reconstruction.confidence[1],
                match_count=np.array(reconstruction.match_count),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_reconstruction(path: str | Path) -> Reconstruction:
    """Restore a cached reconstruction without invoking MASt3R."""
    with np.load(path) as data:
        return Reconstruction(
            images=(data["image0"], data["image1"]),
            points=(data["points0"], data["points1"]),
            depths=(data["depth0"], data["depth1"]),
            intrinsics=(data["intrinsics0"], data["intrinsics1"]),
            world_to_camera=(data["world_to_camera0"], data["world_to_camera1"]),
            confidence=(data["confidence0"], data["confidence1"]),
            match_count=int(data["match_count"]),
        )
