from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CuboidPart:
    """One oriented cuboid fitted to a PartField segment.

    transform is a local-to-world 4x4 matrix. The cuboid is centered at the
    local origin and has side lengths given by size.
    """

    name: str
    segment_id: int
    size: np.ndarray
    transform: np.ndarray
    face_count: int
    surface_area: float
    source_center: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> np.ndarray:
        return np.asarray(self.transform[:3, 3], dtype=np.float64)

    @property
    def rotation(self) -> np.ndarray:
        return np.asarray(self.transform[:3, :3], dtype=np.float64)

    @property
    def volume(self) -> float:
        return float(np.prod(self.size))

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "segment_id": int(self.segment_id),
            "center": self.center.tolist(),
            "size": np.asarray(self.size, dtype=float).tolist(),
            "rotation": self.rotation.tolist(),
            "transform": np.asarray(self.transform, dtype=float).tolist(),
            "face_count": int(self.face_count),
            "surface_area": float(self.surface_area),
            "volume": self.volume,
            "source_center": np.asarray(self.source_center, dtype=float).tolist(),
            "metadata": self.metadata,
        }


@dataclass
class PartFieldArtifacts:
    job_id: str
    canonical_input: Path
    feature_dir: Path
    cluster_dir: Path
    normalized_mesh: Path
    labels_path: Path
    colored_segmentation: Path | None


@dataclass
class ExportPaths:
    output_dir: Path
    glb: Path
    obj: Path
    mtl: Path
    texture: Path
    parts_json: Path
    segmented_ply: Path | None
