from __future__ import annotations

import numpy as np

from partfield_mc.cuboid_fit import (
    _ClusterGeometry,
    _constrained_refit_parallel_parts,
    _parallel_overlap_depth,
)
from partfield_mc.models import CuboidPart


def _part(segment_id: int, lower, upper, area: float) -> CuboidPart:
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = (lower + upper) * 0.5
    return CuboidPart(
        name=f"part_{segment_id}",
        segment_id=segment_id,
        size=upper - lower,
        transform=transform,
        face_count=20,
        surface_area=area,
        source_center=(lower + upper) * 0.5,
        metadata={},
    )


def _triangles(centers: np.ndarray, radius: float = 0.03) -> np.ndarray:
    return np.asarray(
        [
            [
                center + [-radius, -radius, 0.0],
                center + [radius, -radius, 0.0],
                center + [0.0, radius, radius],
            ]
            for center in centers
        ],
        dtype=float,
    )


def _geometry(segment_id: int, centers: np.ndarray, raw_min, raw_max, area=1.0):
    return _ClusterGeometry(
        segment_id=segment_id,
        face_vertices_local=_triangles(centers),
        face_centers_local=np.asarray(centers, dtype=float),
        face_areas=np.full(len(centers), float(area), dtype=float),
        raw_min=np.asarray(raw_min, dtype=float),
        raw_max=np.asarray(raw_max, dtype=float),
    )


def test_semantic_refit_protects_face_and_splits_torso() -> None:
    body = _part(0, [0.0, 0.0, 0.0], [2.0, 2.0, 1.5], area=12.0)
    face = _part(1, [1.35, 0.95, 0.25], [2.55, 2.25, 1.35], area=5.0)
    tail = _part(2, [-1.3, 1.05, 0.55], [0.2, 1.55, 0.95], area=2.0)

    # L-shaped torso support: a full lower slab plus an upper-rear slab.  One
    # AABB must lose one of these regions to avoid the protected face; two
    # touching boxes can preserve both.
    body_centers = np.asarray(
        [
            [0.15, 0.25, 0.40],
            [0.55, 0.35, 1.05],
            [1.05, 0.45, 0.55],
            [1.55, 0.55, 1.10],
            [1.90, 0.70, 0.70],
            [0.15, 1.20, 0.45],
            [0.45, 1.45, 1.00],
            [0.80, 1.65, 0.60],
            [1.05, 1.75, 1.10],
            [1.20, 1.35, 0.75],
        ]
    )
    face_centers = np.asarray(
        [
            [1.45, 1.05, 0.35],
            [1.80, 1.35, 0.75],
            [2.15, 1.65, 1.10],
            [2.45, 2.05, 0.65],
        ]
    )
    tail_centers = np.asarray(
        [
            [-1.20, 1.15, 0.65],
            [-0.85, 1.25, 0.75],
            [-0.45, 1.35, 0.80],
            [0.05, 1.45, 0.75],
        ]
    )
    geometry = {
        0: _geometry(0, body_centers, [0.0, 0.0, 0.0], [2.0, 2.0, 1.5]),
        1: _geometry(1, face_centers, [1.35, 0.95, 0.25], [2.55, 2.25, 1.35]),
        2: _geometry(2, tail_centers, [-1.3, 1.05, 0.55], [0.2, 1.55, 0.95]),
    }

    original_face_center = face.center.copy()
    original_face_size = face.size.copy()
    parts = [body, face, tail]
    _constrained_refit_parallel_parts(
        parts,
        geometry,
        adjacency={(0, 1), (0, 2)},
        gap=0.0,
        min_coverage=0.05,
        beam_width=64,
        preserve_contact=True,
        semantic_refit="animal",
        adaptive_split=True,
        max_extra_cuboids=2,
        protected_min_coverage=0.90,
        split_min_coverage_gain=0.04,
    )

    face_after = next(part for part in parts if part.segment_id == 1)
    assert face_after.metadata["semantic_role"] == "face"
    assert face_after.metadata["protected_visual_region"] is True
    assert np.allclose(face_after.center, original_face_center)
    assert np.allclose(face_after.size, original_face_size)

    body_parts = [part for part in parts if part.segment_id == 0]
    assert len(body_parts) == 2
    assert all(part.metadata["adaptive_split"] for part in body_parts)
    assert body_parts[0].metadata["split_combined_coverage_ratio"] >= 0.90

    for index, part_a in enumerate(parts):
        for part_b in parts[index + 1 :]:
            depth = _parallel_overlap_depth(part_a, part_b, np.eye(3))
            assert not np.all(depth > 1e-9)
