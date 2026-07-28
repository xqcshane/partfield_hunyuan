from __future__ import annotations

import numpy as np

from partfield_mc.cuboid_fit import (
    _ClusterGeometry,
    _constrained_refit_parallel_parts,
    _parallel_overlap_depth,
)
from partfield_mc.models import CuboidPart


def _part(segment_id: int, lower: list[float], upper: list[float], area: float) -> CuboidPart:
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = (lower_array + upper_array) * 0.5
    return CuboidPart(
        name=f"part_{segment_id}",
        segment_id=segment_id,
        size=upper_array - lower_array,
        transform=transform,
        face_count=12,
        surface_area=area,
        source_center=(lower_array + upper_array) * 0.5,
        metadata={},
    )


def _triangle_faces(centers: np.ndarray, radius: float = 0.08) -> np.ndarray:
    faces = []
    for center in centers:
        faces.append(
            np.asarray(
                [
                    center + [-radius, -radius, -radius],
                    center + [radius, -radius, -radius],
                    center + [0.0, radius, radius],
                ],
                dtype=float,
            )
        )
    return np.asarray(faces)


def test_refit_keeps_large_body_and_attaches_tail_without_overlap() -> None:
    body = _part(0, [0.0, -1.0, -1.0], [2.0, 1.0, 1.0], area=10.0)
    tail = _part(1, [-1.0, -0.5, -0.5], [0.8, 0.5, 0.5], area=3.0)
    tail_centers = np.asarray(
        [[-0.9, 0.0, 0.0], [-0.7, 0.2, 0.0], [-0.5, -0.2, 0.0], [-0.2, 0.0, 0.0], [0.3, 0.0, 0.0]],
        dtype=float,
    )
    geometry = {
        0: _ClusterGeometry(
            segment_id=0,
            face_vertices_local=np.asarray([[[0.0, -1.0, -1.0], [2.0, -1.0, -1.0], [0.0, 1.0, 1.0]]]),
            face_centers_local=np.asarray([[1.0, 0.0, 0.0]]),
            face_areas=np.asarray([10.0]),
            raw_min=np.asarray([0.0, -1.0, -1.0]),
            raw_max=np.asarray([2.0, 1.0, 1.0]),
        ),
        1: _ClusterGeometry(
            segment_id=1,
            face_vertices_local=_triangle_faces(tail_centers),
            face_centers_local=tail_centers,
            face_areas=np.ones(len(tail_centers)),
            raw_min=np.asarray([-1.0, -0.5, -0.5]),
            raw_max=np.asarray([0.8, 0.5, 0.5]),
        ),
    }
    parts = [body, tail]
    original_body_center = body.center.copy()
    original_body_size = body.size.copy()

    _constrained_refit_parallel_parts(
        parts,
        geometry,
        adjacency={(0, 1)},
        gap=0.0,
        min_coverage=0.02,
        beam_width=64,
        preserve_contact=True,
    )

    assert len(parts) == 2
    body_after = next(part for part in parts if part.segment_id == 0)
    tail_after = next(part for part in parts if part.segment_id == 1)
    assert np.allclose(body_after.center, original_body_center)
    assert np.allclose(body_after.size, original_body_size)
    depth = _parallel_overlap_depth(body_after, tail_after, np.eye(3))
    assert not np.all(depth > 1e-9)
    assert abs((tail_after.center[0] + tail_after.size[0] * 0.5) - 0.0) <= 1e-9
    assert tail_after.metadata["constrained_refit_contact_segments"] == [0]
    assert tail_after.metadata["constrained_refit_coverage_ratio"] >= 0.6


def test_refit_drops_fully_enclosed_cluster_instead_of_relocating() -> None:
    body = _part(0, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], area=10.0)
    small = _part(1, [-0.3, -0.3, -0.3], [0.3, 0.3, 0.3], area=1.0)
    small_centers = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    geometry = {
        0: _ClusterGeometry(
            0,
            np.asarray([[[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, 1.0]]]),
            np.asarray([[0.0, 0.0, 0.0]]),
            np.asarray([10.0]),
            np.asarray([-1.0, -1.0, -1.0]),
            np.asarray([1.0, 1.0, 1.0]),
        ),
        1: _ClusterGeometry(
            1,
            _triangle_faces(small_centers, radius=0.01),
            small_centers,
            np.ones(len(small_centers)),
            np.asarray([-0.3, -0.3, -0.3]),
            np.asarray([0.3, 0.3, 0.3]),
        ),
    }
    parts = [body, small]

    _constrained_refit_parallel_parts(
        parts,
        geometry,
        adjacency={(0, 1)},
        gap=0.0,
        min_coverage=0.02,
        beam_width=64,
        preserve_contact=True,
    )

    assert [part.segment_id for part in parts] == [0]
