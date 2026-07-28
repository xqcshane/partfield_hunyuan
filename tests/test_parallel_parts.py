from __future__ import annotations

import numpy as np
import trimesh

from partfield_mc.cuboid_fit import FitConfig, fit_cuboids_from_labels
from partfield_mc.models import CuboidPart
from partfield_mc.cuboid_fit import _resolve_parallel_overlaps, _trim_parallel_overlaps


def _part(name: str, segment_id: int, center, size) -> CuboidPart:
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = np.asarray(center, dtype=float)
    return CuboidPart(
        name=name,
        segment_id=segment_id,
        size=np.asarray(size, dtype=float),
        transform=transform,
        face_count=12,
        surface_area=1.0,
        source_center=np.asarray(center, dtype=float),
    )


def test_overlap_resolution_separates_parallel_parts() -> None:
    parts = [
        _part("body", 0, [0, 0, 0], [2, 2, 2]),
        _part("head", 1, [0.7, 0, 0], [1, 1, 1]),
    ]
    _resolve_parallel_overlaps(parts, gap=0.1)
    a, b = parts
    overlap = np.minimum(a.center + a.size / 2, b.center + b.size / 2) - np.maximum(
        a.center - a.size / 2,
        b.center - b.size / 2,
    )
    assert np.any(overlap <= 0)


def test_shared_mode_uses_one_rotation_for_all_parts() -> None:
    left = trimesh.creation.box(extents=[2, 1, 1])
    left.apply_translation([-1.2, 0, 0])
    right = trimesh.creation.box(extents=[1, 1, 1])
    right.apply_translation([1.2, 0, 0])
    mesh = trimesh.util.concatenate([left, right])
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(30), [0, 1, 0]))
    labels = np.zeros(len(mesh.faces), dtype=np.int64)
    labels[len(left.faces) :] = 1

    parts = fit_cuboids_from_labels(
        mesh,
        labels,
        FitConfig(fit_mode="shared", min_faces=4),
    )
    assert len(parts) == 2
    assert np.allclose(parts[0].rotation, parts[1].rotation, atol=1e-6)


def test_overlap_trimming_preserves_part_count_and_removes_volume_intersections() -> None:
    parts = [
        _part("cluster_0", 0, [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]),
        _part("cluster_1", 1, [0.8, 0.1, 0.0], [1.5, 1.5, 1.5]),
        _part("cluster_2", 2, [-0.7, -0.1, 0.0], [1.4, 1.4, 1.4]),
    ]
    count = _trim_parallel_overlaps(parts, gap=0.01)
    assert count >= 1
    assert len(parts) == 3
    assert all(np.all(part.size > 0) for part in parts)
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            overlap = np.minimum(parts[i].center + parts[i].size / 2, parts[j].center + parts[j].size / 2) - np.maximum(
                parts[i].center - parts[i].size / 2,
                parts[j].center - parts[j].size / 2,
            )
            assert np.any(overlap <= 1e-9)


def test_strict_aabb_may_drop_fully_occluded_clusters() -> None:
    meshes = []
    labels = []
    for segment_id in range(8):
        box = trimesh.creation.box(extents=[1.2, 1.0, 0.9])
        # Deliberately overlap neighbouring fitted AABBs.
        box.apply_translation([(segment_id % 4) * 0.55, (segment_id // 4) * 0.55, 0.0])
        meshes.append(box)
        labels.extend([segment_id] * len(box.faces))
    mesh = trimesh.util.concatenate(meshes)

    parts = fit_cuboids_from_labels(
        mesh,
        np.asarray(labels, dtype=np.int64),
        FitConfig(
            fit_mode="aabb",
            min_faces=1,
            resolve_overlaps=True,
            overlap_strategy="trim",
            preserve_all_labels=True,
            expected_parts=8,
        ),
    )

    assert 1 <= len(parts) <= 8
    assert {part.segment_id for part in parts}.issubset(set(range(8)))
    assert any(
        part.metadata.get("dropped_segment_ids_during_overlap_resolution")
        for part in parts
    )
    for i in range(len(parts)):
        assert np.all(parts[i].size > 0)
        for j in range(i + 1, len(parts)):
            overlap = np.minimum(parts[i].center + parts[i].size / 2, parts[j].center + parts[j].size / 2) - np.maximum(
                parts[i].center - parts[i].size / 2,
                parts[j].center - parts[j].size / 2,
            )
            assert np.any(overlap <= 1e-9)


def test_large_first_overlap_trimming_keeps_largest_box_unchanged() -> None:
    large = _part("large", 0, [0.0, 0.0, 0.0], [4.0, 3.0, 2.0])
    small = _part("small", 1, [1.6, 0.0, 0.0], [2.0, 1.5, 1.0])
    original_center = large.center.copy()
    original_size = large.size.copy()

    _trim_parallel_overlaps([small, large], gap=0.0)

    assert np.allclose(large.center, original_center)
    assert np.allclose(large.size, original_size)
    assert large.metadata["large_first_preserved"] is True
    assert small.metadata["overlap_priority_rank"] > large.metadata["overlap_priority_rank"]
    overlap = np.minimum(large.center + large.size / 2, small.center + small.size / 2) - np.maximum(
        large.center - large.size / 2,
        small.center - small.size / 2,
    )
    assert np.any(overlap <= 1e-9)


def test_fully_enclosed_small_box_is_dropped_and_large_box_remains_unchanged() -> None:
    large = _part("large", 0, [0.0, 0.0, 0.0], [6.0, 6.0, 6.0])
    small = _part("small", 1, [0.25, 0.1, -0.1], [1.0, 1.0, 1.0])
    large_center = large.center.copy()
    large_size = large.size.copy()
    parts = [large, small]
    _trim_parallel_overlaps(parts, gap=0.0)

    assert np.allclose(large.center, large_center)
    assert np.allclose(large.size, large_size)
    assert parts == [large]
    assert small.metadata["dropped_due_to_overlap"] is True
    assert small.metadata["dropped_by_protected_segment"] == large.segment_id
    assert small.metadata["volume_retention_ratio"] == 0.0
    assert not small.metadata.get("overlap_moves")
