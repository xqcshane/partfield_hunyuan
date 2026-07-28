from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import trimesh

from .models import CuboidPart


@dataclass
class FitConfig:
    fit_mode: str = "obb"
    min_area_ratio: float = 0.0
    min_faces: int = 4
    grid_divisions: int = 0
    category: str = "generic"
    forward_axis: str = "auto"
    resolve_overlaps: bool = False
    part_gap_ratio: float = 0.0
    overlap_strategy: str = "move"
    preserve_all_labels: bool = False
    expected_parts: int | None = None
    refit_min_coverage: float = 0.05
    refit_beam_width: int = 64
    refit_preserve_contact: bool = True
    semantic_refit: str = "auto"
    adaptive_split: bool = True
    max_extra_cuboids: int = 1
    protected_min_coverage: float = 0.85
    split_min_coverage_gain: float = 0.05


@dataclass
class _ClusterGeometry:
    segment_id: int
    face_vertices_local: np.ndarray
    face_centers_local: np.ndarray
    face_areas: np.ndarray
    raw_min: np.ndarray
    raw_max: np.ndarray

    @property
    def total_area(self) -> float:
        return max(float(np.sum(self.face_areas)), 1e-12)


@dataclass
class _RefitState:
    lower: np.ndarray
    upper: np.ndarray
    lower_contacts: dict[int, int]
    upper_contacts: dict[int, int]

    def clone(self) -> "_RefitState":
        return _RefitState(
            lower=np.asarray(self.lower, dtype=np.float64).copy(),
            upper=np.asarray(self.upper, dtype=np.float64).copy(),
            lower_contacts=dict(self.lower_contacts),
            upper_contacts=dict(self.upper_contacts),
        )


@dataclass
class _RefitCandidate:
    lower: np.ndarray
    upper: np.ndarray
    coverage_ratio: float
    retained_area: float
    source_center_retained: bool
    contact_segments: list[int]
    contact_area: float
    volume_ratio: float


@dataclass
class _SemanticRefitPlan:
    mode: str
    roles: dict[int, str]
    scores: dict[int, float]
    body_segment: int | None
    face_segment: int | None
    longitudinal_axis: int
    head_direction: float
    confidence: float


@dataclass
class _SplitSolution:
    candidates: list[_RefitCandidate]
    axis: int
    plane: float
    coverage_ratio: float
    contact_area: float
    score: tuple[float, ...]

def _orthonormalize(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def _fit_obb(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 4:
        raise ValueError("At least four points are required for OBB fitting")
    try:
        to_origin, extents = trimesh.bounds.oriented_bounds(points, angle_digits=1, ordered=True)
        local_to_world = np.linalg.inv(to_origin)
        local_to_world[:3, :3] = _orthonormalize(local_to_world[:3, :3])
        return local_to_world.astype(np.float64), np.asarray(extents, dtype=np.float64)
    except Exception:
        # PCA fallback for nearly planar or numerically unstable segments.
        center = points.mean(axis=0)
        centered = points - center
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        rotation = _orthonormalize(vt.T)
        local = centered @ rotation
        mins = local.min(axis=0)
        maxs = local.max(axis=0)
        extents = np.maximum(maxs - mins, 1e-5)
        local_center = (mins + maxs) * 0.5
        world_center = center + local_center @ rotation.T
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = world_center
        return transform, extents


def _fit_aabb(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = (mins + maxs) * 0.5
    return transform, np.maximum(maxs - mins, 1e-5)


def _fit_shared_aabb(
    points: np.ndarray,
    shared_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an AABB in one shared rotated frame.

    Every returned cuboid uses exactly the same rotation matrix. This is
    tighter than a world-space AABB while preventing per-part rotations.
    """

    rotation = _orthonormalize(shared_rotation)
    local = np.asarray(points, dtype=np.float64) @ rotation
    mins = local.min(axis=0)
    maxs = local.max(axis=0)
    local_center = (mins + maxs) * 0.5

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = local_center @ rotation.T
    return transform, np.maximum(maxs - mins, 1e-5)


def _resolve_parallel_overlaps(
    parts: list[CuboidPart],
    gap: float,
    max_iterations: int = 12,
) -> int:
    """Move the smaller of two overlapping parallel cuboids until separated.

    This keeps cuboid sizes and their common orientation unchanged. The larger
    part (normally the body) stays fixed while limbs/head are moved outward.
    """

    if len(parts) < 2:
        return 0

    rotation = parts[0].rotation
    for part in parts[1:]:
        if not np.allclose(part.rotation, rotation, atol=1e-6):
            raise ValueError("Overlap resolution requires all cuboids to share one rotation")

    moves = 0
    for _ in range(max_iterations):
        changed = False
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                a = parts[i]
                b = parts[j]
                ca = a.center @ rotation
                cb = b.center @ rotation
                amin = ca - a.size * 0.5
                amax = ca + a.size * 0.5
                bmin = cb - b.size * 0.5
                bmax = cb + b.size * 0.5
                overlap = np.minimum(amax, bmax) - np.maximum(amin, bmin)
                if np.any(overlap <= 1e-8):
                    continue

                axis = int(np.argmin(overlap))
                if a.volume <= b.volume:
                    moving, fixed = a, b
                    cm, cf = ca, cb
                else:
                    moving, fixed = b, a
                    cm, cf = cb, ca

                direction = float(np.sign(cm[axis] - cf[axis]))
                if direction == 0.0:
                    source_delta = moving.source_center @ rotation - fixed.source_center @ rotation
                    direction = float(np.sign(source_delta[axis]))
                if direction == 0.0:
                    direction = 1.0 if moving.segment_id > fixed.segment_id else -1.0

                local_delta = np.zeros(3, dtype=np.float64)
                local_delta[axis] = direction * (float(overlap[axis]) + max(float(gap), 0.0))
                world_delta = local_delta @ rotation.T
                moving.transform[:3, 3] += world_delta
                moving.metadata["overlap_adjusted"] = True
                moving.metadata["overlap_move"] = (
                    np.asarray(moving.metadata.get("overlap_move", [0.0, 0.0, 0.0]), dtype=float)
                    + world_delta
                ).tolist()
                moves += 1
                changed = True
        if not changed:
            break
    return moves



def _local_bounds(part: CuboidPart, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = part.center @ rotation
    half = np.asarray(part.size, dtype=np.float64) * 0.5
    return center - half, center + half


def _set_local_axis_bounds(
    part: CuboidPart,
    rotation: np.ndarray,
    axis: int,
    lower: float,
    upper: float,
) -> None:
    if upper <= lower:
        raise ValueError("Cuboid trimming produced a non-positive side length")
    local_center = part.center @ rotation
    local_center[axis] = (float(lower) + float(upper)) * 0.5
    part.transform[:3, 3] = local_center @ rotation.T
    part.size[axis] = float(upper) - float(lower)


def _parallel_overlap_depth(
    a: CuboidPart,
    b: CuboidPart,
    rotation: np.ndarray,
) -> np.ndarray:
    amin, amax = _local_bounds(a, rotation)
    bmin, bmax = _local_bounds(b, rotation)
    return np.minimum(amax, bmax) - np.maximum(amin, bmin)


def _translate_part_local(
    part: CuboidPart,
    rotation: np.ndarray,
    local_delta: np.ndarray,
) -> None:
    part.transform[:3, 3] += np.asarray(local_delta, dtype=np.float64) @ rotation.T


def _intersection_volume_with_parts(
    part: CuboidPart,
    others: list[CuboidPart],
    rotation: np.ndarray,
    tolerance: float,
) -> tuple[int, float]:
    intersections = 0
    volume = 0.0
    for other in others:
        depth = _parallel_overlap_depth(part, other, rotation)
        if np.all(depth > tolerance):
            intersections += 1
            volume += float(np.prod(depth))
    return intersections, volume


def _trim_moving_part_against_fixed(
    moving: CuboidPart,
    fixed: CuboidPart,
    rotation: np.ndarray,
    requested_gap: float,
    model_extent: float,
    tolerance: float,
) -> bool:
    """Trim only ``moving`` so it no longer intersects ``fixed``.

    The fixed cuboid is never modified.  Among all valid one-axis cuts, prefer
    a retained slab containing the moving cluster's source centroid, then the
    cut retaining the greatest volume, and finally the smallest center shift.
    """

    moving_min, moving_max = _local_bounds(moving, rotation)
    fixed_min, fixed_max = _local_bounds(fixed, rotation)
    source = np.asarray(moving.source_center, dtype=np.float64) @ rotation
    old_center = moving.center @ rotation
    old_volume = moving.volume
    old_size = np.asarray(moving.size, dtype=np.float64).copy()

    # Keep every requested cluster as a positive-volume cuboid.  This floor is
    # intentionally tiny; it is only a topology safeguard, not a target size.
    minimum_length = min(
        max(model_extent * 1e-6, 1e-8),
        max(float(np.min(old_size)) * 0.25, tolerance * 2.0),
    )
    gap = max(float(requested_gap), 0.0)
    candidates: list[tuple[int, float, float, int, float, float, str]] = []

    for axis in range(3):
        old_length = float(moving_max[axis] - moving_min[axis])
        if old_length <= tolerance:
            continue

        # Retain the negative-side slab of the moving cuboid.
        low_lower = float(moving_min[axis])
        low_upper = min(float(moving_max[axis]), float(fixed_min[axis]) - gap)
        low_length = low_upper - low_lower
        if low_length >= minimum_length:
            contains_source = int(low_lower - tolerance <= source[axis] <= low_upper + tolerance)
            retained_ratio = low_length / old_length
            new_center = (low_lower + low_upper) * 0.5
            center_shift = abs(new_center - float(old_center[axis]))
            candidates.append(
                (contains_source, retained_ratio, -center_shift, axis, low_lower, low_upper, "max")
            )

        # Retain the positive-side slab of the moving cuboid.
        high_lower = max(float(moving_min[axis]), float(fixed_max[axis]) + gap)
        high_upper = float(moving_max[axis])
        high_length = high_upper - high_lower
        if high_length >= minimum_length:
            contains_source = int(high_lower - tolerance <= source[axis] <= high_upper + tolerance)
            retained_ratio = high_length / old_length
            new_center = (high_lower + high_upper) * 0.5
            center_shift = abs(new_center - float(old_center[axis]))
            candidates.append(
                (contains_source, retained_ratio, -center_shift, axis, high_lower, high_upper, "min")
            )

    if not candidates:
        return False

    contains_source, retained_ratio, _, axis, lower, upper, trimmed_side = max(
        candidates,
        key=lambda item: (item[0], item[1], item[2], -item[3]),
    )
    before_min, before_max = _local_bounds(moving, rotation)
    _set_local_axis_bounds(moving, rotation, axis, lower, upper)
    after_min, after_max = _local_bounds(moving, rotation)

    moving.metadata["overlap_adjusted"] = True
    moving.metadata.setdefault("overlap_trims", []).append(
        {
            "protected_segment": int(fixed.segment_id),
            "axis": int(axis),
            "trimmed_side": trimmed_side,
            "trimmed_min": float(after_min[axis] - before_min[axis]),
            "trimmed_max": float(before_max[axis] - after_max[axis]),
            "source_center_retained": bool(contains_source),
            "step_volume_retention_ratio": float(moving.volume / max(old_volume, tolerance)),
        }
    )
    fixed.metadata.setdefault("preserved_against_smaller_segments", []).append(int(moving.segment_id))
    return True


def _trim_parallel_overlaps(
    parts: list[CuboidPart],
    gap: float,
) -> int:
    """Resolve overlaps with strict large-first priority.

    Cuboids are ranked once using their *original* fitted AABB volume. Larger
    cuboids become protected geometry and are never trimmed or moved because of
    a smaller cuboid. Each lower-priority cuboid is trimmed only on its own
    bounds. If it is fully enclosed and no positive outside slab exists, that
    lower-priority cuboid is dropped instead of being moved outside.
    """

    if len(parts) < 2:
        return 0

    rotation = parts[0].rotation
    for part in parts[1:]:
        if not np.allclose(part.rotation, rotation, atol=1e-6):
            raise ValueError("Overlap trimming requires all cuboids to share one rotation")

    model_extent = max(float(np.max(np.asarray(part.size, dtype=np.float64))) for part in parts)
    tolerance = max(model_extent * 1e-9, 1e-12)
    requested_gap = max(float(gap), 0.0)

    for part in parts:
        part.metadata["original_center_before_overlap"] = part.center.tolist()
        part.metadata["original_size_before_overlap"] = np.asarray(part.size, dtype=float).tolist()
        part.metadata["original_volume_before_overlap"] = float(part.volume)

    priority = sorted(
        parts,
        key=lambda part: (
            float(part.metadata["original_volume_before_overlap"]),
            float(part.surface_area),
            int(part.face_count),
            -int(part.segment_id),
        ),
        reverse=True,
    )
    for rank, part in enumerate(priority):
        part.metadata["overlap_priority_rank"] = int(rank)
        part.metadata["overlap_priority"] = "large_first_original_aabb_volume"

    protected: list[CuboidPart] = []
    adjustments = 0
    max_iterations_per_part = max(24, len(parts) * 12)
    dropped_segments: list[int] = []

    for moving in priority:
        iterations = 0
        dropped = False
        while True:
            overlapping = [
                fixed
                for fixed in protected
                if np.all(_parallel_overlap_depth(moving, fixed, rotation) > tolerance)
            ]
            if not overlapping:
                break

            # Resolve against the largest conflicting protected cuboid first.
            fixed = min(overlapping, key=lambda part: int(part.metadata["overlap_priority_rank"]))
            if _trim_moving_part_against_fixed(
                moving,
                fixed,
                rotation,
                requested_gap,
                model_extent,
                tolerance,
            ):
                adjustments += 1
            else:
                moving.metadata["overlap_adjusted"] = True
                moving.metadata["dropped_due_to_overlap"] = True
                moving.metadata["dropped_by_protected_segment"] = int(fixed.segment_id)
                moving.metadata["drop_reason"] = "fully_enclosed_or_no_positive_outside_slab"
                moving.metadata["volume_retention_ratio"] = 0.0
                dropped_segments.append(int(moving.segment_id))
                adjustments += 1
                dropped = True
                break

            iterations += 1
            if iterations > max_iterations_per_part:
                unresolved = [
                    int(fixed_part.segment_id)
                    for fixed_part in protected
                    if np.all(_parallel_overlap_depth(moving, fixed_part, rotation) > tolerance)
                ]
                raise ValueError(
                    f"Large-first overlap resolution did not converge for segment {moving.segment_id}; "
                    f"still overlaps protected segments {unresolved}"
                )

        if not dropped:
            protected.append(moving)

    parts[:] = [part for part in parts if not part.metadata.get("dropped_due_to_overlap", False)]

    remaining: list[tuple[int, int, list[float]]] = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            depth = _parallel_overlap_depth(parts[i], parts[j], rotation)
            if np.all(depth > tolerance):
                remaining.append((int(parts[i].segment_id), int(parts[j].segment_id), depth.tolist()))
    if remaining:
        raise ValueError(f"Large-first overlap resolution failed; remaining intersections: {remaining}")

    for part in parts:
        original_volume = float(part.metadata["original_volume_before_overlap"])
        part.metadata["volume_retention_ratio"] = float(part.volume / max(original_volume, tolerance))
        part.metadata["large_first_preserved"] = bool(
            np.allclose(
                part.center,
                np.asarray(part.metadata["original_center_before_overlap"], dtype=float),
                atol=tolerance,
            )
            and np.allclose(
                part.size,
                np.asarray(part.metadata["original_size_before_overlap"], dtype=float),
                atol=tolerance,
            )
        )

    for part in parts:
        part.metadata["dropped_segment_ids_during_overlap_resolution"] = dropped_segments

    return adjustments


def _label_adjacency(mesh: trimesh.Trimesh, labels: np.ndarray) -> set[tuple[int, int]]:
    """Return label pairs that share an edge in the source segmented mesh."""

    result: set[tuple[int, int]] = set()
    adjacency = np.asarray(
        getattr(mesh, "face_adjacency", np.empty((0, 2))), dtype=np.int64
    )
    for face_a, face_b in adjacency:
        label_a = int(labels[int(face_a)])
        label_b = int(labels[int(face_b)])
        if label_a == label_b:
            continue
        result.add((min(label_a, label_b), max(label_a, label_b)))
    return result


def _labels_are_adjacent(a: int, b: int, adjacency: set[tuple[int, int]]) -> bool:
    return (min(int(a), int(b)), max(int(a), int(b))) in adjacency


def _build_cluster_geometry(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    label_id: int,
    rotation: np.ndarray,
) -> _ClusterGeometry:
    face_ids = np.flatnonzero(labels == int(label_id))
    world_vertices = np.asarray(mesh.vertices, dtype=np.float64)[
        np.asarray(mesh.faces, dtype=np.int64)[face_ids]
    ]
    local_vertices = world_vertices @ rotation
    local_centers = local_vertices.mean(axis=1)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)[face_ids]
    flat = local_vertices.reshape(-1, 3)
    return _ClusterGeometry(
        segment_id=int(label_id),
        face_vertices_local=local_vertices,
        face_centers_local=local_centers,
        face_areas=areas,
        raw_min=flat.min(axis=0),
        raw_max=flat.max(axis=0),
    )


def _set_part_local_bounds(
    part: CuboidPart,
    rotation: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if np.any(upper <= lower):
        raise ValueError("Constrained AABB refit produced a non-positive side length")
    local_center = (lower + upper) * 0.5
    part.transform[:3, 3] = local_center @ rotation.T
    part.size = upper - lower


def _bounds_overlap_depth(
    lower: np.ndarray,
    upper: np.ndarray,
    fixed: CuboidPart,
    rotation: np.ndarray,
) -> np.ndarray:
    fixed_lower, fixed_upper = _local_bounds(fixed, rotation)
    return np.minimum(upper, fixed_upper) - np.maximum(lower, fixed_lower)


def _candidate_contact_metrics(
    lower: np.ndarray,
    upper: np.ndarray,
    moving_segment: int,
    protected: list[CuboidPart],
    rotation: np.ndarray,
    adjacency: set[tuple[int, int]],
    tolerance: float,
) -> tuple[list[int], float]:
    contacts: list[int] = []
    total_area = 0.0
    for fixed in protected:
        if not _labels_are_adjacent(moving_segment, fixed.segment_id, adjacency):
            continue
        fixed_lower, fixed_upper = _local_bounds(fixed, rotation)
        for axis in range(3):
            touching = (
                abs(float(upper[axis] - fixed_lower[axis])) <= tolerance
                or abs(float(lower[axis] - fixed_upper[axis])) <= tolerance
            )
            if not touching:
                continue
            other_axes = [value for value in range(3) if value != axis]
            overlap = [
                min(float(upper[k]), float(fixed_upper[k]))
                - max(float(lower[k]), float(fixed_lower[k]))
                for k in other_axes
            ]
            if overlap[0] > tolerance and overlap[1] > tolerance:
                contacts.append(int(fixed.segment_id))
                total_area += float(overlap[0] * overlap[1])
                break
    return sorted(set(contacts)), total_area


def _fit_refit_state(
    geometry: _ClusterGeometry,
    state: _RefitState,
    source_center_local: np.ndarray,
    protected: list[CuboidPart],
    rotation: np.ndarray,
    adjacency: set[tuple[int, int]],
    tolerance: float,
    minimum_length: float,
) -> _RefitCandidate | None:
    lower_limit = np.asarray(state.lower, dtype=np.float64)
    upper_limit = np.asarray(state.upper, dtype=np.float64)
    if np.any(upper_limit - lower_limit <= minimum_length):
        return None

    centers = geometry.face_centers_local
    mask = np.all(centers >= lower_limit - tolerance, axis=1) & np.all(
        centers <= upper_limit + tolerance, axis=1
    )
    if not np.any(mask):
        return None

    lower = lower_limit.copy()
    upper = upper_limit.copy()
    for _ in range(3):
        selected_vertices = geometry.face_vertices_local[mask].reshape(-1, 3)
        fitted_lower = np.maximum(selected_vertices.min(axis=0), lower_limit)
        fitted_upper = np.minimum(selected_vertices.max(axis=0), upper_limit)

        # If source labels were adjacent, extend exactly to the chosen separating
        # plane. This keeps a face-to-face connection but never creates volume overlap.
        for axis in state.lower_contacts:
            fitted_lower[int(axis)] = lower_limit[int(axis)]
        for axis in state.upper_contacts:
            fitted_upper[int(axis)] = upper_limit[int(axis)]

        if np.any(fitted_upper - fitted_lower <= minimum_length):
            return None
        lower, upper = fitted_lower, fitted_upper
        new_mask = np.all(centers >= lower - tolerance, axis=1) & np.all(
            centers <= upper + tolerance, axis=1
        )
        if not np.any(new_mask):
            return None
        if np.array_equal(mask, new_mask):
            break
        mask = new_mask

    retained_area = float(np.sum(geometry.face_areas[mask]))
    coverage_ratio = retained_area / geometry.total_area
    raw_volume = max(float(np.prod(geometry.raw_max - geometry.raw_min)), tolerance)
    volume_ratio = float(np.prod(upper - lower) / raw_volume)
    source_retained = bool(
        np.all(source_center_local >= lower - tolerance)
        and np.all(source_center_local <= upper + tolerance)
    )
    contacts, contact_area = _candidate_contact_metrics(
        lower,
        upper,
        geometry.segment_id,
        protected,
        rotation,
        adjacency,
        tolerance,
    )
    return _RefitCandidate(
        lower=lower,
        upper=upper,
        coverage_ratio=float(coverage_ratio),
        retained_area=retained_area,
        source_center_retained=source_retained,
        contact_segments=contacts,
        contact_area=float(contact_area),
        volume_ratio=volume_ratio,
    )


def _candidate_conflicts(
    candidate: _RefitCandidate,
    protected: list[CuboidPart],
    rotation: np.ndarray,
    tolerance: float,
) -> list[CuboidPart]:
    return [
        fixed
        for fixed in protected
        if np.all(
            _bounds_overlap_depth(candidate.lower, candidate.upper, fixed, rotation)
            > tolerance
        )
    ]


def _state_key(state: _RefitState) -> tuple[float, ...]:
    values = np.concatenate([state.lower, state.upper])
    return tuple(np.round(values, 12).tolist())


def _candidate_rank(
    candidate: _RefitCandidate,
    *,
    adjacent_protected: set[int],
    preserve_contact: bool,
) -> tuple[float, ...]:
    has_required_contact = bool(
        adjacent_protected.intersection(candidate.contact_segments)
    )
    return (
        float(has_required_contact if preserve_contact and adjacent_protected else True),
        float(candidate.coverage_ratio),
        float(candidate.source_center_retained),
        float(candidate.contact_area),
        float(candidate.volume_ratio),
    )


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _infer_semantic_refit_plan(
    parts: list[CuboidPart],
    adjacency: set[tuple[int, int]],
    rotation: np.ndarray,
    mode: str,
) -> _SemanticRefitPlan:
    """Infer high-level visual roles using deterministic geometry only.

    This deliberately does not use an LLM/agent.  The goal is not a perfect
    semantic segmentation; it is a stable refit priority that protects the
    visually important head/face box before the much larger torso box is
    fitted.  PartField labels remain the source of truth.
    """

    normalized_mode = str(mode).lower()
    if normalized_mode not in {"off", "auto", "animal", "person"}:
        raise ValueError("semantic_refit must be off, auto, animal, or person")

    roles = {int(part.segment_id): "generic" for part in parts}
    scores = {int(part.segment_id): 0.0 for part in parts}
    if normalized_mode == "off" or len(parts) < 2:
        return _SemanticRefitPlan(
            mode="off",
            roles=roles,
            scores=scores,
            body_segment=None,
            face_segment=None,
            longitudinal_axis=0,
            head_direction=1.0,
            confidence=0.0,
        )

    local_centers = {
        int(part.segment_id): np.asarray(part.center, dtype=np.float64) @ rotation
        for part in parts
    }
    local_bounds = {
        int(part.segment_id): _local_bounds(part, rotation) for part in parts
    }
    all_lower = np.min(np.vstack([value[0] for value in local_bounds.values()]), axis=0)
    all_upper = np.max(np.vstack([value[1] for value in local_bounds.values()]), axis=0)
    global_span = np.maximum(all_upper - all_lower, 1e-9)

    # World Y remains the canonical up direction even when shared fitting is
    # used.  The longitudinal direction is selected in the fitting frame.
    horizontal_axis = 0 if global_span[0] >= global_span[2] else 2
    horizontal_span = float(global_span[horizontal_axis])
    world_centers = np.vstack([part.center for part in parts])
    world_y_span = max(float(np.ptp(world_centers[:, 1])), 1e-9)
    horizontal_model_span = max(float(global_span[0]), float(global_span[2]))
    vertical_model_span = max(
        float(max(part.center[1] + part.size[1] * 0.5 for part in parts)
              - min(part.center[1] - part.size[1] * 0.5 for part in parts)),
        1e-9,
    )

    inferred_mode = normalized_mode
    if normalized_mode == "auto":
        inferred_mode = "person" if vertical_model_span > 1.20 * horizontal_model_span else "animal"

    # The torso is usually the largest true source-surface cluster.  AABB
    # volume alone is avoided because a curved tail can have a huge empty box.
    body = max(
        parts,
        key=lambda part: (
            float(part.surface_area),
            int(part.face_count),
            float(part.volume),
        ),
    )
    body_id = int(body.segment_id)
    roles[body_id] = "body"
    scores[body_id] = 1.0

    remaining = [part for part in parts if part is not body]
    if not remaining:
        return _SemanticRefitPlan(
            mode=inferred_mode,
            roles=roles,
            scores=scores,
            body_segment=body_id,
            face_segment=None,
            longitudinal_axis=horizontal_axis,
            head_direction=1.0,
            confidence=0.0,
        )

    if inferred_mode == "person":
        # For an upright person the highest sizeable adjacent component is the
        # head.  Surface area prevents a tiny hair/ear cluster from winning.
        max_other_area = max(float(part.surface_area) for part in remaining)
        ranked: list[tuple[float, CuboidPart]] = []
        for part in remaining:
            area_score = _clamp01(float(part.surface_area) / max(max_other_area, 1e-12))
            y_score = _clamp01(
                (float(part.center[1]) - float(np.min(world_centers[:, 1]))) / world_y_span
            )
            adjacent_score = 1.0 if _labels_are_adjacent(part.segment_id, body_id, adjacency) else 0.0
            compactness = float(np.min(part.size) / max(float(np.max(part.size)), 1e-12))
            score = 2.8 * y_score + 1.2 * area_score + 0.8 * adjacent_score + 0.4 * compactness
            ranked.append((score, part))
    else:
        max_other_area = max(float(part.surface_area) for part in remaining)
        body_local = local_centers[body_id]
        body_height = max(float(body.size[1]), 1e-9)
        ranked = []
        for part in remaining:
            segment_id = int(part.segment_id)
            center = local_centers[segment_id]
            delta_long = float(center[horizontal_axis] - body_local[horizontal_axis])
            end_score = _clamp01(abs(delta_long) / max(0.5 * horizontal_span, 1e-9))
            area_ratio = float(part.surface_area) / max(float(body.surface_area), 1e-12)
            area_score = _clamp01(area_ratio / 0.35)
            relative_area = _clamp01(float(part.surface_area) / max(max_other_area, 1e-12))
            compactness = float(np.min(part.size) / max(float(np.max(part.size)), 1e-12))
            elongation = float(np.max(part.size) / max(float(np.min(part.size)), 1e-12))
            slender_penalty = _clamp01((elongation - 2.5) / 4.0)
            vertical_score = _clamp01(
                (float(part.center[1]) - (float(body.center[1]) - 0.35 * body_height))
                / max(0.85 * body_height, 1e-9)
            )
            adjacent_score = 1.0 if _labels_are_adjacent(segment_id, body_id, adjacency) else 0.0
            # A face/head is normally a compact, sizeable, elevated endpoint.
            # A tail can also be an endpoint, but is usually much more slender.
            score = (
                2.25 * end_score
                + 1.55 * compactness
                + 1.35 * vertical_score
                + 1.10 * area_score
                + 0.55 * relative_area
                + 0.90 * adjacent_score
                - 1.35 * slender_penalty
            )
            ranked.append((score, part))

    ranked.sort(key=lambda item: (item[0], item[1].surface_area), reverse=True)
    best_score, face = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else best_score - 1.0
    face_id = int(face.segment_id)
    roles[face_id] = "face"
    scores[face_id] = float(best_score)
    confidence = _clamp01(0.5 + 0.25 * (best_score - second_score))

    face_local = local_centers[face_id]
    body_local = local_centers[body_id]
    head_direction = float(np.sign(face_local[horizontal_axis] - body_local[horizontal_axis]))
    if head_direction == 0.0:
        head_direction = 1.0

    # Mark secondary head-side pieces and likely limbs/tail.  These roles are
    # lower priority than the primary face, but are useful in metadata/naming.
    for part in remaining:
        segment_id = int(part.segment_id)
        if segment_id == face_id:
            continue
        center = local_centers[segment_id]
        side = float((center[horizontal_axis] - body_local[horizontal_axis]) * head_direction)
        below_body = float(part.center[1]) < float(body.center[1]) - 0.15 * max(float(body.size[1]), 1e-9)
        if below_body:
            roles[segment_id] = "limb"
        elif side > 0.0 and (
            _labels_are_adjacent(segment_id, face_id, adjacency)
            or float(part.center[1]) >= float(face.center[1]) - 0.35 * max(float(face.size[1]), 1e-9)
        ):
            roles[segment_id] = "head_aux"
        elif side < 0.0:
            elongation = float(np.max(part.size) / max(float(np.min(part.size)), 1e-12))
            roles[segment_id] = "tail" if elongation >= 1.8 else "rear_aux"

    return _SemanticRefitPlan(
        mode=inferred_mode,
        roles=roles,
        scores=scores,
        body_segment=body_id,
        face_segment=face_id,
        longitudinal_axis=horizontal_axis,
        head_direction=head_direction,
        confidence=confidence,
    )


def _semantic_priority_key(part: CuboidPart, plan: _SemanticRefitPlan) -> tuple[float, ...]:
    role = plan.roles.get(int(part.segment_id), "generic")
    role_priority = {
        "face": 100.0,
        "head_aux": 88.0,
        "body": 80.0,
        "tail": 45.0,
        "rear_aux": 40.0,
        "limb": 35.0,
        "generic": 30.0,
    }.get(role, 30.0)
    return (
        role_priority,
        float(plan.scores.get(int(part.segment_id), 0.0)),
        float(part.surface_area),
        int(part.face_count),
        float(part.volume),
        -int(part.segment_id),
    )


def _split_plane_candidates(
    geometry: _ClusterGeometry,
    protected: list[CuboidPart],
    rotation: np.ndarray,
    preferred_axis: int,
    tolerance: float,
) -> list[tuple[int, float]]:
    candidates: list[tuple[int, float]] = []
    axis_order = [preferred_axis] + [axis for axis in range(3) if axis != preferred_axis]
    for axis in axis_order:
        coords = np.asarray(geometry.face_centers_local[:, axis], dtype=np.float64)
        if len(coords) < 2 or float(np.ptp(coords)) <= tolerance:
            continue
        for quantile in (0.30, 0.40, 0.50, 0.60, 0.70):
            candidates.append((axis, float(np.quantile(coords, quantile))))
        # Protected boundaries are especially useful for an L-shaped torso:
        # one box can remain below/in front of the face and the other behind it.
        for fixed in protected:
            fixed_lower, fixed_upper = _local_bounds(fixed, rotation)
            for value in (float(fixed_lower[axis]), float(fixed_upper[axis])):
                if geometry.raw_min[axis] + tolerance < value < geometry.raw_max[axis] - tolerance:
                    candidates.append((axis, value))

    unique: dict[tuple[int, float], tuple[int, float]] = {}
    for axis, plane in candidates:
        key = (int(axis), round(float(plane), 9))
        unique[key] = (int(axis), float(plane))
    return list(unique.values())


def _split_contact_area(a: _RefitCandidate, b: _RefitCandidate, axis: int) -> float:
    other_axes = [value for value in range(3) if value != axis]
    overlap = [
        min(float(a.upper[k]), float(b.upper[k])) - max(float(a.lower[k]), float(b.lower[k]))
        for k in other_axes
    ]
    if overlap[0] <= 0.0 or overlap[1] <= 0.0:
        return 0.0
    return float(overlap[0] * overlap[1])


def _refit_one_part_nonoverlapping(
    moving: CuboidPart,
    geometry: _ClusterGeometry,
    protected: list[CuboidPart],
    rotation: np.ndarray,
    adjacency: set[tuple[int, int]],
    gap: float,
    min_coverage: float,
    beam_width: int,
    preserve_contact: bool,
    model_extent: float,
    tolerance: float,
    initial_state: _RefitState | None = None,
) -> _RefitCandidate | None:
    if not protected and initial_state is None:
        lower, upper = _local_bounds(moving, rotation)
        return _RefitCandidate(
            lower=lower,
            upper=upper,
            coverage_ratio=1.0,
            retained_area=geometry.total_area,
            source_center_retained=True,
            contact_segments=[],
            contact_area=0.0,
            volume_ratio=1.0,
        )

    minimum_length = max(model_extent * 1e-6, 1e-8)
    initial = (
        initial_state.clone()
        if initial_state is not None
        else _RefitState(
            lower=geometry.raw_min.copy(),
            upper=geometry.raw_max.copy(),
            lower_contacts={},
            upper_contacts={},
        )
    )
    frontier = [initial]
    visited: set[tuple[float, ...]] = set()
    solutions: list[_RefitCandidate] = []
    adjacent_protected = {
        int(fixed.segment_id)
        for fixed in protected
        if _labels_are_adjacent(moving.segment_id, fixed.segment_id, adjacency)
    }
    max_depth = max(6, len(protected) * 4)

    for _ in range(max_depth):
        next_states: list[tuple[tuple[float, ...], _RefitState]] = []
        for state in frontier:
            key = _state_key(state)
            if key in visited:
                continue
            visited.add(key)
            candidate = _fit_refit_state(
                geometry,
                state,
                np.asarray(moving.source_center, dtype=np.float64) @ rotation,
                protected,
                rotation,
                adjacency,
                tolerance,
                minimum_length,
            )
            if candidate is None:
                continue
            conflicts = _candidate_conflicts(candidate, protected, rotation, tolerance)
            if not conflicts:
                if candidate.coverage_ratio + tolerance >= min_coverage:
                    solutions.append(candidate)
                continue

            fixed = conflicts[0]
            fixed_lower, fixed_upper = _local_bounds(fixed, rotation)
            adjacent = _labels_are_adjacent(moving.segment_id, fixed.segment_id, adjacency)
            for axis in range(3):
                negative = state.clone()
                negative.upper[axis] = min(
                    float(negative.upper[axis]), float(fixed_lower[axis] - gap)
                )
                # Refit the surviving source-supported slab up to the chosen
                # separating plane. For adjacent labels this creates contact;
                # for non-adjacent labels it merely avoids a degenerate zero-thickness slab.
                negative.upper_contacts[axis] = int(fixed.segment_id)
                if negative.upper[axis] - negative.lower[axis] > minimum_length:
                    fitted = _fit_refit_state(
                        geometry,
                        negative,
                        np.asarray(moving.source_center, dtype=np.float64) @ rotation,
                        protected,
                        rotation,
                        adjacency,
                        tolerance,
                        minimum_length,
                    )
                    if fitted is not None:
                        next_states.append(
                            (
                                _candidate_rank(
                                    fitted,
                                    adjacent_protected=adjacent_protected,
                                    preserve_contact=preserve_contact,
                                ),
                                negative,
                            )
                        )

                positive = state.clone()
                positive.lower[axis] = max(
                    float(positive.lower[axis]), float(fixed_upper[axis] + gap)
                )
                positive.lower_contacts[axis] = int(fixed.segment_id)
                if positive.upper[axis] - positive.lower[axis] > minimum_length:
                    fitted = _fit_refit_state(
                        geometry,
                        positive,
                        np.asarray(moving.source_center, dtype=np.float64) @ rotation,
                        protected,
                        rotation,
                        adjacency,
                        tolerance,
                        minimum_length,
                    )
                    if fitted is not None:
                        next_states.append(
                            (
                                _candidate_rank(
                                    fitted,
                                    adjacent_protected=adjacent_protected,
                                    preserve_contact=preserve_contact,
                                ),
                                positive,
                            )
                        )

        if solutions:
            connected = [
                candidate
                for candidate in solutions
                if adjacent_protected.intersection(candidate.contact_segments)
            ]
            if not (preserve_contact and adjacent_protected) or connected:
                break
        if not next_states:
            break
        next_states.sort(key=lambda item: item[0], reverse=True)
        frontier = [state for _, state in next_states[: max(1, int(beam_width))]]

    if not solutions:
        return None
    connected_solutions = [
        candidate
        for candidate in solutions
        if adjacent_protected.intersection(candidate.contact_segments)
    ]
    selectable = (
        connected_solutions
        if preserve_contact and adjacent_protected and connected_solutions
        else solutions
    )
    return max(
        selectable,
        key=lambda candidate: _candidate_rank(
            candidate,
            adjacent_protected=adjacent_protected,
            preserve_contact=preserve_contact,
        ),
    )


def _refit_part_as_two_nonoverlapping_boxes(
    moving: CuboidPart,
    geometry: _ClusterGeometry,
    protected: list[CuboidPart],
    rotation: np.ndarray,
    adjacency: set[tuple[int, int]],
    gap: float,
    min_coverage: float,
    beam_width: int,
    preserve_contact: bool,
    model_extent: float,
    tolerance: float,
    preferred_axis: int,
) -> _SplitSolution | None:
    """Approximate one source cluster with two touching AABBs.

    The two search domains are separated by one plane, so the resulting boxes
    can touch but can never overlap.  Both boxes are independently refitted
    against already protected semantic regions such as the face/head.
    """

    minimum_length = max(model_extent * 1e-6, 1e-8)
    best: _SplitSolution | None = None
    for axis, plane in _split_plane_candidates(
        geometry,
        protected,
        rotation,
        preferred_axis,
        tolerance,
    ):
        lower_state = _RefitState(
            lower=geometry.raw_min.copy(),
            upper=geometry.raw_max.copy(),
            lower_contacts={},
            upper_contacts={axis: -1},
        )
        lower_state.upper[axis] = min(float(lower_state.upper[axis]), float(plane))
        upper_state = _RefitState(
            lower=geometry.raw_min.copy(),
            upper=geometry.raw_max.copy(),
            lower_contacts={axis: -1},
            upper_contacts={},
        )
        upper_state.lower[axis] = max(float(upper_state.lower[axis]), float(plane))
        if (
            lower_state.upper[axis] - lower_state.lower[axis] <= minimum_length
            or upper_state.upper[axis] - upper_state.lower[axis] <= minimum_length
        ):
            continue

        half_min_coverage = max(0.005, min(float(min_coverage) * 0.35, 0.05))
        first = _refit_one_part_nonoverlapping(
            moving,
            geometry,
            protected,
            rotation,
            adjacency,
            gap,
            half_min_coverage,
            beam_width,
            preserve_contact,
            model_extent,
            tolerance,
            initial_state=lower_state,
        )
        second = _refit_one_part_nonoverlapping(
            moving,
            geometry,
            protected,
            rotation,
            adjacency,
            gap,
            half_min_coverage,
            beam_width,
            preserve_contact,
            model_extent,
            tolerance,
            initial_state=upper_state,
        )
        if first is None or second is None:
            continue

        # The forced split plane must remain a real shared rectangle, not only
        # a line/point contact, otherwise the torso would visibly break apart.
        contact_area = _split_contact_area(first, second, axis)
        if contact_area <= tolerance * tolerance:
            continue
        if abs(float(first.upper[axis] - second.lower[axis])) > tolerance * 4.0:
            continue

        coverage = min(1.0, float(first.coverage_ratio + second.coverage_ratio))
        retained_center = float(first.source_center_retained or second.source_center_retained)
        preferred = float(axis == preferred_axis)
        volume_ratio = float(first.volume_ratio + second.volume_ratio)
        score = (
            coverage,
            retained_center,
            preferred,
            contact_area,
            volume_ratio,
        )
        solution = _SplitSolution(
            candidates=[first, second],
            axis=int(axis),
            plane=float(plane),
            coverage_ratio=coverage,
            contact_area=float(contact_area),
            score=score,
        )
        if best is None or solution.score > best.score:
            best = solution
    return best


def _split_suffixes(
    axis: int,
    rotation: np.ndarray,
    head_direction: float,
) -> tuple[str, str]:
    # For AABB fitting, local X/Z are world-horizontal and Y is up.  Shared
    # rotation still receives stable generic names if the axis is ambiguous.
    if axis == 1:
        return "lower", "upper"
    if axis in {0, 2}:
        if head_direction >= 0.0:
            return "rear", "front"
        return "front", "rear"
    return "a", "b"


def _parts_from_split_solution(
    parent: CuboidPart,
    solution: _SplitSolution,
    rotation: np.ndarray,
    plan: _SemanticRefitPlan,
) -> list[CuboidPart]:
    suffixes = _split_suffixes(solution.axis, rotation, plan.head_direction)
    result: list[CuboidPart] = []
    for index, (candidate, suffix) in enumerate(zip(solution.candidates, suffixes)):
        child = copy.deepcopy(parent)
        _set_part_local_bounds(child, rotation, candidate.lower, candidate.upper)
        child.name = f"body_{suffix}" if plan.roles.get(int(parent.segment_id)) == "body" else f"{parent.name}_{suffix}"
        child.metadata.update(
            {
                "semantic_name": child.name,
                "adaptive_split": True,
                "parent_segment_id": int(parent.segment_id),
                "cuboid_index_within_segment": int(index),
                "split_axis": int(solution.axis),
                "split_plane_local": float(solution.plane),
                "split_contact_area": float(solution.contact_area),
                "split_combined_coverage_ratio": float(solution.coverage_ratio),
                "constrained_refit": True,
                "constrained_refit_coverage_ratio": float(candidate.coverage_ratio),
                "constrained_refit_retained_surface_area": float(candidate.retained_area),
                "constrained_refit_source_center_retained": bool(candidate.source_center_retained),
                "constrained_refit_contact_segments": candidate.contact_segments,
                "constrained_refit_contact_area": float(candidate.contact_area),
                "constrained_refit_volume_ratio": float(candidate.volume_ratio),
                "constrained_refit_bounds_local": {
                    "min": candidate.lower.tolist(),
                    "max": candidate.upper.tolist(),
                },
            }
        )
        result.append(child)
    return result


def _constrained_refit_parallel_parts(
    parts: list[CuboidPart],
    cluster_geometry: dict[int, _ClusterGeometry],
    adjacency: set[tuple[int, int]],
    gap: float,
    min_coverage: float,
    beam_width: int,
    preserve_contact: bool,
    semantic_refit: str = "off",
    adaptive_split: bool = False,
    max_extra_cuboids: int = 0,
    protected_min_coverage: float = 0.85,
    split_min_coverage_gain: float = 0.05,
) -> int:
    """Refit parallel AABBs under semantic and non-overlap constraints.

    When semantic refit is enabled, the primary face/head component is fixed
    before the torso.  If one torso AABB cannot preserve enough source surface
    without entering that protected face box, the torso may be represented by
    two touching AABBs.  This is deterministic geometry optimisation, not an
    LLM agent.
    """

    if len(parts) < 2:
        return 0
    rotation = parts[0].rotation
    for part in parts[1:]:
        if not np.allclose(part.rotation, rotation, atol=1e-6):
            raise ValueError("Constrained refit requires all cuboids to share one rotation")

    if max_extra_cuboids < 0:
        raise ValueError("max_extra_cuboids must be >= 0")
    if not 0.0 <= protected_min_coverage <= 1.0:
        raise ValueError("protected_min_coverage must be between 0 and 1")
    if split_min_coverage_gain < 0.0:
        raise ValueError("split_min_coverage_gain must be >= 0")

    model_extent = max(float(np.max(part.size)) for part in parts)
    tolerance = max(model_extent * 1e-9, 1e-12)
    for part in parts:
        part.metadata["raw_aabb_center"] = part.center.tolist()
        part.metadata["raw_aabb_size"] = np.asarray(part.size, dtype=float).tolist()
        part.metadata["raw_aabb_volume"] = float(part.volume)
        part.metadata["source_surface_area"] = float(part.surface_area)

    plan = _infer_semantic_refit_plan(parts, adjacency, rotation, semantic_refit)
    semantic_enabled = plan.mode != "off"
    if semantic_enabled:
        priority = sorted(parts, key=lambda part: _semantic_priority_key(part, plan), reverse=True)
        priority_name = "semantic_face_first_then_source_surface_area"
    else:
        # Backwards-compatible fallback used by direct unit tests and by users
        # who explicitly disable semantic refitting.
        priority = sorted(
            parts,
            key=lambda part: (
                float(part.surface_area),
                int(part.face_count),
                float(part.volume),
                -int(part.segment_id),
            ),
            reverse=True,
        )
        priority_name = "source_surface_area"

    for rank, part in enumerate(priority):
        role = plan.roles.get(int(part.segment_id), "generic")
        part.metadata["semantic_refit_mode"] = plan.mode
        part.metadata["semantic_role"] = role
        part.metadata["semantic_role_score"] = float(plan.scores.get(int(part.segment_id), 0.0))
        part.metadata["semantic_plan_confidence"] = float(plan.confidence)
        part.metadata["semantic_face_segment"] = plan.face_segment
        part.metadata["semantic_body_segment"] = plan.body_segment
        part.metadata["protected_visual_region"] = bool(role == "face")
        part.metadata["constrained_refit_priority_rank"] = int(rank)
        part.metadata["constrained_refit_priority"] = priority_name

    protected: list[CuboidPart] = []
    output_parts: list[CuboidPart] = []
    dropped: list[int] = []
    changed = 0
    extra_cuboids_used = 0
    gap_value = max(float(gap), 0.0)
    min_coverage_value = max(float(min_coverage), 0.0)
    beam_width_value = max(int(beam_width), 1)

    for moving in priority:
        segment_id = int(moving.segment_id)
        geometry = cluster_geometry[segment_id]
        role = plan.roles.get(segment_id, "generic")
        candidate = _refit_one_part_nonoverlapping(
            moving,
            geometry,
            protected,
            rotation,
            adjacency,
            gap_value,
            min_coverage_value,
            beam_width_value,
            bool(preserve_contact),
            model_extent,
            tolerance,
        )

        split_solution: _SplitSolution | None = None
        split_allowed = (
            bool(adaptive_split)
            and role == "body"
            and len(priority) >= 3
            and plan.confidence >= 0.35
            and extra_cuboids_used < int(max_extra_cuboids)
            and len(geometry.face_centers_local) >= 4
        )
        if split_allowed:
            split_solution = _refit_part_as_two_nonoverlapping_boxes(
                moving,
                geometry,
                protected,
                rotation,
                adjacency,
                gap_value,
                min_coverage_value,
                beam_width_value,
                bool(preserve_contact),
                model_extent,
                tolerance,
                preferred_axis=plan.longitudinal_axis,
            )

        single_coverage = float(candidate.coverage_ratio) if candidate is not None else 0.0
        split_coverage = float(split_solution.coverage_ratio) if split_solution is not None else 0.0
        face_is_protected = bool(
            plan.face_segment is not None
            and any(int(part.segment_id) == int(plan.face_segment) for part in protected)
        )
        choose_split = bool(
            split_solution is not None
            and (
                candidate is None
                or (
                    face_is_protected
                    and single_coverage + tolerance < float(protected_min_coverage)
                    and split_coverage >= single_coverage + float(split_min_coverage_gain) - tolerance
                )
                or split_coverage >= single_coverage + max(float(split_min_coverage_gain), 0.02) - tolerance
            )
        )

        if choose_split and split_solution is not None:
            children = _parts_from_split_solution(moving, split_solution, rotation, plan)
            for child in children:
                child.metadata["semantic_refit_mode"] = plan.mode
                child.metadata["semantic_role"] = role
                child.metadata["semantic_role_score"] = float(plan.scores.get(segment_id, 0.0))
                child.metadata["semantic_plan_confidence"] = float(plan.confidence)
                child.metadata["semantic_face_segment"] = plan.face_segment
                child.metadata["semantic_body_segment"] = plan.body_segment
                child.metadata["protected_visual_region"] = False
                child.metadata["constrained_refit_priority_rank"] = moving.metadata[
                    "constrained_refit_priority_rank"
                ]
                child.metadata["constrained_refit_priority"] = priority_name
                child.metadata["single_box_coverage_before_split"] = single_coverage
                child.metadata["split_coverage_gain"] = float(split_coverage - single_coverage)
            protected.extend(children)
            output_parts.extend(children)
            extra_cuboids_used += len(children) - 1
            changed += 1
            continue

        if candidate is None:
            moving.metadata["dropped_due_to_constrained_refit"] = True
            moving.metadata["drop_reason"] = "no_nonoverlapping_candidate_above_min_coverage"
            moving.metadata["constrained_refit_coverage_ratio"] = 0.0
            dropped.append(segment_id)
            changed += 1
            continue

        old_center = moving.center.copy()
        old_size = np.asarray(moving.size, dtype=np.float64).copy()
        _set_part_local_bounds(moving, rotation, candidate.lower, candidate.upper)
        moving.metadata["constrained_refit"] = True
        moving.metadata["adaptive_split"] = False
        moving.metadata["parent_segment_id"] = segment_id
        moving.metadata["cuboid_index_within_segment"] = 0
        moving.metadata["constrained_refit_coverage_ratio"] = float(candidate.coverage_ratio)
        moving.metadata["constrained_refit_retained_surface_area"] = float(candidate.retained_area)
        moving.metadata["constrained_refit_source_center_retained"] = bool(candidate.source_center_retained)
        moving.metadata["constrained_refit_contact_segments"] = candidate.contact_segments
        moving.metadata["constrained_refit_contact_area"] = float(candidate.contact_area)
        moving.metadata["constrained_refit_volume_ratio"] = float(candidate.volume_ratio)
        moving.metadata["constrained_refit_bounds_local"] = {
            "min": candidate.lower.tolist(),
            "max": candidate.upper.tolist(),
        }
        if role == "face":
            moving.name = "face_head"
            moving.metadata["semantic_name"] = moving.name
        elif role == "body":
            moving.name = "body"
            moving.metadata["semantic_name"] = moving.name
        if not (
            np.allclose(old_center, moving.center, atol=tolerance)
            and np.allclose(old_size, moving.size, atol=tolerance)
        ):
            changed += 1
        protected.append(moving)
        output_parts.append(moving)

    parts[:] = output_parts
    remaining = []
    for index, part_a in enumerate(parts):
        for part_b in parts[index + 1 :]:
            depth = _parallel_overlap_depth(part_a, part_b, rotation)
            if np.all(depth > tolerance):
                remaining.append(
                    (
                        int(part_a.segment_id),
                        int(part_b.segment_id),
                        depth.tolist(),
                    )
                )
    if remaining:
        raise ValueError(
            "Constrained AABB refit failed; remaining positive-volume overlaps: "
            f"{remaining}"
        )

    unique_segments = sorted({int(part.segment_id) for part in parts})
    for part in parts:
        part.metadata["dropped_segment_ids_during_constrained_refit"] = sorted(set(dropped))
        part.metadata["nonoverlap_constraint_satisfied"] = True
        part.metadata["adaptive_split_enabled"] = bool(adaptive_split)
        part.metadata["extra_cuboids_used"] = int(extra_cuboids_used)
        part.metadata["output_cuboid_count"] = int(len(parts))
        part.metadata["output_unique_segment_count"] = int(len(unique_segments))
    return changed

def _snap_parts(parts: list[CuboidPart], grid_divisions: int, global_extent: float) -> None:
    if grid_divisions <= 0:
        return
    step = global_extent / float(grid_divisions)
    if step <= 0:
        return
    for part in parts:
        part.size = np.maximum(step, np.round(part.size / step) * step)
        part.transform[:3, 3] = np.round(part.transform[:3, 3] / step) * step
        part.metadata["grid_step"] = step


def fit_cuboids_from_labels(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    config: FitConfig,
) -> list[CuboidPart]:
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    if len(labels) != len(mesh.faces):
        raise ValueError(
            f"Label count ({len(labels)}) does not match face count ({len(mesh.faces)}). "
            "Use PartField's normalized input_*.ply with its matching cluster_out/*.npy."
        )
    if config.fit_mode not in {"obb", "aabb", "shared"}:
        raise ValueError("fit_mode must be obb, aabb, or shared")
    if config.part_gap_ratio < 0:
        raise ValueError("part_gap_ratio must be >= 0")
    if config.overlap_strategy not in {"move", "trim", "refit"}:
        raise ValueError("overlap_strategy must be move, trim, or refit")
    if not 0.0 <= config.refit_min_coverage <= 1.0:
        raise ValueError("refit_min_coverage must be between 0 and 1")
    if config.refit_beam_width < 1:
        raise ValueError("refit_beam_width must be >= 1")
    if config.semantic_refit not in {"off", "auto", "animal", "person"}:
        raise ValueError("semantic_refit must be off, auto, animal, or person")
    if config.max_extra_cuboids < 0:
        raise ValueError("max_extra_cuboids must be >= 0")
    if not 0.0 <= config.protected_min_coverage <= 1.0:
        raise ValueError("protected_min_coverage must be between 0 and 1")
    if config.split_min_coverage_gain < 0.0:
        raise ValueError("split_min_coverage_gain must be >= 0")

    unique_labels = sorted(np.unique(labels).tolist())
    if config.expected_parts is not None and len(unique_labels) != config.expected_parts:
        raise ValueError(
            f"Requested {config.expected_parts} clusters, but the PartField label file contains "
            f"{len(unique_labels)} non-empty labels: {unique_labels}"
        )

    shared_rotation = np.eye(3, dtype=np.float64)
    if config.fit_mode == "shared":
        global_transform, _ = _fit_obb(np.asarray(mesh.vertices, dtype=np.float64))
        shared_rotation = global_transform[:3, :3]

    cluster_geometry: dict[int, _ClusterGeometry] = {}
    label_adjacency: set[tuple[int, int]] = set()
    if config.resolve_overlaps and config.overlap_strategy == "refit":
        fitting_rotation = (
            shared_rotation
            if config.fit_mode == "shared"
            else np.eye(3, dtype=np.float64)
        )
        cluster_geometry = {
            int(label_id): _build_cluster_geometry(
                mesh, labels, int(label_id), fitting_rotation
            )
            for label_id in unique_labels
        }
        label_adjacency = _label_adjacency(mesh, labels)

    total_area = max(float(np.sum(mesh.area_faces)), 1e-12)
    parts: list[CuboidPart] = []
    for label_id in unique_labels:
        face_ids = np.flatnonzero(labels == label_id)
        if not config.preserve_all_labels and len(face_ids) < config.min_faces:
            continue
        area = float(np.sum(mesh.area_faces[face_ids]))
        if not config.preserve_all_labels and area / total_area < config.min_area_ratio:
            continue

        vertex_ids = np.unique(np.asarray(mesh.faces[face_ids], dtype=np.int64).reshape(-1))
        points = np.asarray(mesh.vertices[vertex_ids], dtype=np.float64)
        minimum_points = 1 if config.fit_mode in {"aabb", "shared"} else 4
        if len(points) < minimum_points:
            if config.preserve_all_labels:
                raise ValueError(
                    f"Cluster {label_id} has only {len(points)} unique vertices and cannot be fitted "
                    f"with fit_mode={config.fit_mode}"
                )
            continue

        if config.fit_mode == "obb":
            transform, size = _fit_obb(points)
        elif config.fit_mode == "aabb":
            transform, size = _fit_aabb(points)
        else:
            transform, size = _fit_shared_aabb(points, shared_rotation)

        part = CuboidPart(
            name=f"part_{label_id:02d}",
            segment_id=int(label_id),
            size=np.maximum(size, 1e-5),
            transform=transform,
            face_count=int(len(face_ids)),
            surface_area=area,
            source_center=points.mean(axis=0),
            metadata={"area_ratio": area / total_area},
        )
        parts.append(part)

    if not parts:
        raise ValueError("No cuboids survived the segment filters")
    if config.expected_parts is not None and len(parts) != config.expected_parts:
        raise ValueError(
            f"Expected exactly {config.expected_parts} fitted cuboids, but produced {len(parts)}. "
            "Use --fit-mode aabb and avoid cluster filtering for strict cluster preservation."
        )

    global_extent = float((mesh.bounds[1] - mesh.bounds[0]).max())
    _snap_parts(parts, config.grid_divisions, global_extent)
    overlap_moves = 0
    if config.resolve_overlaps:
        if config.fit_mode == "obb":
            raise ValueError(
                "--resolve-overlaps requires --fit-mode aabb or shared because "
                "independently rotated OBBs do not share one coordinate frame"
            )
        if config.overlap_strategy == "trim":
            overlap_moves = _trim_parallel_overlaps(
                parts,
                gap=global_extent * config.part_gap_ratio,
            )
        elif config.overlap_strategy == "refit":
            overlap_moves = _constrained_refit_parallel_parts(
                parts,
                cluster_geometry=cluster_geometry,
                adjacency=label_adjacency,
                gap=global_extent * config.part_gap_ratio,
                min_coverage=config.refit_min_coverage,
                beam_width=config.refit_beam_width,
                preserve_contact=config.refit_preserve_contact,
                semantic_refit=config.semantic_refit,
                adaptive_split=config.adaptive_split,
                max_extra_cuboids=config.max_extra_cuboids,
                protected_min_coverage=config.protected_min_coverage,
                split_min_coverage_gain=config.split_min_coverage_gain,
            )
        else:
            overlap_moves = _resolve_parallel_overlaps(
                parts,
                gap=global_extent * config.part_gap_ratio,
            )
    for part in parts:
        part.metadata["shared_orientation"] = config.fit_mode in {"aabb", "shared"}
        part.metadata["overlap_resolution_enabled"] = bool(config.resolve_overlaps)
        part.metadata["overlap_strategy"] = config.overlap_strategy if config.resolve_overlaps else "none"
        part.metadata["part_gap_ratio"] = float(config.part_gap_ratio)
        part.metadata["overlap_adjustment_count_total"] = int(overlap_moves)
        part.metadata["overlap_move_count_total"] = (
            int(overlap_moves) if config.resolve_overlaps and config.overlap_strategy == "move" else 0
        )
        unique_output_segments = {int(item.segment_id) for item in parts}
        part.metadata["cluster_preserved"] = bool(
            config.preserve_all_labels
            and (
                config.expected_parts is None
                or len(unique_output_segments) == config.expected_parts
            )
        )
    assign_part_names(parts, category=config.category, forward_axis=config.forward_axis)
    # Semantic refit names are more important than the coarse post-hoc naming
    # heuristic.  Re-apply them after assign_part_names.
    for part in parts:
        semantic_name = part.metadata.get("semantic_name")
        if semantic_name:
            part.name = str(semantic_name)
            continue
        role = part.metadata.get("semantic_role")
        if role == "face":
            part.name = "face_head"
        elif role == "body" and not part.metadata.get("adaptive_split", False):
            part.name = "body"
    return parts


def _axis_index(axis: str, parts: Iterable[CuboidPart]) -> tuple[int, float]:
    normalized = axis.lower()
    mapping = {"+x": (0, 1.0), "-x": (0, -1.0), "+z": (2, 1.0), "-z": (2, -1.0)}
    if normalized in mapping:
        return mapping[normalized]

    centers = np.vstack([part.center for part in parts])
    span_x = float(np.ptp(centers[:, 0]))
    span_z = float(np.ptp(centers[:, 2]))
    return (0, 1.0) if span_x >= span_z else (2, 1.0)


def assign_part_names(parts: list[CuboidPart], category: str, forward_axis: str = "auto") -> None:
    """Assign best-effort names; PartField itself is class-agnostic.

    Geometry is never changed by this function. Names are hints only and the
    segment_id remains the authoritative identity.
    """

    if category not in {"generic", "animal", "person", "auto"}:
        raise ValueError("category must be generic, animal, person, or auto")
    if category == "generic":
        return

    # Body is usually the segment with the largest fitted volume, using area as
    # a tie breaker.
    body = max(parts, key=lambda p: (p.volume, p.surface_area))
    body.name = "body"
    remaining = [part for part in parts if part is not body]
    if not remaining:
        return

    if category == "auto":
        all_centers = np.vstack([p.center for p in parts])
        y_span = float(np.ptp(all_centers[:, 1]))
        horizontal_span = max(float(np.ptp(all_centers[:, 0])), float(np.ptp(all_centers[:, 2])))
        category = "person" if y_span > 1.2 * horizontal_span else "animal"

    if category == "person":
        head = max(remaining, key=lambda p: p.center[1])
        head.name = "head"
        rest = [p for p in remaining if p is not head]
        left_right = sorted(rest, key=lambda p: p.center[0])
        low = [p for p in left_right if p.center[1] < body.center[1]]
        high = [p for p in left_right if p.center[1] >= body.center[1]]
        if low:
            low[0].name = "left_leg"
            if len(low) > 1:
                low[-1].name = "right_leg"
        if high:
            high[0].name = "left_arm"
            if len(high) > 1:
                high[-1].name = "right_arm"
        return

    axis_index, sign = _axis_index(forward_axis, parts)
    # Head candidate: sizeable part farthest from body along either end. We do
    # not know forward sign from PartField, so prefer the larger of both extremes.
    extreme = sorted(
        remaining,
        key=lambda p: abs((p.center[axis_index] - body.center[axis_index]) * sign),
        reverse=True,
    )
    head = max(extreme[: max(2, len(extreme) // 3)], key=lambda p: (p.volume, p.surface_area))
    head.name = "head"
    head_direction = np.sign(head.center[axis_index] - body.center[axis_index]) or 1.0

    rest = [p for p in remaining if p is not head]
    below = [p for p in rest if p.center[1] < body.center[1] - 0.05 * max(body.size)]
    below = sorted(
        below,
        key=lambda p: (
            -head_direction * (p.center[axis_index] - body.center[axis_index]),
            p.center[0 if axis_index == 2 else 2],
        ),
    )
    leg_names = ["back_left_leg", "back_right_leg", "front_left_leg", "front_right_leg"]
    for part, name in zip(below[:4], leg_names):
        part.name = name

    unassigned = [p for p in rest if p.name.startswith("part_")]
    above_head = [p for p in unassigned if p.center[1] > head.center[1]]
    above_head = sorted(above_head, key=lambda p: p.center[0 if axis_index == 2 else 2])
    if above_head:
        above_head[0].name = "left_ear"
        if len(above_head) > 1:
            above_head[-1].name = "right_ear"

    unassigned = [p for p in rest if p.name.startswith("part_")]
    if unassigned:
        opposite = [
            p
            for p in unassigned
            if (p.center[axis_index] - body.center[axis_index]) * head_direction < 0
        ]
        if opposite:
            tail = max(
                opposite,
                key=lambda p: abs(p.center[axis_index] - body.center[axis_index]) / max(p.volume, 1e-8),
            )
            tail.name = "tail"
