from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, QhullError, cKDTree
from scipy.optimize import linear_sum_assignment

from .cuboid_fit import assign_part_names


@dataclass
class PrimitiveFitConfig:
    """Configuration for automatic paper-safe primitive fitting.

    The source PartField mesh is already the simplified mesh used for
    segmentation.  ``target_faces=0`` derives a per-cluster paper face budget
    from that simplified cluster's triangle count.
    """

    min_area_ratio: float = 0.0
    min_faces: int = 4
    target_faces: int = 0
    max_faces: int = 48
    max_sides: int = 24
    fit_samples: int = 2500
    complexity_weight: float = 0.025
    allowed_types: tuple[str, ...] = (
        "box",
        "prism",
        "frustum",
        "cone",
        "ellipsoid",
        "convex",
    )
    resolve_overlaps: bool = True
    overlap_gap_ratio: float = 0.001
    preserve_contacts: bool = True
    contact_overlap_ratio: float = 0.0
    contact_mode: str = "fixed"
    connector_sides: int = 4
    connector_radius_ratio: float = 0.028
    connector_inset_ratio: float = 0.28
    connector_min_length_ratio: float = 0.002
    interface_max_sides: int = 8
    interface_min_width_ratio: float = 0.006
    interface_plane_tolerance_ratio: float = 1e-6
    category: str = "generic"
    forward_axis: str = "auto"
    seed: int = 12345


@dataclass
class PrimitivePart:
    """One independent closed polyhedral shell fitted to a PartField segment.

    ``polygons`` stores the canonical paper faces (triangles, quads, or convex
    n-gons).  Geometry vertices are shared across adjacent faces so the OBJ
    exporter can preserve watertight topology for Blender's Paper Model add-on.
    """

    name: str
    segment_id: int
    vertices: np.ndarray
    polygons: list[list[int]]
    source_face_count: int
    source_surface_area: float
    source_center: np.ndarray
    primitive_type: str
    target_face_count: int
    fit_score: float
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def center(self) -> np.ndarray:
        bounds = self.bounds
        return (bounds[0] + bounds[1]) * 0.5

    @property
    def bounds(self) -> np.ndarray:
        vertices = np.asarray(self.vertices, dtype=np.float64)
        return np.vstack((vertices.min(axis=0), vertices.max(axis=0)))

    @property
    def size(self) -> np.ndarray:
        bounds = self.bounds
        return np.maximum(bounds[1] - bounds[0], 1e-12)

    @property
    def face_count(self) -> int:
        return len(self.polygons)

    @property
    def triangle_count(self) -> int:
        return sum(max(0, len(face) - 2) for face in self.polygons)

    @property
    def surface_area(self) -> float:
        return float(self.to_trimesh().area)

    @property
    def volume(self) -> float:
        return abs(float(self.to_trimesh().volume))

    def to_trimesh(self) -> trimesh.Trimesh:
        return trimesh.Trimesh(
            vertices=np.asarray(self.vertices, dtype=np.float64),
            faces=triangulate_polygons(self.polygons),
            process=False,
        )

    def scale_about(self, anchor: np.ndarray, factor: float) -> None:
        anchor = np.asarray(anchor, dtype=np.float64)
        self.vertices = anchor[None, :] + (
            np.asarray(self.vertices, dtype=np.float64) - anchor[None, :]
        ) * float(factor)

    def translate(self, delta: np.ndarray) -> None:
        delta = np.asarray(delta, dtype=np.float64)
        self.vertices = np.asarray(self.vertices, dtype=np.float64) + delta[None, :]
        self.metadata["proxy_translation"] = (
            np.asarray(self.metadata.get("proxy_translation", [0.0, 0.0, 0.0]), dtype=np.float64)
            + delta
        ).tolist()

    def rotate_about(self, anchor: np.ndarray, rotation: np.ndarray) -> None:
        anchor = np.asarray(anchor, dtype=np.float64)
        rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        self.vertices = (
            np.asarray(self.vertices, dtype=np.float64) - anchor[None, :]
        ) @ rotation.T + anchor[None, :]
        self.metadata.setdefault("contact_rotations", []).append(
            {
                "anchor": anchor.tolist(),
                "matrix": rotation.tolist(),
            }
        )

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "segment_id": int(self.segment_id),
            "primitive_type": self.primitive_type,
            "source_face_count": int(self.source_face_count),
            "target_face_count": int(self.target_face_count),
            "paper_face_count": int(self.face_count),
            "triangle_count": int(self.triangle_count),
            "source_surface_area": float(self.source_surface_area),
            "fitted_surface_area": float(self.surface_area),
            "volume": float(self.volume),
            "center": self.center.tolist(),
            "size": self.size.tolist(),
            "bounds": self.bounds.tolist(),
            "source_center": np.asarray(self.source_center, dtype=float).tolist(),
            "fit_score": float(self.fit_score),
            "vertices": np.asarray(self.vertices, dtype=float).tolist(),
            "polygons": [[int(index) for index in face] for face in self.polygons],
            "metadata": self.metadata,
        }


@dataclass
class _Candidate:
    primitive_type: str
    vertices: np.ndarray
    polygons: list[list[int]]
    type_penalty: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)
    score: float = float("inf")
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def face_count(self) -> int:
        return len(self.polygons)

    def mesh(self) -> trimesh.Trimesh:
        return trimesh.Trimesh(
            vertices=np.asarray(self.vertices, dtype=np.float64),
            faces=triangulate_polygons(self.polygons),
            process=False,
        )


@dataclass(frozen=True)
class _SourceContact:
    """One PartField label boundary measured on the original segmented mesh."""

    segment_a: int
    segment_b: int
    anchor: np.ndarray
    boundary_length: float
    edge_count: int
    boundary_points: np.ndarray
    direction_a_to_b: np.ndarray
    interface_normal: np.ndarray
    interface_axis_u: np.ndarray
    interface_axis_v: np.ndarray


@dataclass(frozen=True)
class _FrozenInterface:
    """A low-face interface reconstructed once from the source label seam.

    The same 3D polygon is inserted into both adjacent fitted parts.  All other
    vertices are restricted to their original side of the interface plane, so
    primitive fitting changes the outer silhouette but never relocates, rotates,
    shrinks, or replaces the source joint.
    """

    segment_a: int
    segment_b: int
    anchor: np.ndarray
    normal_a_to_b: np.ndarray
    axis_u: np.ndarray
    axis_v: np.ndarray
    polygon_2d: np.ndarray
    polygon_3d: np.ndarray
    area: float
    source_boundary_length: float
    source_boundary_edge_count: int
    fallback_rectangle: bool


def triangulate_polygons(polygons: Sequence[Sequence[int]]) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    for polygon in polygons:
        ids = [int(value) for value in polygon]
        if len(ids) < 3:
            continue
        for offset in range(1, len(ids) - 1):
            triangles.append((ids[0], ids[offset], ids[offset + 1]))
    if not triangles:
        raise ValueError("A primitive shell must contain at least one polygon")
    return np.asarray(triangles, dtype=np.int64)


def _polygon_normal(vertices: np.ndarray, polygon: Sequence[int]) -> np.ndarray:
    points = np.asarray(vertices[np.asarray(polygon, dtype=np.int64)], dtype=np.float64)
    normal = np.zeros(3, dtype=np.float64)
    for index in range(len(points)):
        current = points[index]
        following = points[(index + 1) % len(points)]
        normal += np.cross(current, following)
    length = float(np.linalg.norm(normal))
    if length <= 1e-12:
        return normal
    return normal / length


def _orient_outward(vertices: np.ndarray, polygons: Sequence[Sequence[int]]) -> list[list[int]]:
    vertices = np.asarray(vertices, dtype=np.float64)
    centroid = vertices.mean(axis=0)
    oriented: list[list[int]] = []
    for polygon in polygons:
        ids = [int(value) for value in polygon]
        normal = _polygon_normal(vertices, ids)
        face_center = vertices[np.asarray(ids, dtype=np.int64)].mean(axis=0)
        if float(np.dot(normal, face_center - centroid)) < 0.0:
            ids.reverse()
        oriented.append(ids)
    return oriented


def _validate_closed_shell(vertices: np.ndarray, polygons: Sequence[Sequence[int]]) -> None:
    if len(vertices) < 4 or len(polygons) < 4:
        raise ValueError("Primitive candidate is too small to form a closed shell")
    edge_counts: dict[tuple[int, int], int] = {}
    for polygon in polygons:
        if len(polygon) < 3:
            raise ValueError("Primitive polygon has fewer than three vertices")
        for a, b in zip(polygon, list(polygon[1:]) + [polygon[0]]):
            edge = tuple(sorted((int(a), int(b))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    invalid = [edge for edge, count in edge_counts.items() if count != 2]
    if invalid:
        raise ValueError(f"Primitive shell is not watertight; invalid edges={invalid[:5]}")


def _canonical_candidate(
    primitive_type: str,
    vertices: np.ndarray,
    polygons: Sequence[Sequence[int]],
    *,
    type_penalty: float = 0.0,
    metadata: dict[str, object] | None = None,
) -> _Candidate:
    vertices = np.asarray(vertices, dtype=np.float64)
    oriented = _orient_outward(vertices, polygons)
    _validate_closed_shell(vertices, oriented)
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangulate_polygons(oriented), process=False)
    if not np.isfinite(mesh.vertices).all() or abs(float(mesh.volume)) <= 1e-12:
        raise ValueError(f"Degenerate primitive candidate: {primitive_type}")
    return _Candidate(
        primitive_type=primitive_type,
        vertices=vertices,
        polygons=oriented,
        type_penalty=float(type_penalty),
        metadata=dict(metadata or {}),
    )


def _sample_triangles(
    triangles: np.ndarray,
    areas: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    triangles = np.asarray(triangles, dtype=np.float64)
    areas = np.asarray(areas, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(areas) & (areas > 1e-15))
    if len(valid) == 0:
        raise ValueError("Cluster contains no non-degenerate triangle")
    probabilities = areas[valid] / areas[valid].sum()
    selected = rng.choice(valid, size=max(64, int(count)), replace=True, p=probabilities)
    chosen = triangles[selected]
    r1 = np.sqrt(rng.random(len(selected)))
    r2 = rng.random(len(selected))
    bary = np.column_stack((1.0 - r1, r1 * (1.0 - r2), r1 * r2))
    return np.einsum("ni,nic->nc", bary, chosen)


def _sample_mesh(mesh: trimesh.Trimesh, count: int, rng: np.random.Generator) -> np.ndarray:
    return _sample_triangles(mesh.triangles, mesh.area_faces, count, rng)


def _pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    center = np.mean(points, axis=0)
    centered = points - center[None, :]
    covariance = centered.T @ centered / max(len(points), 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    vectors = vectors[:, order]
    if np.linalg.det(vectors) < 0:
        vectors[:, -1] *= -1.0
    return center, vectors, np.maximum(values[order], 0.0)


def _box_candidate(points: np.ndarray) -> _Candidate:
    try:
        to_origin, extents = trimesh.bounds.oriented_bounds(points, angle_digits=1, ordered=True)
        local_to_world = np.linalg.inv(to_origin)
        extents = np.maximum(np.asarray(extents, dtype=np.float64), 1e-6)
        x, y, z = extents * 0.5
        local = np.asarray(
            [
                (-x, -y, -z),
                (x, -y, -z),
                (x, y, -z),
                (-x, y, -z),
                (-x, -y, z),
                (x, -y, z),
                (x, y, z),
                (-x, y, z),
            ],
            dtype=np.float64,
        )
        vertices = trimesh.transform_points(local, local_to_world)
    except Exception:
        center, rotation, _ = _pca_frame(points)
        local_points = (points - center[None, :]) @ rotation
        mins = np.quantile(local_points, 0.005, axis=0)
        maxs = np.quantile(local_points, 0.995, axis=0)
        local_center = (mins + maxs) * 0.5
        half = np.maximum((maxs - mins) * 0.5, 1e-6)
        x, y, z = half
        local = np.asarray(
            [
                (-x, -y, -z),
                (x, -y, -z),
                (x, y, -z),
                (-x, y, -z),
                (-x, -y, z),
                (x, -y, z),
                (x, y, z),
                (-x, y, z),
            ],
            dtype=np.float64,
        ) + local_center[None, :]
        vertices = local @ rotation.T + center[None, :]
    polygons = [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]
    return _canonical_candidate("box", vertices, polygons)


def _axis_basis(rotation: np.ndarray, axis: int) -> np.ndarray:
    remaining = [index for index in range(3) if index != axis]
    basis = np.column_stack((rotation[:, axis], rotation[:, remaining[0]], rotation[:, remaining[1]]))
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1.0
    return basis


def _radial_summary(local_points: np.ndarray) -> dict[str, np.ndarray | float]:
    t = local_points[:, 0]
    t0 = float(np.quantile(t, 0.005))
    t1 = float(np.quantile(t, 0.995))
    span = max(t1 - t0, 1e-6)
    low_mask = t <= t0 + 0.33 * span
    high_mask = t >= t1 - 0.33 * span
    if np.count_nonzero(low_mask) < 16:
        low_mask = t <= np.median(t)
    if np.count_nonzero(high_mask) < 16:
        high_mask = t >= np.median(t)

    def section(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        section_points = local_points[mask, 1:3]
        center = np.median(section_points, axis=0)
        radii = np.quantile(np.abs(section_points - center[None, :]), 0.98, axis=0)
        return center, np.maximum(radii, span * 1e-4)

    center_low, radii_low = section(low_mask)
    center_high, radii_high = section(high_mask)
    center_all = np.median(local_points[:, 1:3], axis=0)
    radii_all = np.quantile(np.abs(local_points[:, 1:3] - center_all[None, :]), 0.99, axis=0)
    radii_all = np.maximum(radii_all, span * 1e-4)
    return {
        "t0": t0,
        "t1": t1,
        "span": span,
        "center_low": center_low,
        "center_high": center_high,
        "radii_low": radii_low,
        "radii_high": radii_high,
        "center_all": center_all,
        "radii_all": radii_all,
    }


def _ring(center_t: float, center_yz: np.ndarray, radii: np.ndarray, sides: int, phase: float) -> np.ndarray:
    angles = phase + np.arange(sides, dtype=np.float64) * (2.0 * np.pi / sides)
    return np.column_stack(
        (
            np.full(sides, float(center_t), dtype=np.float64),
            float(center_yz[0]) + float(radii[0]) * np.cos(angles),
            float(center_yz[1]) + float(radii[1]) * np.sin(angles),
        )
    )


def _prism_candidate(
    points: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    axis: int,
    sides: int,
    phase: float,
    *,
    frustum: bool,
) -> _Candidate:
    basis = _axis_basis(rotation, axis)
    local_points = (points - center[None, :]) @ basis
    summary = _radial_summary(local_points)
    t0 = float(summary["t0"])
    t1 = float(summary["t1"])
    if frustum:
        center_low = np.asarray(summary["center_low"], dtype=np.float64)
        center_high = np.asarray(summary["center_high"], dtype=np.float64)
        radii_low = np.asarray(summary["radii_low"], dtype=np.float64)
        radii_high = np.asarray(summary["radii_high"], dtype=np.float64)
        taper_strength = float(
            np.mean(
                np.abs(
                    np.log(
                        np.maximum(radii_low, 1e-9)
                        / np.maximum(radii_high, 1e-9)
                    )
                )
            )
        )
    else:
        center_low = np.asarray(summary["center_low"], dtype=np.float64)
        center_high = np.asarray(summary["center_high"], dtype=np.float64)
        radii = np.maximum(
            np.asarray(summary["radii_all"], dtype=np.float64),
            np.maximum(
                np.asarray(summary["radii_low"], dtype=np.float64),
                np.asarray(summary["radii_high"], dtype=np.float64),
            ),
        )
        radii_low = radii_high = radii
    low = _ring(t0, center_low, radii_low, sides, phase)
    high = _ring(t1, center_high, radii_high, sides, phase)
    local_vertices = np.vstack((low, high))
    vertices = local_vertices @ basis.T + center[None, :]
    polygons: list[list[int]] = [list(reversed(range(sides))), list(range(sides, 2 * sides))]
    for index in range(sides):
        following = (index + 1) % sides
        polygons.append([index, following, sides + following, sides + index])
    label = "frustum" if frustum else "prism"
    return _canonical_candidate(
        f"{label}_{sides}",
        vertices,
        polygons,
        type_penalty=(0.004 if frustum and taper_strength < 0.08 else 0.001) if frustum else 0.0,
        metadata={
            "axis": int(axis),
            "sides": int(sides),
            "phase": float(phase),
            **({"taper_strength": taper_strength} if frustum else {}),
        },
    )


def _cone_candidate(
    points: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    axis: int,
    sides: int,
    phase: float,
) -> _Candidate:
    basis = _axis_basis(rotation, axis)
    local_points = (points - center[None, :]) @ basis
    summary = _radial_summary(local_points)
    low_radius = float(np.prod(np.asarray(summary["radii_low"], dtype=np.float64)))
    high_radius = float(np.prod(np.asarray(summary["radii_high"], dtype=np.float64)))
    if low_radius <= high_radius:
        apex_t = float(summary["t0"])
        apex_center = np.asarray(summary["center_low"], dtype=np.float64)
        base_t = float(summary["t1"])
        base_center = np.asarray(summary["center_high"], dtype=np.float64)
        base_radii = np.maximum(
            np.asarray(summary["radii_high"], dtype=np.float64),
            np.asarray(summary["radii_all"], dtype=np.float64) * 0.75,
        )
    else:
        apex_t = float(summary["t1"])
        apex_center = np.asarray(summary["center_high"], dtype=np.float64)
        base_t = float(summary["t0"])
        base_center = np.asarray(summary["center_low"], dtype=np.float64)
        base_radii = np.maximum(
            np.asarray(summary["radii_low"], dtype=np.float64),
            np.asarray(summary["radii_all"], dtype=np.float64) * 0.75,
        )
    radius_ratio = float(
        np.sqrt(max(min(low_radius, high_radius), 1e-12) / max(max(low_radius, high_radius), 1e-12))
    )
    ring = _ring(base_t, base_center, base_radii, sides, phase)
    apex = np.asarray([[apex_t, apex_center[0], apex_center[1]]], dtype=np.float64)
    local_vertices = np.vstack((ring, apex))
    vertices = local_vertices @ basis.T + center[None, :]
    apex_index = sides
    polygons: list[list[int]] = [list(range(sides))]
    for index in range(sides):
        following = (index + 1) % sides
        polygons.append([index, following, apex_index])
    return _canonical_candidate(
        f"cone_{sides}",
        vertices,
        polygons,
        type_penalty=0.010 if radius_ratio > 0.70 else 0.0,
        metadata={
            "axis": int(axis),
            "sides": int(sides),
            "phase": float(phase),
            "end_radius_ratio": radius_ratio,
        },
    )


def _ellipsoid_candidate(
    points: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    axis: int,
    segments: int,
    rings: int,
    phase: float,
) -> _Candidate:
    basis = _axis_basis(rotation, axis)
    local_points = (points - center[None, :]) @ basis
    mins = np.quantile(local_points, 0.005, axis=0)
    maxs = np.quantile(local_points, 0.995, axis=0)
    local_center = (mins + maxs) * 0.5
    radii = np.maximum((maxs - mins) * 0.5, 1e-6)

    local_vertices: list[list[float]] = [[local_center[0] + radii[0], local_center[1], local_center[2]]]
    for ring_index in range(1, rings + 1):
        theta = np.pi * ring_index / (rings + 1)
        axial = local_center[0] + radii[0] * np.cos(theta)
        radial_scale = np.sin(theta)
        for segment in range(segments):
            phi = phase + 2.0 * np.pi * segment / segments
            local_vertices.append(
                [
                    axial,
                    local_center[1] + radii[1] * radial_scale * np.cos(phi),
                    local_center[2] + radii[2] * radial_scale * np.sin(phi),
                ]
            )
    bottom_index = len(local_vertices)
    local_vertices.append([local_center[0] - radii[0], local_center[1], local_center[2]])
    local_array = np.asarray(local_vertices, dtype=np.float64)
    vertices = local_array @ basis.T + center[None, :]

    polygons: list[list[int]] = []
    first_ring = 1
    for segment in range(segments):
        following = (segment + 1) % segments
        polygons.append([0, first_ring + segment, first_ring + following])
    for ring_index in range(rings - 1):
        current = 1 + ring_index * segments
        following_ring = current + segments
        for segment in range(segments):
            following = (segment + 1) % segments
            polygons.append(
                [
                    current + segment,
                    following_ring + segment,
                    following_ring + following,
                    current + following,
                ]
            )
    last_ring = 1 + (rings - 1) * segments
    for segment in range(segments):
        following = (segment + 1) % segments
        polygons.append([last_ring + segment, bottom_index, last_ring + following])
    return _canonical_candidate(
        f"ellipsoid_s{segments}_r{rings}",
        vertices,
        polygons,
        type_penalty=0.003,
        metadata={
            "axis": int(axis),
            "segments": int(segments),
            "rings": int(rings),
            "phase": float(phase),
        },
    )


def _fibonacci_directions(count: int) -> np.ndarray:
    count = max(6, int(count))
    indices = np.arange(count, dtype=np.float64)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (indices + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    angles = golden * indices
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles), z))


def _convex_candidate(points: np.ndarray, target_faces: int, direction_count: int) -> _Candidate:
    center, rotation, _ = _pca_frame(points)
    local_points = (points - center[None, :]) @ rotation
    directions = np.vstack(
        (
            _fibonacci_directions(direction_count),
            np.eye(3, dtype=np.float64),
            -np.eye(3, dtype=np.float64),
        )
    )
    chosen: list[int] = []
    for direction in directions:
        chosen.append(int(np.argmax(local_points @ direction)))
    unique = np.unique(np.asarray(chosen, dtype=np.int64))
    support = local_points[unique]
    if len(support) < 4:
        raise ValueError("Not enough support points for a convex candidate")
    hull = ConvexHull(support)
    hull_vertices = support[hull.vertices]
    remap = {int(old): index for index, old in enumerate(hull.vertices.tolist())}
    polygons: list[list[int]] = []
    for simplex in hull.simplices:
        if all(int(index) in remap for index in simplex):
            polygons.append([remap[int(index)] for index in simplex])
    vertices = hull_vertices @ rotation.T + center[None, :]
    return _canonical_candidate(
        f"convex_{len(polygons)}",
        vertices,
        polygons,
        type_penalty=0.012,
        metadata={
            "direction_count": int(direction_count),
            "requested_target_faces": int(target_faces),
        },
    )


def _auto_target_faces(source_face_count: int, max_faces: int) -> int:
    # A square-root schedule keeps detailed clusters more expressive without
    # making paper assembly scale linearly with the source triangle count.
    target = int(round(2.0 * np.sqrt(max(int(source_face_count), 1))))
    return int(np.clip(target, 6, max(int(max_faces), 6)))


def _candidate_side_counts(target_faces: int, max_faces: int, max_sides: int) -> list[int]:
    preferred = {3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32}
    nearest = max(3, min(max_sides, target_faces - 2))
    preferred.update({nearest, max(3, nearest - 2), min(max_sides, nearest + 2)})
    available = sorted(
        sides
        for sides in preferred
        if 3 <= sides <= max_sides and sides + 2 <= max_faces
    )
    # Evaluate only the most relevant ring resolutions.  Candidate fitting is
    # bidirectional and sample based, so an exhaustive sweep adds substantial
    # cost without materially changing the selected paper complexity.
    ranked = sorted(available, key=lambda sides: (abs((sides + 2) - target_faces), sides))
    selected = set(ranked[:5])
    selected.update(sides for sides in (3, 4, 6, 8) if sides in available)
    return sorted(selected)


def _score_candidate(
    candidate: _Candidate,
    source_points: np.ndarray,
    source_tree: cKDTree,
    source_hull_volume: float,
    target_faces: int,
    max_faces: int,
    fit_samples: int,
    complexity_weight: float,
    rng: np.random.Generator,
) -> None:
    mesh = candidate.mesh()
    candidate_points = _sample_mesh(mesh, fit_samples, rng)
    candidate_tree = cKDTree(candidate_points)
    distance_source_to_candidate = candidate_tree.query(source_points, k=1, workers=-1)[0]
    distance_candidate_to_source = source_tree.query(candidate_points, k=1, workers=-1)[0]
    diagonal = max(float(np.linalg.norm(np.ptp(source_points, axis=0))), 1e-8)

    source_mean = float(np.mean(distance_source_to_candidate) / diagonal)
    candidate_mean = float(np.mean(distance_candidate_to_source) / diagonal)
    source_p95 = float(np.quantile(distance_source_to_candidate, 0.95) / diagonal)
    candidate_p95 = float(np.quantile(distance_candidate_to_source, 0.95) / diagonal)
    bidirectional_mean = 0.72 * source_mean + 0.28 * candidate_mean
    bidirectional_p95 = 0.72 * source_p95 + 0.28 * candidate_p95

    face_delta = abs(candidate.face_count - target_faces) / max(target_faces, 1)
    paper_complexity = candidate.face_count / max(max_faces, 1)
    candidate_volume = max(abs(float(mesh.volume)), 1e-12)
    if source_hull_volume > 1e-12:
        volume_log_error = abs(float(np.log(candidate_volume / source_hull_volume)))
    else:
        volume_log_error = 0.0

    score = (
        bidirectional_mean
        + 0.20 * bidirectional_p95
        + float(complexity_weight) * (0.70 * face_delta + 0.30 * paper_complexity)
        + 0.010 * volume_log_error
        + float(candidate.type_penalty)
    )
    candidate.score = float(score)
    candidate.metrics = {
        "source_to_candidate_mean": source_mean,
        "candidate_to_source_mean": candidate_mean,
        "source_to_candidate_p95": source_p95,
        "candidate_to_source_p95": candidate_p95,
        "bidirectional_mean": bidirectional_mean,
        "bidirectional_p95": bidirectional_p95,
        "face_delta_ratio": float(face_delta),
        "paper_complexity_ratio": float(paper_complexity),
        "volume_log_error": float(volume_log_error),
        "score": float(score),
    }


def _build_candidates(
    points: np.ndarray,
    target_faces: int,
    config: PrimitiveFitConfig,
) -> list[_Candidate]:
    allowed = set(config.allowed_types)
    candidates: list[_Candidate] = []
    center, rotation, eigenvalues = _pca_frame(points)
    side_counts = _candidate_side_counts(target_faces, config.max_faces, config.max_sides)

    def append(factory) -> None:
        try:
            candidate = factory()
            if candidate.face_count <= config.max_faces:
                candidates.append(candidate)
        except (ValueError, QhullError, np.linalg.LinAlgError):
            return

    if "box" in allowed:
        append(lambda: _box_candidate(points))

    phases_by_sides = lambda sides: (0.0, np.pi / max(sides, 1))
    for axis in range(3):
        for sides in side_counts:
            for phase in phases_by_sides(sides):
                if "prism" in allowed:
                    append(
                        lambda axis=axis, sides=sides, phase=phase: _prism_candidate(
                            points, center, rotation, axis, sides, phase, frustum=False
                        )
                    )
                if "frustum" in allowed:
                    append(
                        lambda axis=axis, sides=sides, phase=phase: _prism_candidate(
                            points, center, rotation, axis, sides, phase, frustum=True
                        )
                    )
                if "cone" in allowed and sides + 1 <= config.max_faces:
                    append(
                        lambda axis=axis, sides=sides, phase=phase: _cone_candidate(
                            points, center, rotation, axis, sides, phase
                        )
                    )

    if "ellipsoid" in allowed:
        ellipsoid_specs: set[tuple[int, int]] = set()
        for rings in (1, 2, 3, 4):
            approximate_segments = int(round(target_faces / max(rings + 1, 1)))
            for segments in (
                max(4, approximate_segments - 2),
                max(4, approximate_segments),
                max(4, approximate_segments + 2),
                6,
                8,
                12,
            ):
                if segments * (rings + 1) <= config.max_faces and segments <= config.max_sides:
                    ellipsoid_specs.add((segments, rings))
        ellipsoid_ranked = sorted(
            ellipsoid_specs,
            key=lambda item: (abs(item[0] * (item[1] + 1) - target_faces), item[0], item[1]),
        )[:8]
        for axis in range(3):
            for segments, rings in ellipsoid_ranked:
                for phase in phases_by_sides(segments):
                    append(
                        lambda axis=axis, segments=segments, rings=rings, phase=phase: _ellipsoid_candidate(
                            points, center, rotation, axis, segments, rings, phase
                        )
                    )

    if "convex" in allowed:
        estimated_vertices = max(6, int(round((target_faces + 4) * 0.5)))
        for direction_count in sorted(
            {
                max(6, estimated_vertices - 4),
                max(6, estimated_vertices),
                max(6, estimated_vertices + 4),
                min(32, max(8, target_faces)),
            }
        ):
            append(
                lambda direction_count=direction_count: _convex_candidate(
                    points, target_faces, direction_count
                )
            )

    if not candidates:
        raise ValueError("No valid primitive candidates could be generated")
    return candidates


def _hull_volume(points: np.ndarray) -> float:
    try:
        return float(ConvexHull(points).volume)
    except (QhullError, ValueError):
        return 0.0


def _signed_polygon_area_2d(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return 0.0
    return 0.5 * float(
        np.sum(
            points[:, 0] * np.roll(points[:, 1], -1)
            - points[:, 1] * np.roll(points[:, 0], -1)
        )
    )


def _resample_convex_polygon(points: np.ndarray, max_vertices: int) -> np.ndarray:
    """Reduce a convex polygon without changing its plane or overall footprint."""

    points = np.asarray(points, dtype=np.float64)
    if len(points) <= int(max_vertices):
        return points
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    perimeter = float(np.sum(lengths))
    if perimeter <= 1e-12:
        return points[: max(3, int(max_vertices))]
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    samples: list[np.ndarray] = []
    for distance in np.linspace(0.0, perimeter, int(max_vertices), endpoint=False):
        edge_index = min(int(np.searchsorted(cumulative, distance, side="right") - 1), len(points) - 1)
        edge_length = max(float(lengths[edge_index]), 1e-12)
        local = (float(distance) - float(cumulative[edge_index])) / edge_length
        samples.append(points[edge_index] * (1.0 - local) + points[(edge_index + 1) % len(points)] * local)
    sampled = np.asarray(samples, dtype=np.float64)
    try:
        hull = ConvexHull(sampled)
        sampled = sampled[hull.vertices]
    except QhullError:
        pass
    if _signed_polygon_area_2d(sampled) < 0.0:
        sampled = sampled[::-1]
    return sampled


def _build_frozen_interfaces(
    contacts: dict[tuple[int, int], _SourceContact],
    *,
    model_extent: float,
    max_sides: int,
    min_width_ratio: float,
) -> dict[tuple[int, int], _FrozenInterface]:
    """Convert every source seam into one immutable low-face joint polygon."""

    frozen: dict[tuple[int, int], _FrozenInterface] = {}
    minimum_half_width = max(float(model_extent) * float(min_width_ratio), 1e-7)
    for pair, contact in contacts.items():
        points = np.asarray(contact.boundary_points, dtype=np.float64)
        anchor = np.asarray(contact.anchor, dtype=np.float64)
        axis_u = np.asarray(contact.interface_axis_u, dtype=np.float64)
        axis_v = np.asarray(contact.interface_axis_v, dtype=np.float64)
        if len(points):
            projected = np.column_stack(
                ((points - anchor[None, :]) @ axis_u, (points - anchor[None, :]) @ axis_v)
            )
            rounded = np.round(projected, decimals=12)
            _, unique_indices = np.unique(rounded, axis=0, return_index=True)
            projected = projected[np.sort(unique_indices)]
        else:
            projected = np.empty((0, 2), dtype=np.float64)

        fallback = False
        polygon_2d: np.ndarray
        if len(projected) >= 3:
            try:
                hull_2d = ConvexHull(projected)
                polygon_2d = projected[hull_2d.vertices]
            except QhullError:
                polygon_2d = np.empty((0, 2), dtype=np.float64)
        else:
            polygon_2d = np.empty((0, 2), dtype=np.float64)

        area = abs(_signed_polygon_area_2d(polygon_2d))
        minimum_area = 4.0 * minimum_half_width * minimum_half_width
        if len(polygon_2d) < 3 or area < minimum_area * 0.15:
            fallback = True
            if len(projected):
                center_2d = np.mean(projected, axis=0)
                mins = projected.min(axis=0)
                maxs = projected.max(axis=0)
                half = np.maximum((maxs - mins) * 0.5, minimum_half_width)
            else:
                center_2d = np.zeros(2, dtype=np.float64)
                half = np.full(2, minimum_half_width, dtype=np.float64)
            polygon_2d = np.asarray(
                [
                    center_2d + [-half[0], -half[1]],
                    center_2d + [half[0], -half[1]],
                    center_2d + [half[0], half[1]],
                    center_2d + [-half[0], half[1]],
                ],
                dtype=np.float64,
            )
        else:
            polygon_2d = _resample_convex_polygon(polygon_2d, max(3, int(max_sides)))

        if _signed_polygon_area_2d(polygon_2d) < 0.0:
            polygon_2d = polygon_2d[::-1]
        polygon_3d = (
            anchor[None, :]
            + polygon_2d[:, 0:1] * axis_u[None, :]
            + polygon_2d[:, 1:2] * axis_v[None, :]
        )
        frozen[pair] = _FrozenInterface(
            segment_a=int(pair[0]),
            segment_b=int(pair[1]),
            anchor=anchor,
            normal_a_to_b=np.asarray(contact.interface_normal, dtype=np.float64),
            axis_u=axis_u,
            axis_v=axis_v,
            polygon_2d=polygon_2d,
            polygon_3d=polygon_3d,
            area=abs(_signed_polygon_area_2d(polygon_2d)),
            source_boundary_length=float(contact.boundary_length),
            source_boundary_edge_count=int(contact.edge_count),
            fallback_rectangle=bool(fallback),
        )
    return frozen


def _interface_side_sign(
    interface: _FrozenInterface,
    segment_id: int,
    source_center: np.ndarray,
) -> float:
    signed = float(
        np.dot(
            np.asarray(source_center, dtype=np.float64) - np.asarray(interface.anchor),
            np.asarray(interface.normal_a_to_b),
        )
    )
    if abs(signed) > 1e-10:
        return 1.0 if signed > 0.0 else -1.0
    return -1.0 if int(segment_id) == int(interface.segment_a) else 1.0


def _polygon_area_3d(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return 0.0
    origin = points[0]
    return 0.5 * float(
        sum(
            np.linalg.norm(np.cross(points[index] - origin, points[index + 1] - origin))
            for index in range(1, len(points) - 1)
        )
    )


def _merge_hull_facets(points: np.ndarray, hull: ConvexHull) -> tuple[np.ndarray, list[list[int]]]:
    """Merge coplanar Qhull triangles into canonical convex paper faces."""

    points = np.asarray(points, dtype=np.float64)
    hull_ids = np.asarray(hull.vertices, dtype=np.int64)
    vertices = points[hull_ids]
    remap = {int(old): int(new) for new, old in enumerate(hull_ids.tolist())}
    extent = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 1e-8)
    normal_tolerance = 2e-7
    offset_tolerance = extent * 2e-7
    groups: list[dict[str, object]] = []

    for simplex, equation in zip(hull.simplices, hull.equations):
        if not all(int(index) in remap for index in simplex):
            continue
        normal = np.asarray(equation[:3], dtype=np.float64)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        offset = float(equation[3])
        group = None
        for existing in groups:
            existing_normal = np.asarray(existing["normal"], dtype=np.float64)
            if (
                float(np.dot(existing_normal, normal)) >= 1.0 - normal_tolerance
                and abs(float(existing["offset"]) - offset) <= offset_tolerance
            ):
                group = existing
                break
        if group is None:
            group = {"normal": normal, "offset": offset, "ids": set()}
            groups.append(group)
        ids = group["ids"]
        assert isinstance(ids, set)
        ids.update(int(index) for index in simplex)

    polygons: list[list[int]] = []
    for group in groups:
        original_ids = sorted(int(value) for value in group["ids"])
        local_ids = [remap[value] for value in original_ids]
        face_points = vertices[np.asarray(local_ids, dtype=np.int64)]
        normal = np.asarray(group["normal"], dtype=np.float64)
        axis_u, axis_v = _basis_from_normal(normal)
        origin = face_points.mean(axis=0)
        projected = np.column_stack(
            ((face_points - origin[None, :]) @ axis_u, (face_points - origin[None, :]) @ axis_v)
        )
        if len(local_ids) > 3:
            try:
                face_hull = ConvexHull(projected)
                ordered = [local_ids[int(index)] for index in face_hull.vertices]
            except QhullError:
                ordered = local_ids
        else:
            ordered = local_ids
        if len(ordered) < 3:
            continue
        polygon_normal = _polygon_normal(vertices, ordered)
        if float(np.dot(polygon_normal, normal)) < 0.0:
            ordered.reverse()
        polygons.append(ordered)

    used = sorted({int(index) for polygon in polygons for index in polygon})
    compact_remap = {old: new for new, old in enumerate(used)}
    compact_vertices = vertices[np.asarray(used, dtype=np.int64)]
    compact_polygons = [[compact_remap[int(index)] for index in polygon] for polygon in polygons]
    return compact_vertices, compact_polygons


def _find_interface_face_index(
    vertices: np.ndarray,
    polygons: Sequence[Sequence[int]],
    interface: _FrozenInterface,
    tolerance: float,
) -> int | None:
    vertices = np.asarray(vertices, dtype=np.float64)
    normal = np.asarray(interface.normal_a_to_b, dtype=np.float64)
    anchor = np.asarray(interface.anchor, dtype=np.float64)
    candidates: list[tuple[float, int]] = []
    for face_index, polygon in enumerate(polygons):
        points = vertices[np.asarray(polygon, dtype=np.int64)]
        distances = np.abs((points - anchor[None, :]) @ normal)
        if float(np.max(distances)) > float(tolerance):
            continue
        candidates.append((_polygon_area_3d(points), int(face_index)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], -item[1]))[1]


def _constrain_candidate_to_interfaces_convex(
    candidate: _Candidate,
    *,
    segment_id: int,
    source_center: np.ndarray,
    interfaces: Sequence[_FrozenInterface],
    model_extent: float,
    plane_tolerance_ratio: float,
) -> _Candidate:
    """Refit a candidate around immutable source interfaces.

    Candidate vertices are allowed to move only implicitly through primitive
    selection.  The final convex shell is rebuilt from candidate support points
    on the legal side of every source interface plus the exact shared interface
    polygons.  Therefore each adjacent pair receives identical contact geometry.
    """

    if not interfaces:
        return candidate
    tolerance = max(float(model_extent) * float(plane_tolerance_ratio), 1e-9)
    interior_margin = tolerance * 4.0
    base_vertices = np.asarray(candidate.vertices, dtype=np.float64)
    support: list[np.ndarray] = []
    for vertex in base_vertices:
        adjusted = np.asarray(vertex, dtype=np.float64).copy()
        valid = True
        for interface in interfaces:
            sign = _interface_side_sign(interface, segment_id, source_center)
            signed = sign * float(
                np.dot(adjusted - np.asarray(interface.anchor), interface.normal_a_to_b)
            )
            if signed < -tolerance:
                valid = False
                break
            if signed < interior_margin:
                adjusted = adjusted + sign * np.asarray(interface.normal_a_to_b) * (
                    interior_margin - signed
                )
        if valid:
            support.append(adjusted)

    center_support = np.asarray(source_center, dtype=np.float64).copy()
    for interface in interfaces:
        sign = _interface_side_sign(interface, segment_id, source_center)
        signed = sign * float(
            np.dot(center_support - np.asarray(interface.anchor), interface.normal_a_to_b)
        )
        if signed < interior_margin:
            center_support = center_support + sign * np.asarray(interface.normal_a_to_b) * (
                interior_margin - signed
            )
    support.append(center_support)
    for interface in interfaces:
        support.extend(np.asarray(interface.polygon_3d, dtype=np.float64))

    support_array = np.asarray(support, dtype=np.float64)
    rounded = np.round(support_array, decimals=12)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    support_array = support_array[np.sort(unique_indices)]
    if len(support_array) < 4:
        raise ValueError("Frozen-interface candidate has fewer than four support points")
    hull = ConvexHull(support_array)
    vertices, polygons = _merge_hull_facets(support_array, hull)
    constrained = _canonical_candidate(
        candidate.primitive_type,
        vertices,
        polygons,
        type_penalty=candidate.type_penalty,
        metadata={
            **candidate.metadata,
            "base_primitive_type": candidate.primitive_type,
            "frozen_interface_count": int(len(interfaces)),
            "frozen_interface_neighbors": sorted(
                int(interface.segment_b if int(segment_id) == interface.segment_a else interface.segment_a)
                for interface in interfaces
            ),
            "fitting_strategy": "fit_outer_surface_around_fixed_source_interfaces",
        },
    )
    face_indices: dict[str, int] = {}
    interface_areas: dict[str, float] = {}
    for interface in interfaces:
        neighbor = int(
            interface.segment_b if int(segment_id) == interface.segment_a else interface.segment_a
        )
        face_index = _find_interface_face_index(
            constrained.vertices,
            constrained.polygons,
            interface,
            tolerance * 8.0,
        )
        if face_index is None:
            raise ValueError(
                f"Frozen interface {segment_id}<->{neighbor} is not a face of constrained candidate"
            )
        face_indices[str(neighbor)] = int(face_index)
        points = constrained.vertices[np.asarray(constrained.polygons[face_index], dtype=np.int64)]
        interface_areas[str(neighbor)] = float(_polygon_area_3d(points))
    constrained.metadata["frozen_interface_face_indices"] = face_indices
    constrained.metadata["frozen_interface_areas"] = interface_areas
    constrained.metadata["untextured_contact_face_indices"] = sorted(face_indices.values())
    return constrained



def _orient_closed_manifold(
    vertices: np.ndarray,
    polygons: Sequence[Sequence[int]],
) -> list[list[int]]:
    """Orient a watertight polygon shell consistently, including non-convex shells.

    The old centroid test is valid only for convex bodies.  Fixed-interface
    adapters may create a locally concave shell, so face orientation is
    propagated through shared edges and the whole shell is flipped according to
    its signed volume.
    """

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = [[int(value) for value in polygon] for polygon in polygons]
    edge_uses: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for face_index, face in enumerate(faces):
        for a, b in zip(face, face[1:] + face[:1]):
            edge_uses.setdefault(tuple(sorted((a, b))), []).append((face_index, a, b))
    invalid = [edge for edge, uses in edge_uses.items() if len(uses) != 2]
    if invalid:
        raise ValueError(f"Adapter shell is not watertight; invalid edges={invalid[:5]}")

    adjacency: dict[int, list[tuple[int, bool]]] = {index: [] for index in range(len(faces))}
    for uses in edge_uses.values():
        (face_a, a0, a1), (face_b, b0, b1) = uses
        same_direction = a0 == b0 and a1 == b1
        adjacency[face_a].append((face_b, same_direction))
        adjacency[face_b].append((face_a, same_direction))

    flipped: dict[int, bool] = {}
    for seed in range(len(faces)):
        if seed in flipped:
            continue
        flipped[seed] = False
        queue = [seed]
        while queue:
            current = queue.pop()
            for neighbor, same_direction in adjacency[current]:
                required = bool(flipped[current]) ^ bool(same_direction)
                if neighbor in flipped:
                    if bool(flipped[neighbor]) != required:
                        raise ValueError("Adapter shell has non-orientable face adjacency")
                    continue
                flipped[neighbor] = required
                queue.append(neighbor)

    oriented = [face[::-1] if flipped[index] else face for index, face in enumerate(faces)]
    signed_volume = 0.0
    for triangle in triangulate_polygons(oriented):
        a, b, c = vertices[np.asarray(triangle, dtype=np.int64)]
        signed_volume += float(np.dot(a, np.cross(b, c))) / 6.0
    if signed_volume < 0.0:
        oriented = [face[::-1] for face in oriented]
    return oriented


def _canonical_interface_adapter_candidate(
    primitive_type: str,
    vertices: np.ndarray,
    polygons: Sequence[Sequence[int]],
    *,
    type_penalty: float,
    metadata: dict[str, object],
) -> _Candidate:
    vertices = np.asarray(vertices, dtype=np.float64)
    oriented = _orient_closed_manifold(vertices, polygons)
    _validate_closed_shell(vertices, oriented)
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=triangulate_polygons(oriented),
        process=False,
    )
    if not np.isfinite(mesh.vertices).all() or abs(float(mesh.volume)) <= 1e-12:
        raise ValueError(f"Degenerate fixed-interface adapter: {primitive_type}")
    return _Candidate(
        primitive_type=primitive_type,
        vertices=vertices,
        polygons=oriented,
        type_penalty=float(type_penalty),
        metadata=dict(metadata),
    )


def _triangulated_polygons(polygons: Sequence[Sequence[int]]) -> list[list[int]]:
    result: list[list[int]] = []
    for polygon in polygons:
        ids = [int(value) for value in polygon]
        for offset in range(1, len(ids) - 1):
            result.append([ids[0], ids[offset], ids[offset + 1]])
    return result


def _resample_closed_loop(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    perimeter = float(np.sum(lengths))
    if perimeter <= 1e-12:
        return np.repeat(points[:1], int(count), axis=0)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    samples: list[np.ndarray] = []
    for distance in np.linspace(0.0, perimeter, int(count), endpoint=False):
        edge_index = min(
            int(np.searchsorted(cumulative, distance, side="right") - 1),
            len(points) - 1,
        )
        local = (float(distance) - float(cumulative[edge_index])) / max(
            float(lengths[edge_index]), 1e-12
        )
        samples.append(
            points[edge_index] * (1.0 - local)
            + points[(edge_index + 1) % len(points)] * local
        )
    return np.asarray(samples, dtype=np.float64)


def _align_interface_loop(base_points: np.ndarray, interface_points: np.ndarray) -> np.ndarray:
    """Choose interface winding and cyclic offset with the least strip twist."""

    base_points = np.asarray(base_points, dtype=np.float64)
    interface_points = np.asarray(interface_points, dtype=np.float64)
    sample_count = max(24, 4 * max(len(base_points), len(interface_points)))
    base_samples = _resample_closed_loop(base_points, sample_count)
    best_score = float("inf")
    best = interface_points.copy()
    for oriented in (interface_points, interface_points[::-1]):
        for shift in range(len(oriented)):
            candidate = np.roll(oriented, shift, axis=0)
            candidate_samples = _resample_closed_loop(candidate, sample_count)
            score = float(np.mean(np.sum((candidate_samples - base_samples) ** 2, axis=1)))
            if score < best_score:
                best_score = score
                best = candidate.copy()
    return best


def _zipper_strip(loop_a: Sequence[int], loop_b: Sequence[int]) -> list[list[int]]:
    """Triangulate an annulus between two closed loops of arbitrary size."""

    a = [int(value) for value in loop_a]
    b = [int(value) for value in loop_b]
    if len(a) < 3 or len(b) < 3:
        raise ValueError("Interface adapter loops require at least three vertices")
    triangles: list[list[int]] = []
    i = 0
    j = 0
    while i < len(a) or j < len(b):
        next_a = (i + 1) / len(a) if i < len(a) else float("inf")
        next_b = (j + 1) / len(b) if j < len(b) else float("inf")
        a_current = a[i % len(a)]
        b_current = b[j % len(b)]
        if next_a <= next_b:
            a_next = a[(i + 1) % len(a)]
            triangles.append([a_current, a_next, b_current])
            i += 1
        else:
            b_next = b[(j + 1) % len(b)]
            triangles.append([a_current, b_next, b_current])
            j += 1
    return triangles


def _adapter_face_assignment(
    vertices: np.ndarray,
    triangles: Sequence[Sequence[int]],
    *,
    segment_id: int,
    source_center: np.ndarray,
    interfaces: Sequence[_FrozenInterface],
    model_extent: float,
) -> list[int]:
    """Assign every frozen interface a distinct local triangle on the primitive."""

    if len(interfaces) > len(triangles):
        raise ValueError(
            f"Segment {segment_id} has {len(interfaces)} interfaces but only "
            f"{len(triangles)} primitive triangles"
        )
    extent = max(float(model_extent), 1e-8)
    costs = np.empty((len(interfaces), len(triangles)), dtype=np.float64)
    for interface_index, interface in enumerate(interfaces):
        sign = _interface_side_sign(interface, segment_id, source_center)
        desired_outward = -sign * np.asarray(interface.normal_a_to_b, dtype=np.float64)
        anchor = np.asarray(interface.anchor, dtype=np.float64)
        target_area = max(float(interface.area), extent * extent * 1e-10)
        for face_index, triangle in enumerate(triangles):
            points = vertices[np.asarray(triangle, dtype=np.int64)]
            center = points.mean(axis=0)
            normal = _polygon_normal(vertices, triangle)
            area = max(_polygon_area_3d(points), extent * extent * 1e-12)
            normal_penalty = 1.0 - float(
                np.clip(np.dot(normal, desired_outward), -1.0, 1.0)
            )
            plane_penalty = abs(float(np.dot(center - anchor, desired_outward))) / extent
            lateral = center - anchor
            lateral -= desired_outward * float(np.dot(lateral, desired_outward))
            lateral_penalty = float(np.linalg.norm(lateral)) / extent
            area_penalty = max(0.0, np.sqrt(target_area / area) - 1.0)
            costs[interface_index, face_index] = (
                2.8 * normal_penalty
                + 1.4 * plane_penalty
                + 0.9 * lateral_penalty
                + 0.25 * area_penalty
            )
    rows, columns = linear_sum_assignment(costs)
    assignment = [-1] * len(interfaces)
    for row, column in zip(rows.tolist(), columns.tolist()):
        assignment[int(row)] = int(column)
    if any(index < 0 for index in assignment):
        raise ValueError(f"Unable to assign adapter faces for segment {segment_id}")
    return assignment


def _constrain_candidate_with_local_adapters(
    candidate: _Candidate,
    *,
    segment_id: int,
    source_center: np.ndarray,
    interfaces: Sequence[_FrozenInterface],
    model_extent: float,
    plane_tolerance_ratio: float,
    fallback_reason: str,
) -> _Candidate:
    """Preserve each source interface by deforming only a local primitive patch.

    Unlike the V24 convex-hull reconstruction, this construction does not
    require all interface planes to bound one convex body.  A distinct primitive
    triangle is replaced by a triangulated transition strip ending at the exact
    immutable interface polygon.  The main primitive remains in place and every
    resulting segment is one watertight (possibly locally concave) paper shell.
    """

    if not interfaces:
        return candidate
    tolerance = max(float(model_extent) * float(plane_tolerance_ratio), 1e-9)
    interior_margin = tolerance * 4.0
    vertices = np.asarray(candidate.vertices, dtype=np.float64).copy()

    # Keep the primitive body on the segment side of every fixed interface.
    # Projection is iterative because several attachment planes may meet around
    # a torso.  Interface polygons themselves are never modified.
    for _ in range(3):
        changed = False
        for interface in interfaces:
            sign = _interface_side_sign(interface, segment_id, source_center)
            normal = np.asarray(interface.normal_a_to_b, dtype=np.float64)
            signed = sign * ((vertices - np.asarray(interface.anchor)[None, :]) @ normal)
            mask = signed < interior_margin
            if np.any(mask):
                vertices[mask] += (
                    sign * normal[None, :] * (interior_margin - signed[mask])[:, None]
                )
                changed = True
        if not changed:
            break

    triangles = _triangulated_polygons(candidate.polygons)
    assignment = _adapter_face_assignment(
        vertices,
        triangles,
        segment_id=segment_id,
        source_center=source_center,
        interfaces=interfaces,
        model_extent=model_extent,
    )
    selected_faces = set(assignment)
    polygons: list[list[int]] = [
        list(face) for index, face in enumerate(triangles) if index not in selected_faces
    ]
    face_indices: dict[str, int] = {}
    interface_areas: dict[str, float] = {}
    adapter_records: list[dict[str, object]] = []

    for interface_index, interface in enumerate(interfaces):
        triangle_index = int(assignment[interface_index])
        base_loop = [int(value) for value in triangles[triangle_index]]
        base_points = vertices[np.asarray(base_loop, dtype=np.int64)]
        interface_points = _align_interface_loop(
            base_points,
            np.asarray(interface.polygon_3d, dtype=np.float64),
        )
        start = len(vertices)
        vertices = np.vstack((vertices, interface_points))
        interface_loop = list(range(start, start + len(interface_points)))
        polygons.extend(_zipper_strip(base_loop, interface_loop))
        cap_index = len(polygons)
        polygons.append(interface_loop)

        neighbor = int(
            interface.segment_b
            if int(segment_id) == interface.segment_a
            else interface.segment_a
        )
        face_indices[str(neighbor)] = int(cap_index)
        interface_areas[str(neighbor)] = float(_polygon_area_3d(interface_points))
        adapter_records.append(
            {
                "neighbor_segment_id": neighbor,
                "replaced_primitive_triangle_index": triangle_index,
                "base_triangle_vertex_indices": base_loop,
                "interface_vertex_count": int(len(interface_loop)),
                "interface_area": float(interface_areas[str(neighbor)]),
            }
        )

    constrained = _canonical_interface_adapter_candidate(
        candidate.primitive_type,
        vertices,
        polygons,
        type_penalty=candidate.type_penalty + 0.004 * len(interfaces),
        metadata={
            **candidate.metadata,
            "base_primitive_type": candidate.primitive_type,
            "frozen_interface_count": int(len(interfaces)),
            "frozen_interface_neighbors": sorted(int(value) for value in map(int, face_indices)),
            "fitting_strategy": "local_patch_fit_around_fixed_source_interfaces",
            "fixed_interface_solver": "nonconvex_local_adapter",
            "fixed_interface_convex_failure": str(fallback_reason),
            "local_interface_adapters": adapter_records,
        },
    )
    constrained.metadata["frozen_interface_face_indices"] = face_indices
    constrained.metadata["frozen_interface_areas"] = interface_areas
    constrained.metadata["untextured_contact_face_indices"] = sorted(face_indices.values())
    return constrained


def _constrain_candidate_to_interfaces(
    candidate: _Candidate,
    *,
    segment_id: int,
    source_center: np.ndarray,
    interfaces: Sequence[_FrozenInterface],
    model_extent: float,
    plane_tolerance_ratio: float,
) -> _Candidate:
    """Use the convex V24 solver when possible, otherwise a local-patch solver.

    The fallback is essential for torso-like clusters with several attachment
    seams whose planes cannot all be faces of one convex primitive.  It keeps
    every initial interface polygon exact instead of aborting the whole model.
    """

    try:
        return _constrain_candidate_to_interfaces_convex(
            candidate,
            segment_id=segment_id,
            source_center=source_center,
            interfaces=interfaces,
            model_extent=model_extent,
            plane_tolerance_ratio=plane_tolerance_ratio,
        )
    except (ValueError, QhullError, np.linalg.LinAlgError) as error:
        return _constrain_candidate_with_local_adapters(
            candidate,
            segment_id=segment_id,
            source_center=source_center,
            interfaces=interfaces,
            model_extent=model_extent,
            plane_tolerance_ratio=plane_tolerance_ratio,
            fallback_reason=f"{type(error).__name__}: {error}",
        )


def _fit_one_cluster(
    mesh: trimesh.Trimesh,
    face_ids: np.ndarray,
    label_id: int,
    total_area: float,
    config: PrimitiveFitConfig,
    rng: np.random.Generator,
    *,
    frozen_interfaces: Sequence[_FrozenInterface] = (),
    model_extent: float = 1.0,
) -> PrimitivePart:
    triangles = np.asarray(mesh.triangles[face_ids], dtype=np.float64)
    areas = np.asarray(mesh.area_faces[face_ids], dtype=np.float64)
    source_area = float(np.sum(areas))
    source_points = _sample_triangles(triangles, areas, config.fit_samples, rng)
    # Include every cluster vertex so thin tips and extrema are not lost by
    # stochastic sampling before primitive fitting.
    vertex_ids = np.unique(np.asarray(mesh.faces[face_ids], dtype=np.int64).reshape(-1))
    source_points = np.vstack((source_points, np.asarray(mesh.vertices[vertex_ids], dtype=np.float64)))
    target_faces = (
        int(config.target_faces)
        if config.target_faces > 0
        else _auto_target_faces(len(face_ids), config.max_faces)
    )
    target_faces = int(np.clip(target_faces, 4, config.max_faces))
    candidates = _build_candidates(source_points, target_faces, config)
    source_center = np.average(
        np.asarray(mesh.triangles_center[face_ids], dtype=np.float64),
        axis=0,
        weights=np.maximum(areas, 1e-15),
    )
    if frozen_interfaces:
        constrained_candidates: list[_Candidate] = []
        for candidate in candidates:
            try:
                constrained_candidates.append(
                    _constrain_candidate_to_interfaces(
                        candidate,
                        segment_id=int(label_id),
                        source_center=source_center,
                        interfaces=frozen_interfaces,
                        model_extent=model_extent,
                        plane_tolerance_ratio=config.interface_plane_tolerance_ratio,
                    )
                )
            except (ValueError, QhullError, np.linalg.LinAlgError):
                continue
        if not constrained_candidates:
            raise ValueError(
                f"No primitive candidate can preserve the frozen interfaces for segment {label_id}"
            )
        candidates = constrained_candidates
    source_volume = _hull_volume(source_points)
    source_tree = cKDTree(np.asarray(source_points, dtype=np.float64))
    for candidate in candidates:
        _score_candidate(
            candidate,
            source_points,
            source_tree,
            source_volume,
            target_faces,
            max(config.max_faces, candidate.face_count),
            config.fit_samples,
            config.complexity_weight,
            rng,
        )
    candidates.sort(key=lambda item: (item.score, item.face_count, item.primitive_type))
    selected = candidates[0]
    top_candidates = [
        {
            "primitive_type": item.primitive_type,
            "paper_face_count": int(item.face_count),
            "score": float(item.score),
            "metrics": item.metrics,
            "metadata": item.metadata,
        }
        for item in candidates[: min(10, len(candidates))]
    ]
    return PrimitivePart(
        name=f"part_{int(label_id):02d}",
        segment_id=int(label_id),
        vertices=np.asarray(selected.vertices, dtype=np.float64),
        polygons=[list(face) for face in selected.polygons],
        source_face_count=int(len(face_ids)),
        source_surface_area=source_area,
        source_center=source_center,
        primitive_type=selected.primitive_type,
        target_face_count=target_faces,
        fit_score=float(selected.score),
        metadata={
            "area_ratio": float(source_area / max(total_area, 1e-12)),
            "selected_metrics": selected.metrics,
            "selected_candidate_metadata": selected.metadata,
            "candidate_count": int(len(candidates)),
            "top_candidates": top_candidates,
            "source_convex_hull_volume": float(source_volume),
            "paper_safe": True,
            "closed_shell": True,
            "shared_geometry_vertices": True,
            "fitting_strategy": selected.metadata.get(
                "fitting_strategy", "unconstrained_primitive_fit"
            ),
            "frozen_interface_face_indices": selected.metadata.get(
                "frozen_interface_face_indices", {}
            ),
            "frozen_interface_areas": selected.metadata.get(
                "frozen_interface_areas", {}
            ),
            "frozen_interface_neighbors": selected.metadata.get(
                "frozen_interface_neighbors", []
            ),
            "untextured_contact_face_indices": selected.metadata.get(
                "untextured_contact_face_indices", []
            ),
        },
    )


def _part_test_points(part: PrimitivePart) -> np.ndarray:
    vertices = np.asarray(part.vertices, dtype=np.float64)
    points = [vertices, vertices.mean(axis=0, keepdims=True)]
    face_centers = [vertices[np.asarray(face, dtype=np.int64)].mean(axis=0) for face in part.polygons]
    if face_centers:
        points.append(np.asarray(face_centers, dtype=np.float64))
    edge_midpoints: list[np.ndarray] = []
    seen: set[tuple[int, int]] = set()
    for face in part.polygons:
        for a, b in zip(face, face[1:] + face[:1]):
            edge = tuple(sorted((int(a), int(b))))
            if edge in seen:
                continue
            seen.add(edge)
            edge_midpoints.append((vertices[edge[0]] + vertices[edge[1]]) * 0.5)
    if edge_midpoints:
        points.append(np.asarray(edge_midpoints, dtype=np.float64))
    return np.vstack(points)


def _points_strictly_inside_hull(points: np.ndarray, hull: ConvexHull, tolerance: float) -> np.ndarray:
    equations = np.asarray(hull.equations, dtype=np.float64)
    values = np.asarray(points, dtype=np.float64) @ equations[:, :3].T + equations[:, 3][None, :]
    return np.all(values < -abs(float(tolerance)), axis=1)


def _parts_overlap(a: PrimitivePart, b: PrimitivePart, tolerance: float) -> bool:
    a_bounds = a.bounds
    b_bounds = b.bounds
    depth = np.minimum(a_bounds[1], b_bounds[1]) - np.maximum(a_bounds[0], b_bounds[0])
    if np.any(depth <= tolerance):
        return False
    try:
        hull_a = ConvexHull(np.asarray(a.vertices, dtype=np.float64))
        hull_b = ConvexHull(np.asarray(b.vertices, dtype=np.float64))
    except QhullError:
        return True
    points_a = _part_test_points(a)
    points_b = _part_test_points(b)
    if np.any(_points_strictly_inside_hull(points_a, hull_b, tolerance)):
        return True
    if np.any(_points_strictly_inside_hull(points_b, hull_a, tolerance)):
        return True
    # Crossing convex shells can have no original vertex inside the other. Test
    # both centroids and AABB overlap centre as additional interior witnesses.
    witness = np.vstack(
        (
            a.center,
            b.center,
            (np.maximum(a_bounds[0], b_bounds[0]) + np.minimum(a_bounds[1], b_bounds[1])) * 0.5,
        )
    )
    return bool(
        np.any(
            _points_strictly_inside_hull(witness, hull_a, tolerance)
            & _points_strictly_inside_hull(witness, hull_b, tolerance)
        )
    )


def _basis_from_normal(
    normal: np.ndarray,
    preferred_axis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a stable right-handed tangent basis for ``normal``."""

    normal = np.asarray(normal, dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    if preferred_axis is not None:
        axis_u = np.asarray(preferred_axis, dtype=np.float64)
        axis_u = axis_u - normal * float(np.dot(axis_u, normal))
    else:
        axis_u = np.zeros(3, dtype=np.float64)
    if float(np.linalg.norm(axis_u)) <= 1e-10:
        basis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(normal)))]
        axis_u = basis - normal * float(np.dot(basis, normal))
    axis_u /= max(float(np.linalg.norm(axis_u)), 1e-12)
    axis_v = np.cross(normal, axis_u)
    axis_v /= max(float(np.linalg.norm(axis_v)), 1e-12)
    return axis_u, axis_v


def _source_label_contacts(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    valid_segment_ids: set[int],
) -> dict[tuple[int, int], _SourceContact]:
    """Recover source-label boundaries and their weighted 3D anchor points.

    PartField labels live on faces of one source mesh.  Faces with different
    labels that share an edge define the physical joint that must survive the
    later primitive approximation.
    """

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    face_pairs = np.asarray(
        getattr(mesh, "face_adjacency", np.empty((0, 2))), dtype=np.int64
    )
    adjacency_edges = np.asarray(
        getattr(mesh, "face_adjacency_edges", np.empty((0, 2))), dtype=np.int64
    )
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    accumulators: dict[tuple[int, int], dict[str, object]] = {}

    for adjacency_index, (face_a, face_b) in enumerate(face_pairs):
        label_a = int(labels[int(face_a)])
        label_b = int(labels[int(face_b)])
        if label_a == label_b:
            continue
        if label_a not in valid_segment_ids or label_b not in valid_segment_ids:
            continue
        pair = (min(label_a, label_b), max(label_a, label_b))
        if adjacency_index < len(adjacency_edges):
            edge = np.asarray(adjacency_edges[adjacency_index], dtype=np.int64)
        else:
            edge = np.intersect1d(faces[int(face_a)], faces[int(face_b)])
        if len(edge) < 2:
            continue
        p0 = vertices[int(edge[0])]
        p1 = vertices[int(edge[1])]
        length = max(float(np.linalg.norm(p1 - p0)), 1e-12)
        midpoint = (p0 + p1) * 0.5
        entry = accumulators.setdefault(
            pair,
            {
                "weighted_anchor": np.zeros(3, dtype=np.float64),
                "boundary_length": 0.0,
                "edge_count": 0,
                "boundary_points": [],
                "weighted_direction": np.zeros(3, dtype=np.float64),
            },
        )
        entry["weighted_anchor"] = np.asarray(entry["weighted_anchor"]) + midpoint * length
        entry["boundary_length"] = float(entry["boundary_length"]) + length
        entry["edge_count"] = int(entry["edge_count"]) + 1
        points = entry["boundary_points"]
        assert isinstance(points, list)
        points.extend((np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)))

        # Keep the direction consistently oriented from the lower label in the
        # pair to the higher label.  It is a stronger cue for the intended
        # interface orientation than the fitted primitive face centres alone.
        delta = face_centers[int(face_b)] - face_centers[int(face_a)]
        if label_a != pair[0]:
            delta = -delta
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > 1e-12:
            entry["weighted_direction"] = np.asarray(
                entry["weighted_direction"], dtype=np.float64
            ) + delta / delta_norm * length

    contacts: dict[tuple[int, int], _SourceContact] = {}
    for pair, entry in accumulators.items():
        boundary_length = max(float(entry["boundary_length"]), 1e-12)
        boundary_points = np.asarray(entry["boundary_points"], dtype=np.float64)
        if len(boundary_points):
            rounded = np.round(boundary_points, decimals=12)
            _, unique_indices = np.unique(rounded, axis=0, return_index=True)
            boundary_points = boundary_points[np.sort(unique_indices)]
        anchor = np.asarray(entry["weighted_anchor"], dtype=np.float64) / boundary_length
        direction = np.asarray(entry["weighted_direction"], dtype=np.float64)
        if float(np.linalg.norm(direction)) <= 1e-12:
            direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        direction /= max(float(np.linalg.norm(direction)), 1e-12)

        preferred_axis: np.ndarray | None = None
        interface_normal = np.asarray(direction, dtype=np.float64)
        if len(boundary_points) >= 2:
            centered = boundary_points - anchor[None, :]
            try:
                _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
                preferred_axis = np.asarray(vh[0], dtype=np.float64)
                # A closed attachment seam is approximately a planar loop. Use
                # its measured best-fit plane rather than the noisier difference
                # of adjacent triangle centres. Open/near-linear seams retain
                # the label-direction normal because their plane is ambiguous.
                if (
                    len(singular_values) >= 3
                    and float(singular_values[1]) > 1e-10
                    and float(singular_values[2] / singular_values[1]) < 0.35
                ):
                    measured_normal = np.asarray(vh[-1], dtype=np.float64)
                    if float(np.dot(measured_normal, direction)) < 0.0:
                        measured_normal *= -1.0
                    interface_normal = measured_normal / max(
                        float(np.linalg.norm(measured_normal)), 1e-12
                    )
            except np.linalg.LinAlgError:
                preferred_axis = None
        axis_u, axis_v = _basis_from_normal(interface_normal, preferred_axis)
        contacts[pair] = _SourceContact(
            segment_a=int(pair[0]),
            segment_b=int(pair[1]),
            anchor=anchor,
            boundary_length=boundary_length,
            edge_count=int(entry["edge_count"]),
            boundary_points=boundary_points,
            direction_a_to_b=direction,
            interface_normal=interface_normal,
            interface_axis_u=axis_u,
            interface_axis_v=axis_v,
        )
    return contacts


def _contact_spanning_tree(
    parts: Sequence[PrimitivePart],
    contacts: dict[tuple[int, int], _SourceContact],
) -> list[dict[str, object]]:
    """Build a large-part-rooted tree that connects every surviving segment.

    Source adjacency is always preferred.  If the source mesh itself contains
    disconnected components, the nearest source-centre pair bridges the two
    components so the final paper assembly still has one connected structure.
    """

    by_id = {int(part.segment_id): part for part in parts}
    root = max(parts, key=lambda item: (item.source_surface_area, item.volume))
    visited = {int(root.segment_id)}
    remaining = set(by_id) - visited
    tree: list[dict[str, object]] = []

    while remaining:
        source_candidates: list[tuple[tuple[float, float, int, int], int, int, _SourceContact]] = []
        for pair, contact in contacts.items():
            a, b = pair
            if a not in by_id or b not in by_id:
                continue
            if (a in visited) == (b in visited):
                continue
            parent = a if a in visited else b
            child = b if a in visited else a
            distance = float(
                np.linalg.norm(by_id[parent].source_center - by_id[child].source_center)
            )
            key = (-float(contact.boundary_length), distance, int(parent), int(child))
            source_candidates.append((key, parent, child, contact))

        if source_candidates:
            _, parent, child, contact = min(source_candidates, key=lambda item: item[0])
            tree.append(
                {
                    "parent": int(parent),
                    "child": int(child),
                    "anchor": np.asarray(contact.anchor, dtype=np.float64),
                    "boundary_points": np.asarray(contact.boundary_points, dtype=np.float64),
                    "direction_a_to_b": np.asarray(
                        contact.direction_a_to_b, dtype=np.float64
                    ),
                    "interface_normal": np.asarray(
                        contact.interface_normal, dtype=np.float64
                    ),
                    "interface_axis_u": np.asarray(
                        contact.interface_axis_u, dtype=np.float64
                    ),
                    "interface_axis_v": np.asarray(
                        contact.interface_axis_v, dtype=np.float64
                    ),
                    "source_adjacent": True,
                    "boundary_length": float(contact.boundary_length),
                    "boundary_edge_count": int(contact.edge_count),
                }
            )
        else:
            parent, child = min(
                (
                    (parent_id, child_id)
                    for parent_id in visited
                    for child_id in remaining
                ),
                key=lambda pair: float(
                    np.linalg.norm(
                        by_id[pair[0]].source_center - by_id[pair[1]].source_center
                    )
                ),
            )
            anchor = (
                np.asarray(by_id[parent].source_center, dtype=np.float64)
                + np.asarray(by_id[child].source_center, dtype=np.float64)
            ) * 0.5
            bridge_direction = np.asarray(
                by_id[child].source_center - by_id[parent].source_center,
                dtype=np.float64,
            )
            if float(np.linalg.norm(bridge_direction)) <= 1e-12:
                bridge_direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            bridge_direction /= max(float(np.linalg.norm(bridge_direction)), 1e-12)
            bridge_u, bridge_v = _basis_from_normal(bridge_direction)
            tree.append(
                {
                    "parent": int(parent),
                    "child": int(child),
                    "anchor": anchor,
                    "boundary_points": np.empty((0, 3), dtype=np.float64),
                    "direction_a_to_b": bridge_direction,
                    "interface_normal": bridge_direction,
                    "interface_axis_u": bridge_u,
                    "interface_axis_v": bridge_v,
                    "source_adjacent": False,
                    "boundary_length": 0.0,
                    "boundary_edge_count": 0,
                }
            )

        visited.add(int(child))
        remaining.remove(int(child))
    return tree


def _rotation_between_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a proper rotation matrix mapping ``source`` onto ``target``."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source /= max(float(np.linalg.norm(source)), 1e-12)
    target /= max(float(np.linalg.norm(target)), 1e-12)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine <= 1e-12:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        basis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(source)))]
        axis = np.cross(source, basis)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)
    axis = cross / sine
    skew = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    angle_sine = sine
    return np.eye(3) + skew * angle_sine + (skew @ skew) * (1.0 - cosine)


def _face_geometry(
    part: PrimitivePart,
    face_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    polygon = part.polygons[int(face_index)]
    points = np.asarray(part.vertices, dtype=np.float64)[
        np.asarray(polygon, dtype=np.int64)
    ]
    return points, points.mean(axis=0), _polygon_normal(np.asarray(part.vertices), polygon)


def _select_contact_face(
    part: PrimitivePart,
    anchor: np.ndarray,
    desired_outward: np.ndarray,
) -> int:
    anchor = np.asarray(anchor, dtype=np.float64)
    desired = np.asarray(desired_outward, dtype=np.float64)
    desired /= max(float(np.linalg.norm(desired)), 1e-12)
    extent = max(float(np.max(part.size)), 1e-8)
    candidates: list[tuple[float, int]] = []
    for face_index in range(part.face_count):
        _, center, normal = _face_geometry(part, face_index)
        distance = float(np.linalg.norm(center - anchor)) / extent
        plane_distance = abs(float(np.dot(anchor - center, normal))) / extent
        alignment = 1.0 - float(np.clip(np.dot(normal, desired), -1.0, 1.0))
        candidates.append((distance + 1.8 * alignment + 0.35 * plane_distance, face_index))
    return min(candidates, key=lambda item: (item[0], item[1]))[1]


def _closest_point_on_convex_face(points: np.ndarray, query: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    if len(points) == 3:
        triangles = points.reshape(1, 3, 3)
    else:
        triangles = np.asarray(
            [[points[0], points[index], points[index + 1]] for index in range(1, len(points) - 1)],
            dtype=np.float64,
        )
    queries = np.repeat(query.reshape(1, 3), len(triangles), axis=0)
    closest = trimesh.triangles.closest_point(triangles, queries)
    distances = np.linalg.norm(closest - query[None, :], axis=1)
    return np.asarray(closest[int(np.argmin(distances))], dtype=np.float64)


def _face_basis(points: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = np.asarray(points, dtype=np.float64).mean(axis=0)
    edges = np.roll(points, -1, axis=0) - points
    axis_u = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
    axis_u /= max(float(np.linalg.norm(axis_u)), 1e-12)
    axis_v = np.cross(normal, axis_u)
    axis_v /= max(float(np.linalg.norm(axis_v)), 1e-12)
    return origin, axis_u, axis_v


def _polygon_area_2d(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return 0.0
    return 0.5 * float(
        np.sum(points[:, 0] * np.roll(points[:, 1], -1) - points[:, 1] * np.roll(points[:, 0], -1))
    )


def _line_intersection_2d(
    segment_start: np.ndarray,
    segment_end: np.ndarray,
    clip_start: np.ndarray,
    clip_end: np.ndarray,
) -> np.ndarray:
    segment = segment_end - segment_start
    clip = clip_end - clip_start
    denominator = segment[0] * clip[1] - segment[1] * clip[0]
    if abs(float(denominator)) <= 1e-14:
        return (segment_start + segment_end) * 0.5
    offset = clip_start - segment_start
    t = (offset[0] * clip[1] - offset[1] * clip[0]) / denominator
    return segment_start + float(t) * segment


def _cross_2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _convex_intersection_area_2d(subject: np.ndarray, clip: np.ndarray) -> float:
    output = [np.asarray(point, dtype=np.float64) for point in np.asarray(subject)]
    clip = np.asarray(clip, dtype=np.float64)
    if _polygon_area_2d(clip) < 0.0:
        clip = clip[::-1]
    epsilon = 1e-10
    for clip_start, clip_end in zip(clip, np.roll(clip, -1, axis=0)):
        if not output:
            break
        input_points = output
        output = []
        previous = input_points[-1]
        previous_inside = _cross_2d(clip_end - clip_start, previous - clip_start) >= -epsilon
        for current in input_points:
            current_inside = _cross_2d(clip_end - clip_start, current - clip_start) >= -epsilon
            if current_inside:
                if not previous_inside:
                    output.append(
                        _line_intersection_2d(previous, current, clip_start, clip_end)
                    )
                output.append(current)
            elif previous_inside:
                output.append(_line_intersection_2d(previous, current, clip_start, clip_end))
            previous = current
            previous_inside = current_inside
    if len(output) < 3:
        return 0.0
    return abs(_polygon_area_2d(np.asarray(output, dtype=np.float64)))


def _enforce_primitive_contacts_move(
    parts: list[PrimitivePart],
    contacts: dict[tuple[int, int], _SourceContact],
    contact_overlap_ratio: float,
) -> list[dict[str, object]]:
    """Restore source adjacency as exact planar joints between closed shells.

    For each edge of a large-part-rooted contact tree, an existing face is
    selected on both primitives near the original PartField boundary.  The child
    is minimally rotated and translated so the two faces become coplanar with
    opposite outward normals and a positive common area.  This produces a real
    face-to-face paper-model joint rather than merely placing two AABBs nearby.
    """

    if len(parts) < 2:
        for part in parts:
            part.metadata["contact_constraint_enabled"] = True
            part.metadata["contact_graph_connected"] = True
        return []

    by_id = {int(part.segment_id): part for part in parts}
    all_bounds = np.vstack(
        (
            np.vstack([part.bounds[0] for part in parts]).min(axis=0),
            np.vstack([part.bounds[1] for part in parts]).max(axis=0),
        )
    )
    model_extent = max(float(np.max(all_bounds[1] - all_bounds[0])), 1e-8)
    plane_tolerance = model_extent * 2e-6
    requested_overlap = max(0.0, float(contact_overlap_ratio)) * model_extent
    tree = _contact_spanning_tree(parts, contacts)
    records: list[dict[str, object]] = []
    rigid_transforms = {
        int(part.segment_id): np.eye(4, dtype=np.float64) for part in parts
    }

    def transformed_point(segment_id: int, point: np.ndarray) -> np.ndarray:
        transform = rigid_transforms[int(segment_id)]
        return transform[:3, :3] @ np.asarray(point, dtype=np.float64) + transform[:3, 3]

    for edge_index, edge in enumerate(tree):
        parent = by_id[int(edge["parent"])]
        child = by_id[int(edge["child"])]
        source_anchor = np.asarray(edge["anchor"], dtype=np.float64)
        # A parent can itself have been rotated and translated on an earlier tree
        # edge. Apply that accumulated rigid transform to all later source seams.
        anchor = transformed_point(parent.segment_id, source_anchor)
        parent_source_center_current = transformed_point(
            parent.segment_id, parent.source_center
        )
        child_source_center_current = transformed_point(
            child.segment_id, child.source_center
        )
        outward_hint = child_source_center_current - parent_source_center_current
        if float(np.linalg.norm(outward_hint)) <= 1e-12:
            outward_hint = np.asarray(child.center) - np.asarray(parent.center)

        parent_face_index = _select_contact_face(parent, anchor, outward_hint)
        parent_points, parent_face_center, parent_normal = _face_geometry(
            parent, parent_face_index
        )
        child_anchor = transformed_point(child.segment_id, source_anchor)
        child_face_index = _select_contact_face(child, child_anchor, -parent_normal)
        child_points, child_face_center, child_normal = _face_geometry(child, child_face_index)

        rotation = _rotation_between_vectors(child_normal, -parent_normal)
        child.rotate_about(child_face_center, rotation)
        child_points, child_face_center, child_normal_after = _face_geometry(
            child, child_face_index
        )

        projected_anchor = anchor - float(np.dot(anchor - parent_face_center, parent_normal)) * parent_normal
        target = _closest_point_on_convex_face(parent_points, projected_anchor)
        # Move a boundary-clamped target slightly towards the face interior so
        # even a source seam near an edge yields a positive common contact area.
        target = target + 0.08 * (parent_face_center - target)
        translation = target - child_face_center
        if requested_overlap > 0.0:
            # Optional hidden insertion into the parent.  The default is zero,
            # which leaves an exact coplanar, non-interpenetrating paper joint.
            translation -= parent_normal * min(
                requested_overlap,
                0.04 * max(float(np.min(child.size)), plane_tolerance),
            )
        child.translate(translation)
        step_transform = np.eye(4, dtype=np.float64)
        step_transform[:3, :3] = rotation
        step_transform[:3, 3] = (
            child_face_center - rotation @ child_face_center + translation
        )
        rigid_transforms[int(child.segment_id)] = (
            step_transform @ rigid_transforms[int(child.segment_id)]
        )
        child_points, child_face_center, child_normal_after = _face_geometry(
            child, child_face_index
        )

        origin, axis_u, axis_v = _face_basis(parent_points, parent_normal)
        parent_2d = np.column_stack(
            ((parent_points - origin) @ axis_u, (parent_points - origin) @ axis_v)
        )
        child_2d = np.column_stack(
            ((child_points - origin) @ axis_u, (child_points - origin) @ axis_v)
        )
        contact_area = _convex_intersection_area_2d(child_2d, parent_2d)
        plane_error = float(np.max(np.abs((child_points - parent_face_center) @ parent_normal)))
        normal_alignment = float(np.dot(child_normal_after, -parent_normal))
        connected = bool(
            contact_area > plane_tolerance * plane_tolerance
            and plane_error <= plane_tolerance + requested_overlap
            and normal_alignment >= 1.0 - 1e-6
        )

        record = {
            "tree_edge_index": int(edge_index),
            "parent_segment_id": int(parent.segment_id),
            "child_segment_id": int(child.segment_id),
            "source_adjacent": bool(edge["source_adjacent"]),
            "source_boundary_anchor": source_anchor.tolist(),
            "resolved_boundary_anchor": anchor.tolist(),
            "source_boundary_length": float(edge["boundary_length"]),
            "source_boundary_edge_count": int(edge["boundary_edge_count"]),
            "parent_face_index": int(parent_face_index),
            "child_face_index": int(child_face_index),
            "rotation_matrix": rotation.tolist(),
            "accumulated_rigid_transform": rigid_transforms[int(child.segment_id)].tolist(),
            "translation": np.asarray(translation, dtype=np.float64).tolist(),
            "contact_overlap": float(requested_overlap),
            "contact_area": float(contact_area),
            "plane_error": float(plane_error),
            "normal_alignment": float(normal_alignment),
            "connected": connected,
        }
        records.append(record)
        child.metadata["contact_tree_parent_segment_id"] = int(parent.segment_id)
        child.metadata["contact_tree_source_adjacent"] = bool(edge["source_adjacent"])
        child.metadata["contact_tree_parent_face_index"] = int(parent_face_index)
        child.metadata["contact_tree_child_face_index"] = int(child_face_index)
        child.metadata["contact_tree_translation"] = record["translation"]
        child.metadata["contact_tree_rigid_transform"] = record[
            "accumulated_rigid_transform"
        ]
        child.metadata["contact_tree_contact_area"] = float(contact_area)
        child.metadata["contact_tree_plane_error"] = float(plane_error)
        child.metadata["contact_tree_connected"] = connected

    graph_connected = all(bool(record["connected"]) for record in records)
    compact_tree = [
        {
            "parent_segment_id": int(record["parent_segment_id"]),
            "child_segment_id": int(record["child_segment_id"]),
            "source_adjacent": bool(record["source_adjacent"]),
            "parent_face_index": int(record["parent_face_index"]),
            "child_face_index": int(record["child_face_index"]),
            "contact_area": float(record["contact_area"]),
            "connected": bool(record["connected"]),
        }
        for record in records
    ]
    for part in parts:
        part.metadata["contact_constraint_enabled"] = True
        part.metadata["contact_overlap_ratio"] = float(contact_overlap_ratio)
        part.metadata["contact_graph_connected"] = bool(graph_connected)
        part.metadata["contact_tree"] = compact_tree
    if not graph_connected:
        failed = [
            [int(record["parent_segment_id"]), int(record["child_segment_id"])]
            for record in records
            if not bool(record["connected"])
        ]
        raise ValueError(f"Primitive face-contact fitting failed for tree edges: {failed}")
    return records


def _face_contact_center(
    points: np.ndarray,
    normal: np.ndarray,
    anchor: np.ndarray,
    inset_ratio: float,
) -> np.ndarray:
    """Return an interior point on a convex face close to a source seam."""

    points = np.asarray(points, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    face_center = points.mean(axis=0)
    projected = anchor - float(np.dot(anchor - face_center, normal)) * normal
    closest = _closest_point_on_convex_face(points, projected)
    ratio = float(np.clip(inset_ratio, 0.0, 0.9))
    return closest + ratio * (face_center - closest)


def _face_patch_capacity(
    points: np.ndarray,
    normal: np.ndarray,
    center: np.ndarray,
    preferred_axis: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return the largest centred circular patch radius inside a convex face."""

    points = np.asarray(points, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    axis_u, axis_v = _basis_from_normal(normal, preferred_axis)
    polygon_2d = np.column_stack((points @ axis_u, points @ axis_v))
    center_2d = np.asarray([float(center @ axis_u), float(center @ axis_v)])
    if _polygon_area_2d(polygon_2d) < 0.0:
        polygon_2d = polygon_2d[::-1]
    distances: list[float] = []
    for start, end in zip(polygon_2d, np.roll(polygon_2d, -1, axis=0)):
        edge = end - start
        edge_length = float(np.linalg.norm(edge))
        if edge_length <= 1e-12:
            continue
        distances.append(_cross_2d(edge, center_2d - start) / edge_length)
    capacity = min(distances) if distances else 0.0
    area = abs(_polygon_area_2d(polygon_2d))
    return max(0.0, float(capacity)), float(area), axis_u, axis_v


def _select_contact_face_pair(
    parent: PrimitivePart,
    child: PrimitivePart,
    edge: dict[str, object],
    *,
    model_extent: float,
    connector_radius_ratio: float,
    connector_inset_ratio: float,
) -> dict[str, object]:
    """Select a facing pair jointly instead of choosing both faces independently.

    V22 chose one face at a time and then moved the complete child primitive to
    it.  That often selected a narrow triangle or a remote face.  V23 scores a
    face pair using source-seam proximity, opposing normals, connector length,
    and the radius of a patch that can actually fit inside both faces.
    """

    anchor = np.asarray(edge["anchor"], dtype=np.float64)
    source_direction = np.asarray(edge["direction_a_to_b"], dtype=np.float64)
    if int(parent.segment_id) != min(int(parent.segment_id), int(child.segment_id)):
        source_direction = -source_direction
    center_direction = np.asarray(child.source_center - parent.source_center, dtype=np.float64)
    if float(np.linalg.norm(center_direction)) <= 1e-12:
        center_direction = np.asarray(child.center - parent.center, dtype=np.float64)
    if float(np.linalg.norm(center_direction)) > 1e-12:
        center_direction /= float(np.linalg.norm(center_direction))
        if float(np.dot(source_direction, center_direction)) < 0.0:
            source_direction = -source_direction
        desired_parent = 0.72 * source_direction + 0.28 * center_direction
    else:
        desired_parent = source_direction
    desired_parent /= max(float(np.linalg.norm(desired_parent)), 1e-12)
    desired_child = -desired_parent
    preferred_axis = np.asarray(edge["interface_axis_u"], dtype=np.float64)
    target_radius = max(float(connector_radius_ratio) * model_extent, model_extent * 1e-5)

    best: tuple[float, dict[str, object]] | None = None
    for parent_face_index in range(parent.face_count):
        parent_points, parent_face_center, parent_normal = _face_geometry(
            parent, parent_face_index
        )
        parent_contact_center = _face_contact_center(
            parent_points,
            parent_normal,
            anchor,
            connector_inset_ratio,
        )
        parent_capacity, parent_area, parent_u, parent_v = _face_patch_capacity(
            parent_points,
            parent_normal,
            parent_contact_center,
            preferred_axis,
        )
        if parent_capacity <= model_extent * 1e-8:
            continue
        for child_face_index in range(child.face_count):
            child_points, child_face_center, child_normal = _face_geometry(
                child, child_face_index
            )
            child_contact_center = _face_contact_center(
                child_points,
                child_normal,
                anchor,
                connector_inset_ratio,
            )
            child_capacity, child_area, child_u, child_v = _face_patch_capacity(
                child_points,
                child_normal,
                child_contact_center,
                preferred_axis,
            )
            if child_capacity <= model_extent * 1e-8:
                continue

            connector_vector = child_contact_center - parent_contact_center
            connector_length = float(np.linalg.norm(connector_vector))
            if connector_length > model_extent * 1e-10:
                connector_direction = connector_vector / connector_length
            else:
                connector_direction = desired_parent

            anchor_distance = (
                float(np.linalg.norm(parent_contact_center - anchor))
                + float(np.linalg.norm(child_contact_center - anchor))
            ) / model_extent
            plane_distance = (
                abs(float(np.dot(anchor - parent_face_center, parent_normal)))
                + abs(float(np.dot(anchor - child_face_center, child_normal)))
            ) / model_extent
            normal_penalty = (
                1.0 - float(np.clip(np.dot(parent_normal, desired_parent), -1.0, 1.0))
                + 1.0 - float(np.clip(np.dot(child_normal, desired_child), -1.0, 1.0))
            )
            facing_penalty = (
                1.0 - float(np.clip(np.dot(parent_normal, connector_direction), -1.0, 1.0))
                + 1.0 - float(np.clip(np.dot(child_normal, -connector_direction), -1.0, 1.0))
            )
            available_radius = min(parent_capacity, child_capacity) * 0.72
            capacity_penalty = max(0.0, target_radius / max(available_radius, 1e-12) - 1.0)
            compactness_penalty = target_radius / max(
                np.sqrt(max(min(parent_area, child_area), 1e-12)), 1e-12
            )
            score = (
                1.15 * anchor_distance
                + 0.30 * plane_distance
                + 1.65 * normal_penalty
                + 1.15 * facing_penalty
                + 0.22 * connector_length / model_extent
                + 1.80 * capacity_penalty
                + 0.08 * compactness_penalty
            )
            record = {
                "parent_face_index": int(parent_face_index),
                "child_face_index": int(child_face_index),
                "parent_points": parent_points,
                "child_points": child_points,
                "parent_face_center": parent_face_center,
                "child_face_center": child_face_center,
                "parent_normal": parent_normal,
                "child_normal": child_normal,
                "parent_contact_center": parent_contact_center,
                "child_contact_center": child_contact_center,
                "parent_capacity": float(parent_capacity),
                "child_capacity": float(child_capacity),
                "parent_area": float(parent_area),
                "child_area": float(child_area),
                "parent_axis_u": parent_u,
                "parent_axis_v": parent_v,
                "child_axis_u": child_u,
                "child_axis_v": child_v,
                "desired_parent": desired_parent,
                "connector_length": float(connector_length),
                "selection_score": float(score),
            }
            candidate = (float(score), record)
            if best is None or candidate[0] < best[0]:
                best = candidate

    if best is None:
        raise ValueError(
            f"No usable connector face pair for segments {parent.segment_id} and {child.segment_id}"
        )
    return best[1]


def _direct_face_contact_area(
    selection: dict[str, object],
    plane_tolerance: float,
) -> tuple[float, float, float]:
    parent_points = np.asarray(selection["parent_points"], dtype=np.float64)
    child_points = np.asarray(selection["child_points"], dtype=np.float64)
    parent_normal = np.asarray(selection["parent_normal"], dtype=np.float64)
    child_normal = np.asarray(selection["child_normal"], dtype=np.float64)
    parent_center = np.asarray(selection["parent_face_center"], dtype=np.float64)
    plane_error = float(np.max(np.abs((child_points - parent_center) @ parent_normal)))
    normal_alignment = float(np.dot(child_normal, -parent_normal))
    if plane_error > plane_tolerance or normal_alignment < 1.0 - 1e-5:
        return 0.0, plane_error, normal_alignment
    origin, axis_u, axis_v = _face_basis(parent_points, parent_normal)
    parent_2d = np.column_stack(
        ((parent_points - origin) @ axis_u, (parent_points - origin) @ axis_v)
    )
    child_2d = np.column_stack(
        ((child_points - origin) @ axis_u, (child_points - origin) @ axis_v)
    )
    return (
        float(_convex_intersection_area_2d(child_2d, parent_2d)),
        plane_error,
        normal_alignment,
    )


def _regular_ring(
    center: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    radius: float,
    sides: int,
) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, int(sides), endpoint=False)
    return (
        np.asarray(center, dtype=np.float64)[None, :]
        + float(radius) * np.cos(angles)[:, None] * np.asarray(axis_u)[None, :]
        + float(radius) * np.sin(angles)[:, None] * np.asarray(axis_v)[None, :]
    )


def _align_ring_correspondence(parent_ring: np.ndarray, child_ring: np.ndarray) -> np.ndarray:
    parent_ring = np.asarray(parent_ring, dtype=np.float64)
    child_ring = np.asarray(child_ring, dtype=np.float64)
    best_score = float("inf")
    best = child_ring
    for candidate_base in (child_ring, child_ring[::-1]):
        for shift in range(len(child_ring)):
            candidate = np.roll(candidate_base, shift, axis=0)
            score = float(np.sum((candidate - parent_ring) ** 2))
            if score < best_score:
                best_score = score
                best = candidate.copy()
    return best


def _regular_polygon_area(radius: float, sides: int) -> float:
    return 0.5 * int(sides) * float(radius) ** 2 * float(
        np.sin(2.0 * np.pi / int(sides))
    )


def _build_connector_part(
    parent: PrimitivePart,
    child: PrimitivePart,
    edge: dict[str, object],
    selection: dict[str, object],
    *,
    connector_segment_id: int,
    model_extent: float,
    connector_sides: int,
    connector_radius_ratio: float,
    connector_min_length_ratio: float,
) -> tuple[PrimitivePart, dict[str, object]]:
    sides = int(connector_sides)
    boundary_length = float(edge["boundary_length"])
    base_radius = max(float(connector_radius_ratio) * model_extent, model_extent * 1e-5)
    if boundary_length > 0.0:
        equivalent_radius = boundary_length / (2.0 * np.pi)
        requested_radius = max(base_radius, min(2.5 * base_radius, 0.55 * equivalent_radius))
    else:
        requested_radius = base_radius
    available_radius = 0.72 * min(
        float(selection["parent_capacity"]),
        float(selection["child_capacity"]),
    )
    radius = min(requested_radius, available_radius)
    minimum_radius = model_extent * 2e-5
    if radius <= minimum_radius:
        raise ValueError(
            f"Contact faces for segments {parent.segment_id}/{child.segment_id} are too small "
            f"for a paper connector (radius={radius:.8g})"
        )

    parent_center = np.asarray(selection["parent_contact_center"], dtype=np.float64)
    child_center = np.asarray(selection["child_contact_center"], dtype=np.float64)
    parent_normal = np.asarray(selection["parent_normal"], dtype=np.float64)
    child_normal = np.asarray(selection["child_normal"], dtype=np.float64)
    minimum_length = max(float(connector_min_length_ratio) * model_extent, model_extent * 1e-6)
    connector_length = float(np.linalg.norm(child_center - parent_center))
    embedded = False
    if connector_length < minimum_length:
        # Rarely, two selected faces intersect but are not coplanar enough to be
        # accepted as a direct joint.  Give the connector a very small hidden
        # insertion into both shells rather than creating a zero-volume piece.
        parent_center = parent_center - parent_normal * (minimum_length * 0.5)
        child_center = child_center - child_normal * (minimum_length * 0.5)
        connector_length = float(np.linalg.norm(child_center - parent_center))
        embedded = True

    parent_ring = _regular_ring(
        parent_center,
        np.asarray(selection["parent_axis_u"], dtype=np.float64),
        np.asarray(selection["parent_axis_v"], dtype=np.float64),
        radius,
        sides,
    )
    child_ring = _regular_ring(
        child_center,
        np.asarray(selection["child_axis_u"], dtype=np.float64),
        np.asarray(selection["child_axis_v"], dtype=np.float64),
        radius,
        sides,
    )
    child_ring = _align_ring_correspondence(parent_ring, child_ring)
    vertices = np.vstack((parent_ring, child_ring))
    polygons: list[list[int]] = [
        list(range(sides)),
        list(range(sides, 2 * sides)),
    ]
    for index in range(sides):
        following = (index + 1) % sides
        # The two end planes can have different normals.  A four-corner side
        # would therefore often be non-planar, which is invalid for a paper
        # face even though OBJ accepts it.  Two explicit triangles guarantee
        # that every connector face is planar and unfoldable.
        polygons.append([index, following, sides + following])
        polygons.append([index, sides + following, sides + index])
    polygons = _orient_outward(vertices, polygons)
    contact_area = _regular_polygon_area(radius, sides)
    metadata: dict[str, object] = {
        "connector_part": True,
        "paper_joint": True,
        "contact_mode": "connector_patch",
        "parent_segment_id": int(parent.segment_id),
        "child_segment_id": int(child.segment_id),
        "source_adjacent": bool(edge["source_adjacent"]),
        "source_boundary_anchor": np.asarray(edge["anchor"], dtype=float).tolist(),
        "source_boundary_length": boundary_length,
        "source_boundary_edge_count": int(edge["boundary_edge_count"]),
        "parent_face_index": int(selection["parent_face_index"]),
        "child_face_index": int(selection["child_face_index"]),
        "parent_patch_center": parent_center.tolist(),
        "child_patch_center": child_center.tolist(),
        "patch_radius": float(radius),
        "patch_sides": int(sides),
        "parent_contact_area": float(contact_area),
        "child_contact_area": float(contact_area),
        "connector_length": float(connector_length),
        "connector_embedded_for_min_length": bool(embedded),
        "face_pair_selection_score": float(selection["selection_score"]),
        "side_faces_triangulated": True,
        "untextured_contact_face_indices": [0, 1],
        "closed_shell": True,
        "shared_geometry_vertices": True,
    }
    connector = PrimitivePart(
        name=f"joint_{int(parent.segment_id)}_{int(child.segment_id)}",
        segment_id=int(connector_segment_id),
        vertices=np.asarray(vertices, dtype=np.float64),
        polygons=polygons,
        source_face_count=0,
        source_surface_area=float(contact_area * 2.0),
        source_center=(parent_center + child_center) * 0.5,
        primitive_type=f"connector_frustum_{sides}",
        target_face_count=int(2 * sides + 2),
        fit_score=float(selection["selection_score"]),
        metadata=metadata,
    )
    record = {
        "contact_mode": "connector_patch",
        "connector_segment_id": int(connector_segment_id),
        "contact_area": float(contact_area),
        "connector_length": float(connector_length),
        "patch_radius": float(radius),
        "plane_error": 0.0,
        "normal_alignment": 1.0,
        "connected": True,
    }
    return connector, record


def _enforce_primitive_contacts_connector(
    source_parts: list[PrimitivePart],
    contacts: dict[tuple[int, int], _SourceContact],
    config: PrimitiveFitConfig,
) -> tuple[list[dict[str, object]], list[PrimitivePart]]:
    """Connect fitted parts without rotating or relocating their main shapes.

    Each source seam first attempts a genuine existing face-to-face joint.  If
    that is unavailable, a small closed frustum is created between interior
    patches on a jointly selected pair of faces.  The connector is an explicit
    paper piece, so the original primitive silhouettes and PartField placement
    are preserved while every tree edge receives a positive-area joint.
    """

    if len(source_parts) < 2:
        for part in source_parts:
            part.metadata["contact_constraint_enabled"] = True
            part.metadata["contact_graph_connected"] = True
            part.metadata["contact_mode"] = "connector"
        return [], []

    by_id = {int(part.segment_id): part for part in source_parts}
    all_bounds = np.vstack(
        (
            np.vstack([part.bounds[0] for part in source_parts]).min(axis=0),
            np.vstack([part.bounds[1] for part in source_parts]).max(axis=0),
        )
    )
    model_extent = max(float(np.max(all_bounds[1] - all_bounds[0])), 1e-8)
    plane_tolerance = model_extent * 2e-6
    tree = _contact_spanning_tree(source_parts, contacts)
    records: list[dict[str, object]] = []
    connectors: list[PrimitivePart] = []

    for edge_index, edge in enumerate(tree):
        parent = by_id[int(edge["parent"])]
        child = by_id[int(edge["child"])]
        selection = _select_contact_face_pair(
            parent,
            child,
            edge,
            model_extent=model_extent,
            connector_radius_ratio=config.connector_radius_ratio,
            connector_inset_ratio=config.connector_inset_ratio,
        )
        direct_area, plane_error, normal_alignment = _direct_face_contact_area(
            selection,
            plane_tolerance,
        )
        minimum_direct_area = max(
            model_extent * model_extent * 1e-9,
            0.08
            * min(float(selection["parent_area"]), float(selection["child_area"])),
        )

        base_record: dict[str, object] = {
            "tree_edge_index": int(edge_index),
            "parent_segment_id": int(parent.segment_id),
            "child_segment_id": int(child.segment_id),
            "source_adjacent": bool(edge["source_adjacent"]),
            "source_boundary_anchor": np.asarray(edge["anchor"], dtype=float).tolist(),
            "source_boundary_length": float(edge["boundary_length"]),
            "source_boundary_edge_count": int(edge["boundary_edge_count"]),
            "parent_face_index": int(selection["parent_face_index"]),
            "child_face_index": int(selection["child_face_index"]),
            "face_pair_selection_score": float(selection["selection_score"]),
            "main_part_rigid_transform_applied": False,
            "plane_error": float(plane_error),
            "normal_alignment": float(normal_alignment),
        }
        if direct_area >= minimum_direct_area:
            record = {
                **base_record,
                "contact_mode": "direct_face",
                "connector_segment_id": None,
                "contact_area": float(direct_area),
                "connected": True,
            }
        else:
            connector_id = -1_000_000 - edge_index
            connector, connector_record = _build_connector_part(
                parent,
                child,
                edge,
                selection,
                connector_segment_id=connector_id,
                model_extent=model_extent,
                connector_sides=config.connector_sides,
                connector_radius_ratio=config.connector_radius_ratio,
                connector_min_length_ratio=config.connector_min_length_ratio,
            )
            connectors.append(connector)
            record = {**base_record, **connector_record}
        records.append(record)

        child.metadata["contact_tree_parent_segment_id"] = int(parent.segment_id)
        child.metadata["contact_tree_source_adjacent"] = bool(edge["source_adjacent"])
        child.metadata["contact_tree_parent_face_index"] = int(
            selection["parent_face_index"]
        )
        child.metadata["contact_tree_child_face_index"] = int(selection["child_face_index"])
        child.metadata["contact_tree_contact_mode"] = str(record["contact_mode"])
        child.metadata["contact_tree_connector_segment_id"] = record[
            "connector_segment_id"
        ]
        child.metadata["contact_tree_contact_area"] = float(record["contact_area"])
        child.metadata["contact_tree_plane_error"] = float(record["plane_error"])
        child.metadata["contact_tree_connected"] = bool(record["connected"])
        child.metadata["contact_tree_main_part_moved"] = False

    graph_connected = all(bool(record["connected"]) for record in records)
    compact_tree = [
        {
            "parent_segment_id": int(record["parent_segment_id"]),
            "child_segment_id": int(record["child_segment_id"]),
            "source_adjacent": bool(record["source_adjacent"]),
            "contact_mode": str(record["contact_mode"]),
            "connector_segment_id": record["connector_segment_id"],
            "parent_face_index": int(record["parent_face_index"]),
            "child_face_index": int(record["child_face_index"]),
            "contact_area": float(record["contact_area"]),
            "connected": bool(record["connected"]),
        }
        for record in records
    ]
    for part in source_parts:
        part.metadata["contact_constraint_enabled"] = True
        part.metadata["contact_mode"] = "connector"
        part.metadata["contact_overlap_ratio"] = float(config.contact_overlap_ratio)
        part.metadata["contact_graph_connected"] = bool(graph_connected)
        part.metadata["contact_tree"] = compact_tree
        part.metadata["main_part_rigid_transform_applied"] = False
    for connector in connectors:
        connector.metadata["contact_constraint_enabled"] = True
        connector.metadata["contact_graph_connected"] = bool(graph_connected)
        connector.metadata["contact_tree"] = compact_tree
    if not graph_connected:
        failed = [
            [int(record["parent_segment_id"]), int(record["child_segment_id"])]
            for record in records
            if not bool(record["connected"])
        ]
        raise ValueError(f"Primitive connector fitting failed for tree edges: {failed}")
    return records, connectors


def _enforce_primitive_contacts_fixed(
    source_parts: list[PrimitivePart],
    contacts: dict[tuple[int, int], _SourceContact],
    frozen_interfaces: dict[tuple[int, int], _FrozenInterface],
    config: PrimitiveFitConfig,
) -> tuple[list[dict[str, object]], list[PrimitivePart]]:
    """Validate frozen source joints and bridge only truly disconnected components.

    Source-adjacent parts already contain the same interface polygon because the
    interface was inserted before candidate scoring.  This pass never changes a
    source part.  It only records/validates those immutable joints.  A V23-style
    connector is retained solely as a fallback between source components that
    had no PartField boundary at all.
    """

    if len(source_parts) < 2:
        for part in source_parts:
            part.metadata["contact_constraint_enabled"] = True
            part.metadata["contact_graph_connected"] = True
            part.metadata["contact_mode"] = "fixed"
            part.metadata["main_part_rigid_transform_applied"] = False
            part.metadata["source_interface_geometry_changed"] = False
        return [], []

    by_id = {int(part.segment_id): part for part in source_parts}
    bounds = np.vstack(
        (
            np.vstack([part.bounds[0] for part in source_parts]).min(axis=0),
            np.vstack([part.bounds[1] for part in source_parts]).max(axis=0),
        )
    )
    model_extent = max(float(np.max(bounds[1] - bounds[0])), 1e-8)
    tolerance = max(
        model_extent * float(config.interface_plane_tolerance_ratio) * 12.0,
        1e-8,
    )
    tree = _contact_spanning_tree(source_parts, contacts)
    tree_by_pair = {
        tuple(sorted((int(edge["parent"]), int(edge["child"])))): edge for edge in tree
    }
    records: list[dict[str, object]] = []
    connectors: list[PrimitivePart] = []
    record_by_pair: dict[tuple[int, int], dict[str, object]] = {}

    for pair, interface in sorted(frozen_interfaces.items()):
        if pair[0] not in by_id or pair[1] not in by_id:
            continue
        part_a = by_id[pair[0]]
        part_b = by_id[pair[1]]
        face_a = int(part_a.metadata["frozen_interface_face_indices"][str(pair[1])])
        face_b = int(part_b.metadata["frozen_interface_face_indices"][str(pair[0])])
        points_a, _, normal_a = _face_geometry(part_a, face_a)
        points_b, _, normal_b = _face_geometry(part_b, face_b)
        tree_edge = tree_by_pair.get(pair)
        if tree_edge is not None:
            parent_id = int(tree_edge["parent"])
            child_id = int(tree_edge["child"])
        else:
            parent_id, child_id = int(pair[0]), int(pair[1])

        plane_error = max(
            float(np.max(np.abs((points_a - interface.anchor[None, :]) @ interface.normal_a_to_b))),
            float(np.max(np.abs((points_b - interface.anchor[None, :]) @ interface.normal_a_to_b))),
        )
        tree_a = cKDTree(points_a)
        tree_b = cKDTree(points_b)
        vertex_error = max(
            float(np.max(tree_a.query(points_b, k=1)[0])),
            float(np.max(tree_b.query(points_a, k=1)[0])),
        )
        area_a = _polygon_area_3d(points_a)
        area_b = _polygon_area_3d(points_b)
        normal_alignment = float(np.clip(-np.dot(normal_a, normal_b), -1.0, 1.0))
        connected = bool(
            plane_error <= tolerance
            and vertex_error <= tolerance * 8.0
            and min(area_a, area_b) > model_extent * model_extent * 1e-10
            and normal_alignment > 0.999
        )
        record: dict[str, object] = {
            "parent_segment_id": parent_id,
            "child_segment_id": child_id,
            "source_adjacent": True,
            "source_boundary_anchor": np.asarray(interface.anchor, dtype=float).tolist(),
            "source_boundary_length": float(interface.source_boundary_length),
            "source_boundary_edge_count": int(interface.source_boundary_edge_count),
            "contact_mode": "fixed_interface",
            "connector_segment_id": None,
            "parent_face_index": face_a if parent_id == pair[0] else face_b,
            "child_face_index": face_b if child_id == pair[1] else face_a,
            "contact_area": float(min(area_a, area_b)),
            "interface_vertex_count": int(len(interface.polygon_3d)),
            "interface_fallback_rectangle": bool(interface.fallback_rectangle),
            "plane_error": float(plane_error),
            "shared_vertex_error": float(vertex_error),
            "normal_alignment": float(normal_alignment),
            "main_part_rigid_transform_applied": False,
            "source_interface_geometry_changed": False,
            "connected": connected,
        }
        records.append(record)
        record_by_pair[pair] = record

    # Preserve all real source seams.  Only bridge a disconnected source
    # component, where no initial interface exists to freeze.
    for edge_index, edge in enumerate(tree):
        if bool(edge["source_adjacent"]):
            continue
        parent = by_id[int(edge["parent"])]
        child = by_id[int(edge["child"])]
        selection = _select_contact_face_pair(
            parent,
            child,
            edge,
            model_extent=model_extent,
            connector_radius_ratio=config.connector_radius_ratio,
            connector_inset_ratio=config.connector_inset_ratio,
        )
        connector_id = -2_000_000 - edge_index
        connector, connector_record = _build_connector_part(
            parent,
            child,
            edge,
            selection,
            connector_segment_id=connector_id,
            model_extent=model_extent,
            connector_sides=config.connector_sides,
            connector_radius_ratio=config.connector_radius_ratio,
            connector_min_length_ratio=config.connector_min_length_ratio,
        )
        connectors.append(connector)
        record = {
            "parent_segment_id": int(parent.segment_id),
            "child_segment_id": int(child.segment_id),
            "source_adjacent": False,
            "source_boundary_anchor": np.asarray(edge["anchor"], dtype=float).tolist(),
            "source_boundary_length": 0.0,
            "source_boundary_edge_count": 0,
            "parent_face_index": int(selection["parent_face_index"]),
            "child_face_index": int(selection["child_face_index"]),
            "main_part_rigid_transform_applied": False,
            "source_interface_geometry_changed": False,
            **connector_record,
        }
        records.append(record)
        record_by_pair[
            tuple(sorted((int(parent.segment_id), int(child.segment_id))))
        ] = record

    for edge in tree:
        parent_id = int(edge["parent"])
        child_id = int(edge["child"])
        pair = tuple(sorted((parent_id, child_id)))
        record = record_by_pair[pair]
        child = by_id[child_id]
        child.metadata["contact_tree_parent_segment_id"] = parent_id
        child.metadata["contact_tree_source_adjacent"] = bool(edge["source_adjacent"])
        child.metadata["contact_tree_parent_face_index"] = int(record["parent_face_index"])
        child.metadata["contact_tree_child_face_index"] = int(record["child_face_index"])
        child.metadata["contact_tree_contact_mode"] = str(record["contact_mode"])
        child.metadata["contact_tree_connector_segment_id"] = record["connector_segment_id"]
        child.metadata["contact_tree_contact_area"] = float(record["contact_area"])
        child.metadata["contact_tree_plane_error"] = float(record["plane_error"])
        child.metadata["contact_tree_connected"] = bool(record["connected"])
        child.metadata["contact_tree_main_part_moved"] = False

    graph_connected = all(
        bool(record_by_pair[tuple(sorted((int(edge["parent"]), int(edge["child"]))))]["connected"])
        for edge in tree
    )
    compact = [
        {
            "parent_segment_id": int(record["parent_segment_id"]),
            "child_segment_id": int(record["child_segment_id"]),
            "source_adjacent": bool(record["source_adjacent"]),
            "contact_mode": str(record["contact_mode"]),
            "connector_segment_id": record["connector_segment_id"],
            "parent_face_index": int(record["parent_face_index"]),
            "child_face_index": int(record["child_face_index"]),
            "contact_area": float(record["contact_area"]),
            "connected": bool(record["connected"]),
        }
        for record in records
    ]
    for part in source_parts:
        part.metadata["contact_constraint_enabled"] = True
        part.metadata["contact_mode"] = "fixed"
        part.metadata["contact_graph_connected"] = bool(graph_connected)
        part.metadata["contact_tree"] = compact
        part.metadata["main_part_rigid_transform_applied"] = False
        part.metadata["source_interface_geometry_changed"] = False
        part.metadata["connector_part"] = False
    for connector in connectors:
        connector.metadata["contact_constraint_enabled"] = True
        connector.metadata["contact_graph_connected"] = bool(graph_connected)
        connector.metadata["contact_tree"] = compact
        connector.metadata["fallback_for_disconnected_source_component"] = True

    failed = [
        [int(record["parent_segment_id"]), int(record["child_segment_id"])]
        for record in records
        if not bool(record["connected"])
    ]
    if failed:
        raise ValueError(f"Frozen source-interface validation failed for edges: {failed}")
    return records, connectors


def _enforce_primitive_contacts(
    source_parts: list[PrimitivePart],
    contacts: dict[tuple[int, int], _SourceContact],
    config: PrimitiveFitConfig,
    frozen_interfaces: dict[tuple[int, int], _FrozenInterface] | None = None,
) -> tuple[list[dict[str, object]], list[PrimitivePart]]:
    mode = str(config.contact_mode).strip().lower()
    if mode == "move":
        records = _enforce_primitive_contacts_move(
            source_parts,
            contacts,
            config.contact_overlap_ratio,
        )
        for part in source_parts:
            part.metadata["contact_mode"] = "move"
        return records, []
    if mode == "connector":
        return _enforce_primitive_contacts_connector(source_parts, contacts, config)
    if mode == "fixed":
        return _enforce_primitive_contacts_fixed(
            source_parts, contacts, frozen_interfaces or {}, config
        )
    raise ValueError("primitive contact_mode must be 'fixed', 'connector', or 'move'")


def _resolve_primitive_overlaps(
    parts: list[PrimitivePart],
    gap_ratio: float,
    protected_pairs: set[tuple[int, int]] | None = None,
) -> int:
    if len(parts) < 2:
        return 0
    global_bounds = np.vstack((
        np.vstack([part.bounds[0] for part in parts]).min(axis=0),
        np.vstack([part.bounds[1] for part in parts]).max(axis=0),
    ))
    model_extent = max(float(np.max(global_bounds[1] - global_bounds[0])), 1e-8)
    tolerance = model_extent * 1e-6
    requested_gap = max(0.0, float(gap_ratio)) * model_extent
    adjustments = 0

    protected_pairs = set(protected_pairs or set())
    priority = sorted(parts, key=lambda item: (item.volume, item.source_surface_area), reverse=True)
    for fixed_index, fixed in enumerate(priority):
        for moving in priority[fixed_index + 1 :]:
            pair = (
                min(int(fixed.segment_id), int(moving.segment_id)),
                max(int(fixed.segment_id), int(moving.segment_id)),
            )
            if pair in protected_pairs:
                continue
            initial_volume = moving.volume
            scale_total = 1.0
            iterations = 0
            while _parts_overlap(fixed, moving, tolerance) and iterations < 8 and scale_total > 0.86:
                fixed_bounds = fixed.bounds
                moving_bounds = moving.bounds
                overlap = np.minimum(fixed_bounds[1], moving_bounds[1]) - np.maximum(
                    fixed_bounds[0], moving_bounds[0]
                )
                axis = int(np.argmin(np.maximum(overlap, 0.0)))
                relative = max(float(overlap[axis]), tolerance) / max(float(moving.size[axis]), tolerance)
                factor = float(np.clip(1.0 - 0.45 * relative, 0.90, 0.985))
                moving.scale_about(moving.center, factor)
                scale_total *= factor
                iterations += 1
                adjustments += 1

            translated = False
            if _parts_overlap(fixed, moving, tolerance):
                fixed_bounds = fixed.bounds
                moving_bounds = moving.bounds
                overlap = np.minimum(fixed_bounds[1], moving_bounds[1]) - np.maximum(
                    fixed_bounds[0], moving_bounds[0]
                )
                axis = int(np.argmin(np.maximum(overlap, 0.0)))
                direction = float(np.sign(moving.center[axis] - fixed.center[axis]))
                if direction == 0.0:
                    direction = float(np.sign(moving.source_center[axis] - fixed.source_center[axis])) or 1.0
                delta = np.zeros(3, dtype=np.float64)
                delta[axis] = direction * (max(float(overlap[axis]), 0.0) + requested_gap + tolerance)
                moving.translate(delta)
                translated = True
                adjustments += 1

            moving.metadata.setdefault("overlap_adjustments", []).append(
                {
                    "fixed_segment_id": int(fixed.segment_id),
                    "scale_factor": float(scale_total),
                    "iterations": int(iterations),
                    "translated": bool(translated),
                    "volume_retention_ratio": float(moving.volume / max(initial_volume, 1e-12)),
                }
            )

    unresolved: list[list[int]] = []
    for index, a in enumerate(parts):
        for b in parts[index + 1 :]:
            pair = (
                min(int(a.segment_id), int(b.segment_id)),
                max(int(a.segment_id), int(b.segment_id)),
            )
            if pair in protected_pairs:
                continue
            if _parts_overlap(a, b, tolerance):
                unresolved.append([int(a.segment_id), int(b.segment_id)])
    for part in parts:
        part.metadata["overlap_resolution_enabled"] = True
        part.metadata["source_adjacent_overlap_pairs_protected"] = [
            [int(pair[0]), int(pair[1])] for pair in sorted(protected_pairs)
        ]
        part.metadata["overlap_adjustment_count_total"] = int(adjustments)
        part.metadata["unresolved_overlap_pairs"] = unresolved
        part.metadata["nonoverlap_constraint_satisfied"] = not unresolved
    return adjustments


def fit_primitives_from_labels(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    config: PrimitiveFitConfig,
) -> list[PrimitivePart]:
    """Fit one automatically selected closed paper primitive per PartField label."""

    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    if len(labels) != len(mesh.faces):
        raise ValueError(
            f"Label count ({len(labels)}) does not match face count ({len(mesh.faces)}). "
            "Use PartField's normalized input mesh with its matching label file."
        )
    if config.max_faces < 6:
        raise ValueError("primitive max_faces must be >= 6")
    if config.max_sides < 3:
        raise ValueError("primitive max_sides must be >= 3")
    if config.fit_samples < 128:
        raise ValueError("primitive fit_samples must be >= 128")
    if config.contact_overlap_ratio < 0.0:
        raise ValueError("primitive contact_overlap_ratio must be >= 0")
    if str(config.contact_mode).strip().lower() not in {"fixed", "connector", "move"}:
        raise ValueError("primitive contact_mode must be fixed, connector, or move")
    if config.connector_sides < 3 or config.connector_sides > 12:
        raise ValueError("primitive connector_sides must be between 3 and 12")
    if config.connector_radius_ratio <= 0.0:
        raise ValueError("primitive connector_radius_ratio must be > 0")
    if not 0.0 <= config.connector_inset_ratio < 1.0:
        raise ValueError("primitive connector_inset_ratio must be in [0, 1)")
    if config.connector_min_length_ratio < 0.0:
        raise ValueError("primitive connector_min_length_ratio must be >= 0")
    if config.interface_max_sides < 3 or config.interface_max_sides > 32:
        raise ValueError("primitive interface_max_sides must be between 3 and 32")
    if config.interface_min_width_ratio <= 0.0:
        raise ValueError("primitive interface_min_width_ratio must be > 0")
    if config.interface_plane_tolerance_ratio <= 0.0:
        raise ValueError("primitive interface_plane_tolerance_ratio must be > 0")
    allowed = set(config.allowed_types)
    valid_types = {"box", "prism", "frustum", "cone", "ellipsoid", "convex"}
    if not allowed or not allowed.issubset(valid_types):
        raise ValueError(
            "primitive allowed_types must be selected from box, prism, frustum, cone, ellipsoid, convex"
        )

    total_area = float(np.sum(mesh.area_faces))
    cluster_faces: dict[int, np.ndarray] = {}
    for label_id in sorted(np.unique(labels).tolist()):
        face_ids = np.flatnonzero(labels == label_id)
        area = float(np.sum(mesh.area_faces[face_ids]))
        if len(face_ids) < config.min_faces:
            continue
        if total_area > 0.0 and area / total_area < config.min_area_ratio:
            continue
        cluster_faces[int(label_id)] = face_ids
    if not cluster_faces:
        raise ValueError("No primitive parts survived the segment filters")

    valid_segment_ids = set(cluster_faces)
    source_contacts = _source_label_contacts(mesh, labels, valid_segment_ids)
    source_adjacency_pairs = set(source_contacts)
    model_extent = max(float(np.max(np.ptp(np.asarray(mesh.vertices), axis=0))), 1e-8)
    fixed_mode = bool(
        config.preserve_contacts and str(config.contact_mode).strip().lower() == "fixed"
    )
    frozen_interfaces = (
        _build_frozen_interfaces(
            source_contacts,
            model_extent=model_extent,
            max_sides=config.interface_max_sides,
            min_width_ratio=config.interface_min_width_ratio,
        )
        if fixed_mode
        else {}
    )
    interfaces_by_segment: dict[int, list[_FrozenInterface]] = {
        segment_id: [] for segment_id in valid_segment_ids
    }
    for interface in frozen_interfaces.values():
        interfaces_by_segment[interface.segment_a].append(interface)
        interfaces_by_segment[interface.segment_b].append(interface)

    rng = np.random.default_rng(config.seed)
    parts: list[PrimitivePart] = []
    for label_id, face_ids in cluster_faces.items():
        print(
            f"[PrimitiveFit] segment={int(label_id)} source_faces={len(face_ids)} ",
            f"target={'auto' if config.target_faces <= 0 else config.target_faces}",
            flush=True,
        )
        part = _fit_one_cluster(
            mesh,
            face_ids,
            int(label_id),
            total_area,
            config,
            rng,
            frozen_interfaces=interfaces_by_segment.get(int(label_id), ()),
            model_extent=model_extent,
        )
        print(
            f"[PrimitiveFit] segment={int(label_id)} selected={part.primitive_type} "
            f"paper_faces={part.face_count} score={part.fit_score:.6f}",
            flush=True,
        )
        parts.append(part)

    if config.resolve_overlaps and not fixed_mode:
        # Source-adjacent labels are intentionally exempt from separation: they
        # must remain available as joints in the final paper assembly.
        _resolve_primitive_overlaps(
            parts,
            config.overlap_gap_ratio,
            protected_pairs=source_adjacency_pairs,
        )
    elif fixed_mode:
        # Scaling or translating a constrained part would alter its immutable
        # interface.  Report non-neighbour overlaps instead of corrupting joints.
        tolerance = model_extent * 1e-6
        unresolved: list[list[int]] = []
        for index, part_a in enumerate(parts):
            for part_b in parts[index + 1 :]:
                pair = tuple(sorted((int(part_a.segment_id), int(part_b.segment_id))))
                if pair in source_adjacency_pairs:
                    continue
                if _parts_overlap(part_a, part_b, tolerance):
                    unresolved.append([int(part_a.segment_id), int(part_b.segment_id)])
        for part in parts:
            part.metadata["overlap_resolution_enabled"] = False
            part.metadata["overlap_resolution_skipped_for_frozen_interfaces"] = True
            part.metadata["unresolved_overlap_pairs"] = unresolved
            part.metadata["nonoverlap_constraint_satisfied"] = not unresolved
    else:
        for part in parts:
            part.metadata["overlap_resolution_enabled"] = False
            part.metadata["nonoverlap_constraint_satisfied"] = None

    source_parts = parts
    if config.preserve_contacts:
        contact_records, connector_parts = _enforce_primitive_contacts(
            source_parts,
            source_contacts,
            config,
            frozen_interfaces=frozen_interfaces,
        )
    else:
        contact_records = []
        connector_parts = []
        for part in source_parts:
            part.metadata["contact_constraint_enabled"] = False
            part.metadata["contact_graph_connected"] = None

    for part in source_parts:
        part.metadata["source_adjacent_segment_ids"] = sorted(
            other
            for pair in source_adjacency_pairs
            if int(part.segment_id) in pair
            for other in pair
            if other != int(part.segment_id)
        )
        part.metadata["contact_tree_edge_count"] = int(len(contact_records))

    # PrimitivePart intentionally exposes the same naming properties consumed by
    # the existing deterministic naming heuristic.
    assign_part_names(
        source_parts,
        category=config.category,
        forward_axis=config.forward_axis,
    )  # type: ignore[arg-type]
    source_by_id = {int(part.segment_id): part for part in source_parts}
    for connector in connector_parts:
        parent_id = int(connector.metadata["parent_segment_id"])
        child_id = int(connector.metadata["child_segment_id"])
        parent_name = source_by_id[parent_id].name
        child_name = source_by_id[child_id].name
        connector.name = f"joint_{parent_name}_{child_name}"
        connector.metadata["parent_part_name"] = parent_name
        connector.metadata["child_part_name"] = child_name
        connector.metadata["source_adjacent_segment_ids"] = [parent_id, child_id]
        connector.metadata["contact_tree_edge_count"] = int(len(contact_records))

    parts = source_parts + connector_parts
    for part in parts:
        part.metadata["selected_name"] = part.name
    return parts


def parse_primitive_types(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "auto", "all"}:
            return ("box", "prism", "frustum", "cone", "ellipsoid", "convex")
        values = [item.strip().lower() for item in text.split(",") if item.strip()]
    else:
        values = [str(item).strip().lower() for item in value if str(item).strip()]
    valid = {"box", "prism", "frustum", "cone", "ellipsoid", "convex"}
    invalid = sorted(set(values) - valid)
    if invalid:
        raise ValueError(f"Unsupported primitive types: {', '.join(invalid)}")
    if not values:
        raise ValueError("At least one primitive type is required")
    return tuple(dict.fromkeys(values))
