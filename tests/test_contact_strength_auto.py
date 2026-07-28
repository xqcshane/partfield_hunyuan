import numpy as np
import trimesh

from partfield_mc.primitive_fit import (
    PrimitiveFitConfig,
    PrimitivePart,
    _ContactStrength,
    _FrozenInterface,
    _SourceContact,
    _classify_contact_strengths,
    _enforce_primitive_contacts_auto,
    _fit_constrained_surface_cluster,
)


def _box_part(segment_id: int, center) -> PrimitivePart:
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    mesh.apply_translation(np.asarray(center, dtype=float))
    return PrimitivePart(
        name=f"part_{segment_id}",
        segment_id=segment_id,
        vertices=np.asarray(mesh.vertices, dtype=float),
        polygons=[list(map(int, face)) for face in np.asarray(mesh.faces)],
        source_face_count=len(mesh.faces),
        source_surface_area=float(mesh.area),
        source_center=np.asarray(center, dtype=float),
        primitive_type="box",
        target_face_count=12,
        fit_score=0.0,
    )


def _contact(edge_count: int, point_count: int, boundary_length: float) -> _SourceContact:
    points = np.column_stack(
        (
            np.zeros(point_count),
            np.linspace(-0.1, 0.1, point_count),
            np.zeros(point_count),
        )
    )
    return _SourceContact(
        segment_a=1,
        segment_b=2,
        anchor=np.zeros(3),
        boundary_length=boundary_length,
        edge_count=edge_count,
        boundary_points=points,
        direction_a_to_b=np.array([1.0, 0.0, 0.0]),
        interface_normal=np.array([1.0, 0.0, 0.0]),
        interface_axis_u=np.array([0.0, 1.0, 0.0]),
        interface_axis_v=np.array([0.0, 0.0, 1.0]),
    )


def _interface(area: float) -> _FrozenInterface:
    half = np.sqrt(area) * 0.5
    polygon = np.array(
        [
            [0.0, -half, -half],
            [0.0, half, -half],
            [0.0, half, half],
            [0.0, -half, half],
        ]
    )
    return _FrozenInterface(
        segment_a=1,
        segment_b=2,
        anchor=np.zeros(3),
        normal_a_to_b=np.array([1.0, 0.0, 0.0]),
        axis_u=np.array([0.0, 1.0, 0.0]),
        axis_v=np.array([0.0, 0.0, 1.0]),
        polygon_2d=polygon[:, 1:],
        polygon_3d=polygon,
        area=area,
        source_boundary_length=1.0,
        source_boundary_edge_count=20,
        fallback_rectangle=False,
    )


def test_contact_strength_forces_short_contact_weak_and_large_contact_strong():
    mesh = trimesh.util.concatenate((trimesh.creation.box(), trimesh.creation.box()))
    cluster_faces = {
        1: np.arange(0, 12, dtype=np.int64),
        2: np.arange(12, 24, dtype=np.int64),
    }
    config = PrimitiveFitConfig(
        contact_weak_threshold=0.20,
        contact_strong_threshold=0.55,
        contact_min_edge_count=6,
    )

    weak = _classify_contact_strengths(
        mesh,
        cluster_faces,
        {(1, 2): _contact(edge_count=2, point_count=3, boundary_length=0.02)},
        {(1, 2): _interface(area=0.001)},
        config,
    )[(1, 2)]
    assert weak.classification == "weak"
    assert weak.forced_weak_by_edge_count is True

    strong = _classify_contact_strengths(
        mesh,
        cluster_faces,
        {(1, 2): _contact(edge_count=40, point_count=40, boundary_length=2.0)},
        {(1, 2): _interface(area=2.0)},
        config,
    )[(1, 2)]
    assert strong.classification == "strong"
    assert strong.score >= config.contact_strong_threshold


def test_auto_mode_allows_weak_overlap_to_separate_without_connector():
    part_a = _box_part(1, (0.0, 0.0, 0.0))
    part_b = _box_part(2, (0.4, 0.0, 0.0))
    contact = _contact(edge_count=2, point_count=3, boundary_length=0.02)
    strength = _ContactStrength(
        segment_a=1,
        segment_b=2,
        classification="weak",
        score=0.05,
        interface_area_ratio=0.001,
        seam_length_ratio=0.01,
        edge_count_ratio=0.1,
        point_count_ratio=0.1,
        edge_count=2,
        unique_point_count=3,
        forced_weak_by_edge_count=True,
    )
    records, connectors = _enforce_primitive_contacts_auto(
        [part_a, part_b],
        {(1, 2): contact},
        {},
        {(1, 2): strength},
        PrimitiveFitConfig(contact_mode="auto"),
    )
    assert connectors == []
    assert records[0]["contact_mode"] == "weak_contact_separated"
    assert records[0]["required_connection"] is False
    assert records[0]["weak_overlap_separation"]["resolved"] is True


def test_constrained_surface_never_reaches_texture_with_more_than_hard_cap():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    config = PrimitiveFitConfig(
        max_faces=48,
        surface_hard_max_faces=96,
        fit_samples=128,
        surface_search_steps=8,
        part_mode="auto",
    )
    part = _fit_constrained_surface_cluster(
        mesh,
        np.arange(len(mesh.faces), dtype=np.int64),
        1,
        float(mesh.area),
        config,
        np.random.default_rng(123),
        frozen_interfaces=(),
        model_extent=2.0,
    )
    assert part.face_count <= 96
    assert part.metadata["surface_hard_face_cap_satisfied"] is True
