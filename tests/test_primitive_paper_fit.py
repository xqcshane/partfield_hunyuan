from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from partfield_mc.mesh_io import normalize_meshes, partfield_normalization
from partfield_mc.pipeline import PipelineConfig, run_postprocess
from partfield_mc.primitive_exporters import write_primitive_obj
from partfield_mc.primitive_fit import PrimitiveFitConfig, PrimitivePart, fit_primitives_from_labels


def _edge_counts(polygons: list[list[int]]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for polygon in polygons:
        for a, b in zip(polygon, polygon[1:] + polygon[:1]):
            edge = tuple(sorted((int(a), int(b))))
            counts[edge] = counts.get(edge, 0) + 1
    return counts


def test_auto_primitive_fit_selects_box_and_cone() -> None:
    box = trimesh.creation.box(extents=[2.0, 1.0, 1.0])
    cone = trimesh.creation.cone(radius=0.7, height=1.8, sections=12)
    cone.apply_translation([3.0, 0.0, 0.0])
    mesh = trimesh.util.concatenate((box, cone))
    labels = np.concatenate(
        (
            np.zeros(len(box.faces), dtype=np.int64),
            np.ones(len(cone.faces), dtype=np.int64),
        )
    )
    parts = fit_primitives_from_labels(
        mesh,
        labels,
        PrimitiveFitConfig(
            fit_samples=200,
            max_faces=16,
            max_sides=12,
            allowed_types=("box", "cone"),
            resolve_overlaps=False,
        ),
    )
    source_parts = [part for part in parts if not part.metadata.get("connector_part")]
    connectors = [part for part in parts if part.metadata.get("connector_part")]
    assert [part.segment_id for part in source_parts] == [0, 1]
    assert source_parts[0].primitive_type == "box"
    assert source_parts[1].primitive_type.startswith("cone_")
    assert len(connectors) == 1
    assert connectors[0].primitive_type == "connector_frustum_4"
    for part in parts:
        assert set(_edge_counts(part.polygons).values()) == {2}
        if not part.metadata.get("connector_part"):
            assert part.metadata["paper_safe"] is True
            assert part.metadata["top_candidates"]


def test_primitive_obj_preserves_shared_closed_shell_topology(tmp_path: Path) -> None:
    vertices = np.asarray(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, -1.0, 2.0],
            [1.0, -1.0, 2.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    polygons = [
        [0, 2, 1],
        [3, 4, 5],
        [0, 1, 4, 3],
        [1, 2, 5, 4],
        [2, 0, 3, 5],
    ]
    part = PrimitivePart(
        name="tri_prism",
        segment_id=0,
        vertices=vertices,
        polygons=polygons,
        source_face_count=20,
        source_surface_area=10.0,
        source_center=vertices.mean(axis=0),
        primitive_type="prism_3",
        target_face_count=5,
        fit_score=0.0,
    )
    uv = {
        (0, face_index): np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]][: len(face)],
            dtype=np.float64,
        )
        for face_index, face in enumerate(polygons)
    }
    path = tmp_path / "paper_model.obj"
    records = write_primitive_obj([part], uv, path, "paper_model.mtl")
    lines = path.read_text().splitlines()
    assert len([line for line in lines if line.startswith("v ")]) == 6
    assert len([line for line in lines if line.startswith("f ")]) == 5
    assert records[0]["closed"] is True
    assert records[0]["edge_count"] == 9
    assert set(_edge_counts(polygons).values()) == {2}


def test_pipeline_primitive_surface_exports_paper_model(tmp_path: Path) -> None:
    box = trimesh.creation.box(extents=[2.0, 1.0, 1.0])
    cone = trimesh.creation.cone(radius=0.5, height=1.5, sections=12)
    cone.apply_translation([2.0, 0.0, 0.0])
    sources = [box, cone]
    source_path = tmp_path / "source.glb"
    source_path.write_bytes(trimesh.Scene(sources).export(file_type="glb"))

    center, scale = partfield_normalization(sources)
    normalized = normalize_meshes(sources, center, scale)
    segmented = trimesh.util.concatenate(normalized)
    normalized_path = tmp_path / "normalized.ply"
    segmented.export(normalized_path)
    labels = np.concatenate(
        (
            np.zeros(len(normalized[0].faces), dtype=np.int64),
            np.ones(len(normalized[1].faces), dtype=np.int64),
        )
    )
    labels_path = tmp_path / "labels.npy"
    np.save(labels_path, labels)

    output = tmp_path / "out"
    result = run_postprocess(
        source_path,
        output,
        normalized_path,
        labels_path,
        PipelineConfig(
            partfield_repo=tmp_path,
            checkpoint=tmp_path / "unused.ckpt",
            clusters=2,
            fit_mode="primitive",
            obj_mode="surface",
            primitive_types="box,cone",
            primitive_fit_samples=200,
            primitive_max_faces=16,
            primitive_max_sides=12,
            primitive_resolve_overlaps=False,
            face_resolution=4,
            surface_samples=1000,
            padding=0,
        ),
    )
    for filename in (
        "mc_model.glb",
        "mc_model.obj",
        "mc_texture.png",
        "parts.json",
        "paper_model.glb",
        "paper_model.obj",
        "paper_model.mtl",
        "paper_model_texture.png",
        "paper_model_parts.json",
    ):
        assert (output / filename).exists()
    assert Path(result.paper_model_obj).name == "paper_model.obj"
    data = json.loads((output / "parts.json").read_text())
    assert data["conversion"]["fit_mode"] == "primitive"
    assert data["conversion"]["paper_safe_closed_shells"] is True
    source_types = {
        part["primitive_type"].split("_")[0]
        for part in data["parts"]
        if not part["metadata"].get("connector_part")
    }
    assert source_types == {"box", "cone"}
    assert data["conversion"]["primitive_contact_mode"] == "fixed"
    paper_data = json.loads((output / "paper_model_parts.json").read_text())
    assert paper_data["source_part_count"] == 2
    assert paper_data["connector_count"] == 1


def test_primitive_contact_fit_restores_source_adjacency() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    labels = (mesh.triangles_center[:, 0] >= 0.0).astype(np.int64)
    parts = fit_primitives_from_labels(
        mesh,
        labels,
        PrimitiveFitConfig(
            fit_samples=128,
            max_faces=8,
            max_sides=4,
            allowed_types=("box",),
            resolve_overlaps=True,
            preserve_contacts=True,
            contact_overlap_ratio=0.0,
        ),
    )

    source_parts = [part for part in parts if not part.metadata.get("connector_part")]
    connectors = [part for part in parts if part.metadata.get("connector_part")]
    assert len(source_parts) == 2
    assert source_parts[0].metadata["source_adjacent_segment_ids"] == [1]
    assert source_parts[1].metadata["source_adjacent_segment_ids"] == [0]
    assert all(part.metadata["contact_graph_connected"] is True for part in parts)
    child = next(
        part for part in source_parts if "contact_tree_parent_segment_id" in part.metadata
    )
    assert child.metadata["contact_tree_source_adjacent"] is True
    assert child.metadata["contact_tree_connected"] is True
    assert child.metadata["contact_tree_contact_area"] > 0.0
    assert child.metadata["contact_tree_plane_error"] < 1e-5
    assert child.metadata["contact_tree_main_part_moved"] is False
    if child.metadata["contact_tree_contact_mode"] == "connector_patch":
        assert len(connectors) == 1


def test_primitive_contact_fit_connects_disconnected_source_components() -> None:
    first = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    second = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    second.apply_translation([3.0, 0.0, 0.0])
    mesh = trimesh.util.concatenate((first, second))
    labels = np.concatenate(
        (
            np.zeros(len(first.faces), dtype=np.int64),
            np.ones(len(second.faces), dtype=np.int64),
        )
    )
    parts = fit_primitives_from_labels(
        mesh,
        labels,
        PrimitiveFitConfig(
            fit_samples=128,
            max_faces=8,
            max_sides=4,
            allowed_types=("box",),
            resolve_overlaps=False,
            preserve_contacts=True,
        ),
    )

    source_parts = [part for part in parts if not part.metadata.get("connector_part")]
    connectors = [part for part in parts if part.metadata.get("connector_part")]
    child = next(
        part for part in source_parts if "contact_tree_parent_segment_id" in part.metadata
    )
    assert child.metadata["contact_tree_source_adjacent"] is False
    assert all(part.metadata["contact_graph_connected"] is True for part in parts)
    assert child.metadata["contact_tree_contact_area"] > 0.0
    assert child.metadata["contact_tree_plane_error"] < 1e-5
    assert child.metadata["contact_tree_main_part_moved"] is False
    assert len(connectors) == 1
    assert connectors[0].metadata["parent_segment_id"] in {0, 1}
    assert connectors[0].metadata["child_segment_id"] in {0, 1}


def test_connector_contact_mode_does_not_move_fitted_source_parts() -> None:
    first = trimesh.creation.box(extents=[1.3, 1.0, 0.8])
    second = trimesh.creation.cone(radius=0.45, height=1.2, sections=8)
    second.apply_translation([1.7, 0.25, 0.15])
    mesh = trimesh.util.concatenate((first, second))
    labels = np.concatenate(
        (
            np.zeros(len(first.faces), dtype=np.int64),
            np.ones(len(second.faces), dtype=np.int64),
        )
    )
    base_config = dict(
        fit_samples=128,
        max_faces=12,
        max_sides=8,
        allowed_types=("box", "cone"),
        resolve_overlaps=False,
    )
    without_contacts = fit_primitives_from_labels(
        mesh,
        labels,
        PrimitiveFitConfig(**base_config, preserve_contacts=False),
    )
    with_contacts = fit_primitives_from_labels(
        mesh,
        labels,
        PrimitiveFitConfig(**base_config, preserve_contacts=True, contact_mode="connector"),
    )
    baseline = {part.segment_id: part for part in without_contacts}
    fitted = {
        part.segment_id: part
        for part in with_contacts
        if not part.metadata.get("connector_part")
    }
    assert baseline.keys() == fitted.keys()
    for segment_id in baseline:
        assert np.allclose(baseline[segment_id].vertices, fitted[segment_id].vertices)
        assert fitted[segment_id].metadata["main_part_rigid_transform_applied"] is False


def test_connector_patch_is_closed_and_has_positive_face_area() -> None:
    first = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    second = trimesh.creation.box(extents=[0.8, 0.7, 0.9])
    second.apply_translation([1.8, 0.35, -0.2])
    mesh = trimesh.util.concatenate((first, second))
    labels = np.concatenate(
        (
            np.zeros(len(first.faces), dtype=np.int64),
            np.ones(len(second.faces), dtype=np.int64),
        )
    )
    parts = fit_primitives_from_labels(
        mesh,
        labels,
        PrimitiveFitConfig(
            fit_samples=128,
            max_faces=8,
            max_sides=4,
            allowed_types=("box",),
            resolve_overlaps=False,
            preserve_contacts=True,
            contact_mode="connector",
            connector_sides=4,
        ),
    )
    connector = next(part for part in parts if part.metadata.get("connector_part"))
    assert set(_edge_counts(connector.polygons).values()) == {2}
    assert connector.volume > 0.0
    assert connector.metadata["parent_contact_area"] > 0.0
    assert connector.metadata["child_contact_area"] > 0.0
    assert connector.face_count == 10
    assert connector.metadata["side_faces_triangulated"] is True


def test_fixed_contact_mode_freezes_shared_source_interface() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    labels = (mesh.triangles_center[:, 0] >= 0.0).astype(np.int64)
    parts = fit_primitives_from_labels(
        mesh,
        labels,
        PrimitiveFitConfig(
            fit_samples=128,
            max_faces=12,
            max_sides=8,
            allowed_types=("box", "ellipsoid"),
            resolve_overlaps=True,
            preserve_contacts=True,
            contact_mode="fixed",
            interface_max_sides=8,
        ),
    )
    source_parts = [part for part in parts if not part.metadata.get("connector_part")]
    connectors = [part for part in parts if part.metadata.get("connector_part")]
    assert len(source_parts) == 2
    assert connectors == []
    by_id = {part.segment_id: part for part in source_parts}
    first, second = by_id[0], by_id[1]
    face_0 = int(first.metadata["frozen_interface_face_indices"]["1"])
    face_1 = int(second.metadata["frozen_interface_face_indices"]["0"])
    points_0 = first.vertices[np.asarray(first.polygons[face_0], dtype=np.int64)]
    points_1 = second.vertices[np.asarray(second.polygons[face_1], dtype=np.int64)]
    assert len(points_0) == len(points_1)
    assert np.max(cKDTree(points_0).query(points_1, k=1)[0]) < 1e-7
    assert np.max(cKDTree(points_1).query(points_0, k=1)[0]) < 1e-7
    normal_0 = np.cross(points_0[1] - points_0[0], points_0[2] - points_0[0])
    normal_1 = np.cross(points_1[1] - points_1[0], points_1[2] - points_1[0])
    normal_0 /= np.linalg.norm(normal_0)
    normal_1 /= np.linalg.norm(normal_1)
    assert float(np.dot(normal_0, normal_1)) < -0.999
    for part in source_parts:
        assert part.metadata["contact_mode"] == "fixed"
        assert part.metadata["main_part_rigid_transform_applied"] is False
        assert part.metadata["source_interface_geometry_changed"] is False
        assert part.metadata["fitting_strategy"] == "fit_outer_surface_around_fixed_source_interfaces"
        assert set(_edge_counts(part.polygons).values()) == {2}


def test_fixed_interfaces_fall_back_to_local_patch_when_convex_constraints_conflict() -> None:
    from partfield_mc.primitive_fit import (
        _FrozenInterface,
        _box_candidate,
        _constrain_candidate_to_interfaces,
    )

    points = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    candidate = _box_candidate(points)
    interface_x = _FrozenInterface(
        segment_a=0,
        segment_b=1,
        anchor=np.asarray([0.5, 0.0, 0.0]),
        normal_a_to_b=np.asarray([1.0, 0.0, 0.0]),
        axis_u=np.asarray([0.0, 1.0, 0.0]),
        axis_v=np.asarray([0.0, 0.0, 1.0]),
        polygon_2d=np.asarray([[-0.3, -0.3], [0.3, -0.3], [0.3, 0.3], [-0.3, 0.3]]),
        polygon_3d=np.asarray(
            [[0.5, -0.3, -0.3], [0.5, 0.3, -0.3], [0.5, 0.3, 0.3], [0.5, -0.3, 0.3]]
        ),
        area=0.36,
        source_boundary_length=2.4,
        source_boundary_edge_count=4,
        fallback_rectangle=False,
    )
    # This exact y-interface protrudes beyond x=0.5, so the two interface
    # polygons cannot simultaneously be facets of a single convex hull.
    interface_y = _FrozenInterface(
        segment_a=0,
        segment_b=2,
        anchor=np.asarray([0.8, 0.5, 0.0]),
        normal_a_to_b=np.asarray([0.0, 1.0, 0.0]),
        axis_u=np.asarray([1.0, 0.0, 0.0]),
        axis_v=np.asarray([0.0, 0.0, 1.0]),
        polygon_2d=np.asarray([[-0.15, -0.25], [0.15, -0.25], [0.15, 0.25], [-0.15, 0.25]]),
        polygon_3d=np.asarray(
            [[0.65, 0.5, -0.25], [0.95, 0.5, -0.25], [0.95, 0.5, 0.25], [0.65, 0.5, 0.25]]
        ),
        area=0.15,
        source_boundary_length=1.6,
        source_boundary_edge_count=4,
        fallback_rectangle=False,
    )
    constrained = _constrain_candidate_to_interfaces(
        candidate,
        segment_id=0,
        source_center=np.zeros(3),
        interfaces=[interface_x, interface_y],
        model_extent=2.0,
        plane_tolerance_ratio=1e-6,
    )
    assert constrained.metadata["fixed_interface_solver"] == "nonconvex_local_adapter"
    assert set(_edge_counts(constrained.polygons).values()) == {2}
    for neighbor, expected in (("1", interface_x.polygon_3d), ("2", interface_y.polygon_3d)):
        face_index = int(constrained.metadata["frozen_interface_face_indices"][neighbor])
        actual = constrained.vertices[np.asarray(constrained.polygons[face_index], dtype=np.int64)]
        assert np.max(cKDTree(actual).query(expected, k=1)[0]) < 1e-8
        assert np.max(cKDTree(expected).query(actual, k=1)[0]) < 1e-8
