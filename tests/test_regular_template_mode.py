import numpy as np
import trimesh

from partfield_mc.primitive_fit import (
    PrimitiveFitConfig,
    PrimitivePart,
    _SourceContact,
    _augment_contacts_with_spatial_proximity,
    _select_contact_face_pair,
    fit_primitives_from_labels,
)


def _prism_part(segment_id: int, z0: float, z1: float, radius: float = 1.0) -> PrimitivePart:
    sides = 8
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    low = np.column_stack((radius * np.cos(angles), radius * np.sin(angles), np.full(sides, z0)))
    high = np.column_stack((radius * np.cos(angles), radius * np.sin(angles), np.full(sides, z1)))
    vertices = np.vstack((low, high))
    polygons = [list(reversed(range(sides))), list(range(sides, 2 * sides))]
    for i in range(sides):
        j = (i + 1) % sides
        polygons.append([i, j, sides + j, sides + i])
    return PrimitivePart(
        name=f"part_{segment_id}",
        segment_id=segment_id,
        vertices=vertices,
        polygons=polygons,
        source_face_count=16,
        source_surface_area=10.0,
        source_center=vertices.mean(axis=0),
        primitive_type="prism_8",
        target_face_count=len(polygons),
        fit_score=0.0,
        metadata={
            "selected_candidate_metadata": {
                "regular_attachment_face_indices": [0, 1],
                "regular_attachment_policy": "axial_caps",
            }
        },
    )


def test_regular_face_selection_uses_axial_caps():
    parent = _prism_part(1, -1.0, 0.0)
    child = _prism_part(2, 0.1, 0.5, radius=0.7)
    contact = {
        "anchor": np.array([0.0, 0.0, 0.05]),
        "direction_a_to_b": np.array([0.0, 0.0, 1.0]),
        "interface_axis_u": np.array([1.0, 0.0, 0.0]),
    }
    selection = _select_contact_face_pair(
        parent,
        child,
        contact,
        model_extent=2.0,
        connector_radius_ratio=0.02,
        connector_inset_ratio=0.25,
        regular_templates=True,
    )
    assert selection["parent_face_index"] in {0, 1}
    assert selection["child_face_index"] in {0, 1}
    assert selection["regular_attachment_only"] is True
    assert selection["regular_attachment_fallback"] is False


def test_regular_mode_never_uses_constrained_surface_or_frozen_interface_cutting():
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    centers = np.asarray(mesh.triangles_center)
    labels = (centers[:, 0] > 0.0).astype(np.int64)
    parts = fit_primitives_from_labels(
        mesh,
        labels,
        PrimitiveFitConfig(
            template_mode="regular",
            part_mode="auto",
            contact_mode="auto",
            contact_weak_threshold=0.05,
            contact_strong_threshold=0.20,
            contact_min_edge_count=2,
            allowed_types=("ellipsoid", "frustum"),
            max_faces=32,
            max_sides=16,
            fit_samples=200,
            resolve_overlaps=False,
        ),
    )
    source_parts = [part for part in parts if not part.metadata.get("connector_part")]
    assert source_parts
    assert all(part.primitive_type != "constrained_surface" for part in source_parts)
    assert all(part.metadata["primitive_template_mode"] == "regular" for part in source_parts)
    assert all(part.metadata["canonical_template_topology_locked"] is True for part in source_parts)
    assert all(part.metadata["local_interface_deformation_applied"] is False for part in source_parts)
    assert all(not part.metadata.get("frozen_interface_face_indices") for part in source_parts)


def test_regular_connector_patch_is_bounded_by_selected_face_area():
    parent = _prism_part(1, -1.0, 0.0)
    child = _prism_part(2, 0.15, 0.55, radius=0.7)
    contact = _SourceContact(
        segment_a=1,
        segment_b=2,
        anchor=np.array([0.0, 0.0, 0.075]),
        boundary_length=20.0,
        edge_count=32,
        boundary_points=np.array([[0.0, 0.0, 0.075], [0.1, 0.0, 0.075], [0.0, 0.1, 0.075]]),
        direction_a_to_b=np.array([0.0, 0.0, 1.0]),
        interface_normal=np.array([0.0, 0.0, 1.0]),
        interface_axis_u=np.array([1.0, 0.0, 0.0]),
        interface_axis_v=np.array([0.0, 1.0, 0.0]),
    )
    # Use public contact enforcement through a tiny source mesh with a ring seam.
    # The oversized source boundary must not turn into a wide visible flange.
    from partfield_mc.primitive_fit import _enforce_primitive_contacts_connector

    config = PrimitiveFitConfig(
        template_mode="regular",
        contact_mode="connector",
        connector_radius_ratio=0.08,
        regular_connector_max_face_area_ratio=0.05,
    )
    records, connectors = _enforce_primitive_contacts_connector(
        [parent, child], {(1, 2): contact}, config
    )
    assert len(connectors) == 1
    connector = connectors[0]
    assert connector.metadata["regular_template_joint"] is True
    patch_area = float(records[0]["contact_area"])
    selected_face_area = min(
        float(connector.metadata["selected_parent_face_area"]),
        float(connector.metadata["selected_child_face_area"]),
    )
    assert patch_area <= 0.05 * selected_face_area + 1e-9


def test_spatial_proximity_recovers_unwelded_cap_contact():
    body = trimesh.creation.cylinder(radius=1.0, height=2.0, sections=32)
    cap = trimesh.creation.cylinder(radius=0.75, height=0.25, sections=32)
    cap.apply_translation([0.0, 0.0, 1.14])  # 0.015 gap above body top at z=1.0
    mesh = trimesh.util.concatenate((body, cap))
    cluster_faces = {
        1: np.arange(len(body.faces), dtype=np.int64),
        2: np.arange(len(body.faces), len(body.faces) + len(cap.faces), dtype=np.int64),
    }
    contacts = _augment_contacts_with_spatial_proximity(
        mesh,
        cluster_faces,
        {},
        PrimitiveFitConfig(
            contact_proximity_ratio=0.06,
            contact_proximity_min_points=8,
            contact_proximity_min_coverage=0.01,
        ),
        model_extent=2.0,
    )
    assert (1, 2) in contacts
    contact = contacts[(1, 2)]
    assert contact.source_kind == "spatial_proximity"
    assert contact.proximity_point_count >= 8
    assert contact.proximity_distance <= 0.12
