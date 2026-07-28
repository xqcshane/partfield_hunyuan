from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

from partfield_mc.exporters import _face_surface_patches, write_glb, write_obj
from partfield_mc.models import CuboidPart
from partfield_mc.texture import FACE_NAMES


def part(name, segment_id, center, size):
    transform = np.eye(4)
    transform[:3, 3] = np.asarray(center, dtype=float)
    return CuboidPart(name, segment_id, np.asarray(size, dtype=float), transform, 12, 1.0, np.asarray(center, dtype=float))


def test_surface_obj_has_six_quads_and_eight_shared_vertices_per_cluster(tmp_path: Path):
    parts = [part("a", 0, [-1.0, 0, 0], [2, 2, 2]), part("b", 1, [1.0, 0, 0], [2, 2, 2])]
    uv = {(i, face): (0.0, 0.0, 1.0, 1.0) for i in range(2) for face in FACE_NAMES}
    out = tmp_path / "surface.obj"
    write_obj(parts, uv, out, "mc_model.mtl", surface_only=True)
    lines = out.read_text().splitlines()
    faces = [line.split()[1:] for line in lines if line.startswith("f ")]
    vertices = [line for line in lines if line.startswith("v ")]
    assert len(vertices) == 16
    assert len(faces) == 12
    assert all(len(face) == 4 for face in faces)
    assert "triangles=0" in out.read_text()


def test_single_box_surface_obj_has_shared_topology(tmp_path: Path):
    parts = [part("whole_model", 0, [0, 0, 0], [2, 4, 6])]
    uv = {(0, face): (0.0, 0.0, 1.0, 1.0) for face in FACE_NAMES}
    out = tmp_path / "single_box.obj"
    write_obj(parts, uv, out, "mc_model.mtl", surface_only=True)

    lines = out.read_text().splitlines()
    vertices = [line for line in lines if line.startswith("v ")]
    faces = [line.split()[1:] for line in lines if line.startswith("f ")]
    assert len(vertices) == 8
    assert len(faces) == 6
    assert all(len(face) == 4 for face in faces)

    edge_counts = {}
    for face in faces:
        ids = [int(token.split("/")[0]) for token in face]
        for a, b in zip(ids, ids[1:] + ids[:1]):
            edge = tuple(sorted((a, b)))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    assert len(edge_counts) == 12
    assert set(edge_counts.values()) == {2}


def test_partial_contact_region_is_split_and_only_contact_patch_is_untextured(tmp_path: Path):
    # The smaller box touches only one quarter of the larger box's +X face.
    parts = [
        part("large", 0, [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]),
        part("small", 1, [1.5, 0.5, 0.0], [1.0, 1.0, 1.0]),
    ]
    uv = {(i, face): (0.0, 0.0, 1.0, 1.0) for i in range(2) for face in FACE_NAMES}

    large_patches = _face_surface_patches(0, parts[0], "+X", parts, uv)
    hidden_area = sum(patch.area for patch in large_patches if not patch.textured)
    textured_area = sum(patch.area for patch in large_patches if patch.textured)
    assert np.isclose(hidden_area, 1.0)
    assert np.isclose(textured_area, 3.0)
    assert any(patch.textured for patch in large_patches)
    assert any(not patch.textured for patch in large_patches)

    obj_path = tmp_path / "partial_contact.obj"
    write_obj(parts, uv, obj_path, "mc_model.mtl", surface_only=True)
    face_lines = [line for line in obj_path.read_text().splitlines() if line.startswith("f ")]
    untextured_faces = [line for line in face_lines if "/" not in line]
    textured_faces = [line for line in face_lines if "/" in line]
    assert len(untextured_faces) == 2  # one contact patch on each touching cuboid
    assert len(textured_faces) > 10

    glb_path = tmp_path / "partial_contact.glb"
    write_glb(parts, uv, Image.new("RGB", (4, 4), "white"), glb_path, surface_only=True)
    scene = trimesh.load(glb_path, force="scene")
    names = set(scene.geometry.keys())
    assert "large__textured" in names
    assert "large__contact_untextured" in names
    assert "small__textured" in names
    assert "small__contact_untextured" in names
    assert scene.geometry["large__textured"].visual.kind == "texture"
    assert scene.geometry["large__contact_untextured"].visual.kind != "texture"
