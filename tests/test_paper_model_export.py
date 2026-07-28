from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

from partfield_mc.exporters import (
    build_paper_model_texture,
    write_paper_model_glb,
    write_paper_model_obj,
)
from partfield_mc.models import CuboidPart
from partfield_mc.texture import FACE_NAMES


def part(name: str, segment_id: int, center, size) -> CuboidPart:
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


def parse_obj(path: Path):
    vertices = []
    faces = []
    object_names = []
    groups = []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            vertices.append(tuple(float(v) for v in line.split()[1:4]))
        elif line.startswith("f "):
            faces.append([int(token.split("/")[0]) - 1 for token in line.split()[1:]])
        elif line.startswith("o "):
            object_names.append(line.split(maxsplit=1)[1])
        elif line.startswith("g "):
            groups.append(line.split(maxsplit=1)[1])
    return np.asarray(vertices), faces, object_names, groups


def connected_face_components(faces: list[list[int]]) -> list[list[int]]:
    vertex_to_faces: dict[int, list[int]] = {}
    for face_index, face in enumerate(faces):
        for vertex in face:
            vertex_to_faces.setdefault(vertex, []).append(face_index)
    neighbours = [set() for _ in faces]
    for indices in vertex_to_faces.values():
        for a in indices:
            neighbours[a].update(indices)
    remaining = set(range(len(faces)))
    components: list[list[int]] = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        component = [seed]
        while stack:
            current = stack.pop()
            for neighbour in neighbours[current]:
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    stack.append(neighbour)
                    component.append(neighbour)
        components.append(component)
    return components


def test_paper_obj_is_one_object_with_independent_closed_shells(tmp_path: Path) -> None:
    parts = [
        part("body", 0, [-1.0, 0.0, 0.0], [2.0, 2.0, 2.0]),
        part("head", 1, [1.0, 0.5, 0.0], [2.0, 1.0, 1.0]),
    ]
    uv = {(i, face): (0.0, 0.0, 1.0, 1.0) for i in range(2) for face in FACE_NAMES}
    path = tmp_path / "paper_model.obj"
    records = write_paper_model_obj(parts, uv, path, "paper_model.mtl")

    vertices, faces, object_names, groups = parse_obj(path)
    assert object_names == ["paper_model"]
    assert groups == []
    assert len(vertices) == 16
    assert len(faces) == 12
    assert all(len(face) == 4 for face in faces)
    assert len(records) == 2

    components = connected_face_components(faces)
    assert sorted(len(component) for component in components) == [6, 6]
    for component in components:
        component_faces = [faces[index] for index in component]
        vertex_ids = sorted({vertex for face in component_faces for vertex in face})
        assert len(vertex_ids) == 8
        edge_counts: dict[tuple[int, int], int] = {}
        for face in component_faces:
            for a, b in zip(face, face[1:] + face[:1]):
                edge = tuple(sorted((a, b)))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        assert len(edge_counts) == 12
        assert set(edge_counts.values()) == {2}


def test_paper_texture_masks_partial_contact_without_face_subdivision(tmp_path: Path) -> None:
    # small touches the upper half of large's +X face.
    parts = [
        part("large", 0, [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]),
        part("small", 1, [1.5, 0.5, 0.0], [1.0, 1.0, 1.0]),
    ]
    width, height = 120, 20
    image = Image.new("RGB", (width, height), (40, 80, 120))
    uv = {}
    # Six 20-pixel columns per part; each part occupies a 10-pixel row.
    for part_index in range(2):
        row_y0 = part_index * 10
        row_y1 = row_y0 + 10
        for face_index, face in enumerate(FACE_NAMES):
            x0 = face_index * 20
            x1 = x0 + 20
            uv[(part_index, face)] = (
                x0 / width,
                1.0 - row_y1 / height,
                x1 / width,
                1.0 - row_y0 / height,
            )

    paper, stats = build_paper_model_texture(parts, uv, image)
    arr = np.asarray(paper)
    assert stats["contact_rectangle_count"] >= 2
    assert stats["masked_pixel_count"] > 0
    assert np.any(np.all(arr == np.array([255, 255, 255]), axis=2))
    assert np.any(np.all(arr == np.array([40, 80, 120]), axis=2))

    obj = tmp_path / "paper.obj"
    write_paper_model_obj(parts, uv, obj, "paper.mtl")
    face_lines = [line for line in obj.read_text().splitlines() if line.startswith("f ")]
    # Texture masking must not add geometry: always six quads per cuboid.
    assert len(face_lines) == 12


def test_paper_glb_has_one_geometry_and_closed_components(tmp_path: Path) -> None:
    parts = [
        part("a", 0, [-1.0, 0.0, 0.0], [2.0, 2.0, 2.0]),
        part("b", 1, [1.0, 0.0, 0.0], [2.0, 2.0, 2.0]),
    ]
    path = tmp_path / "paper.glb"
    texture = Image.new("RGB", (16, 16), (120, 80, 40))
    uv = {(i, face): (0.0, 0.0, 1.0, 1.0) for i in range(2) for face in FACE_NAMES}
    write_paper_model_glb(parts, uv, texture, path)
    scene = trimesh.load(path, force="scene")
    assert len(scene.geometry) == 1
    mesh = next(iter(scene.geometry.values()))
    # glTF requires UV-seam vertex duplication, so the textured preview uses
    # 24 render vertices per cuboid.  The canonical OBJ remains the watertight
    # 8-vertex-per-shell paper-unfolding artifact.
    assert len(mesh.vertices) == 48
    assert len(mesh.faces) == 24
    assert getattr(mesh.visual, "kind", None) == "texture"
    assert mesh.visual.uv is not None
    assert len(mesh.visual.uv) == 48
    assert getattr(mesh.visual.material, "baseColorTexture", None) is not None
