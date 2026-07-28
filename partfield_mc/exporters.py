from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import numpy as np
from PIL import Image
import trimesh

from .models import CuboidPart
from .texture import FACE_NAMES, cuboid_face_corners


@dataclass(frozen=True)
class FaceSurfacePatch:
    """One rectangular sub-region of a cuboid face.

    ``textured=False`` marks the exact contact/intersection portion shared with
    another cuboid after surface processing.  The geometry is kept so the box
    remains closed, but no image texture is assigned to that patch.
    """

    face: str
    textured: bool
    corners: np.ndarray
    uvs: np.ndarray | None
    local_rect: tuple[float, float, float, float]

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.local_rect
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _safe_name(value: str, fallback: str) -> str:
    """Return a stable GLB/OBJ-friendly part name."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    return cleaned or fallback


def _unique_part_names(parts: Sequence[CuboidPart]) -> list[str]:
    """Create unique names without mutating the source CuboidPart objects."""

    used: dict[str, int] = {}
    result: list[str] = []
    for index, part in enumerate(parts):
        base = _safe_name(part.name, f"part_{index:02d}")
        count = used.get(base, 0)
        used[base] = count + 1
        result.append(base if count == 0 else f"{base}_{count:02d}")
    return result


def _make_material(texture: Image.Image) -> trimesh.visual.material.SimpleMaterial:
    return trimesh.visual.material.SimpleMaterial(
        image=texture,
        diffuse=[255, 255, 255, 255],
        ambient=[255, 255, 255, 255],
        specular=[0, 0, 0, 255],
    )


def build_textured_part_mesh(
    part: CuboidPart,
    part_index: int,
    uv_rects: dict[tuple[int, str], tuple[float, float, float, float]],
    material: trimesh.visual.material.SimpleMaterial,
    *,
    faces_to_include: Sequence[str] | None = None,
) -> trimesh.Trimesh | None:
    """Build one independently selectable textured cuboid mesh for one PartField part."""

    if faces_to_include is None:
        faces_to_include = FACE_NAMES
    faces_to_include = list(faces_to_include)
    if not faces_to_include:
        return None

    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    uvs: list[tuple[float, float]] = []

    for face in faces_to_include:
        base = len(vertices)
        corners = cuboid_face_corners(part, face)
        u0, v0, u1, v1 = uv_rects[(part_index, face)]
        vertices.extend(corners)
        uvs.extend([(u0, v0), (u1, v0), (u1, v1), (u0, v1)])
        faces.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])

    visual = trimesh.visual.texture.TextureVisuals(
        uv=np.asarray(uvs, dtype=np.float64),
        material=material,
    )
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        visual=visual,
        process=False,
    )
    mesh.metadata.update(
        {
            "part_name": part.name,
            "segment_id": int(part.segment_id),
            "independent_part": True,
            "textured_faces": list(faces_to_include),
        }
    )
    return mesh


def build_untextured_part_mesh(
    part: CuboidPart,
    *,
    faces_to_include: Sequence[str],
    rgba: Sequence[int] = (160, 160, 160, 255),
) -> trimesh.Trimesh | None:
    """Build an untextured mesh for hidden/intersecting faces."""

    faces_to_include = list(faces_to_include)
    if not faces_to_include:
        return None

    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    for face in faces_to_include:
        base = len(vertices)
        corners = cuboid_face_corners(part, face)
        vertices.extend(corners)
        faces.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    color = np.asarray(rgba, dtype=np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        face_colors=np.tile(color[None, :], (len(faces), 1)),
    )
    mesh.metadata.update(
        {
            "part_name": part.name,
            "segment_id": int(part.segment_id),
            "independent_part": True,
            "untextured_faces": list(faces_to_include),
        }
    )
    return mesh


def build_textured_surface_patch_mesh(
    part: CuboidPart,
    part_index: int,
    parts: Sequence[CuboidPart],
    uv_rects: dict[tuple[int, str], tuple[float, float, float, float]],
    material: trimesh.visual.material.SimpleMaterial,
) -> trimesh.Trimesh | None:
    """Build only visible/textured patches of one post-surface cuboid."""

    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    uvs: list[tuple[float, float]] = []
    patch_count = 0

    for face in FACE_NAMES:
        for patch in _face_surface_patches(part_index, part, face, parts, uv_rects):
            if not patch.textured:
                continue
            if patch.uvs is None:
                raise RuntimeError("A textured surface patch is missing UV coordinates")
            base = len(vertices)
            vertices.extend(np.asarray(patch.corners, dtype=np.float64))
            uvs.extend([tuple(value) for value in np.asarray(patch.uvs, dtype=np.float64)])
            faces.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])
            patch_count += 1

    if not faces:
        return None

    visual = trimesh.visual.texture.TextureVisuals(
        uv=np.asarray(uvs, dtype=np.float64),
        material=material,
    )
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        visual=visual,
        process=False,
    )
    mesh.metadata.update(
        {
            "part_name": part.name,
            "segment_id": int(part.segment_id),
            "independent_part": True,
            "surface_patch_role": "visible_textured",
            "surface_patch_count": int(patch_count),
        }
    )
    return mesh


def build_untextured_surface_patch_mesh(
    part: CuboidPart,
    part_index: int,
    parts: Sequence[CuboidPart],
    uv_rects: dict[tuple[int, str], tuple[float, float, float, float]],
    rgba: Sequence[int] = (160, 160, 160, 255),
) -> trimesh.Trimesh | None:
    """Build exact contact/intersection patches without any image texture."""

    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    patch_count = 0
    hidden_area = 0.0

    for face in FACE_NAMES:
        for patch in _face_surface_patches(part_index, part, face, parts, uv_rects):
            if patch.textured:
                continue
            base = len(vertices)
            vertices.extend(np.asarray(patch.corners, dtype=np.float64))
            faces.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])
            patch_count += 1
            hidden_area += patch.area

    if not faces:
        return None

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    color = np.asarray(rgba, dtype=np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        face_colors=np.tile(color[None, :], (len(faces), 1)),
    )
    mesh.metadata.update(
        {
            "part_name": part.name,
            "segment_id": int(part.segment_id),
            "independent_part": True,
            "surface_patch_role": "contact_untextured",
            "surface_patch_count": int(patch_count),
            "untextured_contact_area": float(hidden_area),
        }
    )
    return mesh


def build_textured_part_meshes(
    parts: Sequence[CuboidPart],
    uv_rects: dict[tuple[int, str], tuple[float, float, float, float]],
    texture: Image.Image,
    *,
    surface_only: bool = False,
) -> list[tuple[str, trimesh.Trimesh]]:
    """Build one or more meshes per part while sharing one texture atlas."""

    material = _make_material(texture)
    names = _unique_part_names(parts)
    outputs: list[tuple[str, trimesh.Trimesh]] = []
    for part_index, part in enumerate(parts):
        name = names[part_index]
        if surface_only:
            textured = build_textured_surface_patch_mesh(
                part,
                part_index,
                parts,
                uv_rects,
                material,
            )
            if textured is not None:
                outputs.append((f"{name}__textured", textured))
            hidden = build_untextured_surface_patch_mesh(
                part,
                part_index,
                parts,
                uv_rects,
            )
            if hidden is not None:
                outputs.append((f"{name}__contact_untextured", hidden))
        else:
            textured = build_textured_part_mesh(part, part_index, uv_rects, material)
            if textured is not None:
                outputs.append((name, textured))
    return outputs


def build_textured_mesh(
    parts: Sequence[CuboidPart],
    uv_rects: dict[tuple[int, str], tuple[float, float, float, float]],
    texture: Image.Image,
) -> trimesh.Trimesh:
    """Build the legacy merged mesh.

    Kept for API compatibility. The normal GLB/OBJ exporters below now preserve
    independent parts instead of calling this function.
    """

    part_meshes = [mesh for _, mesh in build_textured_part_meshes(parts, uv_rects, texture)]
    if not part_meshes:
        raise ValueError("No parts were provided for export")
    merged = trimesh.util.concatenate(part_meshes)
    merged.metadata["independent_part"] = False
    return merged


def write_glb(
    parts: Sequence[CuboidPart],
    uv_rects: dict[tuple[int, str], tuple[float, float, float, float]],
    texture: Image.Image,
    output_path: Path,
    *,
    surface_only: bool = False,
) -> None:
    """Write one GLB containing one independently selectable node/mesh per part."""

    part_meshes = build_textured_part_meshes(parts, uv_rects, texture, surface_only=surface_only)
    if not part_meshes:
        raise ValueError("No parts were provided for GLB export")

    scene = trimesh.Scene(base_frame="world")
    for name, mesh in part_meshes:
        scene.add_geometry(
            mesh,
            node_name=name,
            geom_name=name,
            transform=np.eye(4, dtype=np.float64),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(scene.export(file_type="glb"))




_PAPER_FACE_VERTEX_IDS: dict[str, tuple[int, int, int, int]] = {
    "+X": (5, 1, 2, 6),
    "-X": (0, 4, 7, 3),
    "+Y": (7, 6, 2, 3),
    "-Y": (0, 1, 5, 4),
    "+Z": (4, 5, 6, 7),
    "-Z": (1, 0, 3, 2),
}


def _cuboid_shared_vertices(part: CuboidPart) -> np.ndarray:
    """Return the eight shared geometry vertices of one closed cuboid shell.

    The ordering is stable and is shared by all six quad faces.  UV seams are
    represented by OBJ texture-coordinate indices, not by duplicating geometry
    vertices.  This is important for Blender's Export Paper Model add-on: each
    cuboid remains a genuinely watertight connected component.
    """

    sx, sy, sz = np.asarray(part.size, dtype=np.float64)
    x0, x1 = -sx * 0.5, sx * 0.5
    y0, y1 = -sy * 0.5, sy * 0.5
    z0, z1 = -sz * 0.5, sz * 0.5
    local = np.asarray(
        [
            (x0, y0, z0),  # 0
            (x1, y0, z0),  # 1
            (x1, y1, z0),  # 2
            (x0, y1, z0),  # 3
            (x0, y0, z1),  # 4
            (x1, y0, z1),  # 5
            (x1, y1, z1),  # 6
            (x0, y1, z1),  # 7
        ],
        dtype=np.float64,
    )
    return trimesh.transform_points(local, part.transform)


def build_paper_model_texture(
    parts: Sequence[CuboidPart],
    uv_rects: dict[tuple[int, str], tuple[float, float, float, float]],
    texture: Image.Image,
    *,
    contact_color: Sequence[int] = (255, 255, 255),
) -> tuple[Image.Image, dict[str, float | int | list[int]]]:
    """Create a paper-safe atlas without subdividing cuboid geometry.

    The post-surface exporter previously represented a partial contact region by
    splitting a face into textured and untextured sub-quads.  That is visually
    correct but is undesirable for paper unfolding.  Here every cuboid keeps
    exactly six quads.  Contact rectangles are instead painted directly into a
    copy of the atlas using a neutral paper colour.
    """

    if not parts:
        raise ValueError("No parts were provided for paper texture export")
    rgb = texture.convert("RGB")
    pixels = np.asarray(rgb, dtype=np.uint8).copy()
    image_height, image_width = pixels.shape[:2]
    fill = np.asarray(tuple(contact_color)[:3], dtype=np.uint8)
    if fill.shape != (3,):
        raise ValueError("contact_color must provide at least three channels")

    masked_pixels = np.zeros((image_height, image_width), dtype=bool)
    contact_area = 0.0
    contact_rectangle_count = 0

    for part_index, part in enumerate(parts):
        for face in FACE_NAMES:
            _, _, _, width, height, tolerance, rectangles = _face_contact_rectangles(
                part_index,
                part,
                face,
                parts,
            )
            if width <= tolerance or height <= tolerance:
                continue
            atlas_u0, atlas_v0, atlas_u1, atlas_v1 = uv_rects[(part_index, face)]
            for x0, y0, x1, y1 in rectangles:
                if x1 - x0 <= tolerance or y1 - y0 <= tolerance:
                    continue
                fx0, fx1 = x0 / width, x1 / width
                fy0, fy1 = y0 / height, y1 / height
                u0 = atlas_u0 + fx0 * (atlas_u1 - atlas_u0)
                u1 = atlas_u0 + fx1 * (atlas_u1 - atlas_u0)
                v0 = atlas_v0 + fy0 * (atlas_v1 - atlas_v0)
                v1 = atlas_v0 + fy1 * (atlas_v1 - atlas_v0)

                px0 = max(0, min(image_width, int(np.floor(min(u0, u1) * image_width))))
                px1 = max(0, min(image_width, int(np.ceil(max(u0, u1) * image_width))))
                # Image rows run top-to-bottom while OBJ/GLTF V runs bottom-to-top.
                py0 = max(0, min(image_height, int(np.floor((1.0 - max(v0, v1)) * image_height))))
                py1 = max(0, min(image_height, int(np.ceil((1.0 - min(v0, v1)) * image_height))))
                if px1 <= px0 or py1 <= py0:
                    continue
                pixels[py0:py1, px0:px1] = fill
                masked_pixels[py0:py1, px0:px1] = True
                contact_area += float((x1 - x0) * (y1 - y0))
                contact_rectangle_count += 1

    return Image.fromarray(pixels, mode="RGB"), {
        "contact_color": [int(value) for value in fill.tolist()],
        "contact_rectangle_count": int(contact_rectangle_count),
        "contact_area": float(contact_area),
        "masked_pixel_count": int(np.count_nonzero(masked_pixels)),
    }


def write_paper_model_obj(
    parts: Sequence[CuboidPart],
    uv_rects: dict[tuple[int, str], tuple[float, float, float, float]],
    obj_path: Path,
    mtl_filename: str,
    *,
    object_name: str = "paper_model",
) -> list[dict[str, object]]:
    """Write one Blender object containing independent closed cuboid shells.

    This is deliberately *not* a Boolean union.  Cuboids do not share geometry
    vertices, even when they touch.  Each shell uses exactly eight position
    vertices and six quad faces, while OBJ's separate UV indices preserve one
    atlas rectangle per face without breaking the mesh topology.
    """

    if not parts:
        raise ValueError("No parts were provided for paper-model OBJ export")
    _assert_non_overlapping_parallel_cuboids(parts)

    safe_object_name = _safe_name(object_name, "paper_model")
    lines = [
        "# PartField Paper Model export",
        "# One Blender object containing independent watertight cuboid shells.",
        "# No Boolean union, no cross-shell vertex welding, six quads per shell.",
        f"# shell_count={len(parts)} total_vertices={len(parts) * 8} total_quads={len(parts) * 6}",
        f"mtllib {mtl_filename}",
        "s off",
        f"o {safe_object_name}",
        "usemtl mc_material",
    ]

    vertex_offset = 1
    uv_offset = 1
    shell_records: list[dict[str, object]] = []
    names = _unique_part_names(parts)

    for shell_index, (part, name) in enumerate(zip(parts, names)):
        vertices = _cuboid_shared_vertices(part)
        vertex_start = vertex_offset
        face_start = shell_index * 6 + 1
        lines.extend(
            [
                "",
                f"# shell_begin index={shell_index} name={name} segment_id={int(part.segment_id)}",
                f"# vertex_range={vertex_start}-{vertex_start + 7} face_range={face_start}-{face_start + 5}",
            ]
        )
        for vertex in vertices:
            lines.append(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}")

        for face in FACE_NAMES:
            u0, v0, u1, v1 = uv_rects[(shell_index, face)]
            face_uvs = ((u0, v0), (u1, v0), (u1, v1), (u0, v1))
            for uv in face_uvs:
                lines.append(f"vt {uv[0]:.8f} {uv[1]:.8f}")
            local_ids = _PAPER_FACE_VERTEX_IDS[face]
            geometry_ids = [vertex_offset + local_id for local_id in local_ids]
            texture_ids = [uv_offset + local_id for local_id in range(4)]
            lines.append(
                "f "
                + " ".join(
                    f"{geometry_id}/{texture_id}"
                    for geometry_id, texture_id in zip(geometry_ids, texture_ids)
                )
            )
            uv_offset += 4

        lines.append(f"# shell_end index={shell_index}")
        shell_records.append(
            {
                "shell_index": int(shell_index),
                "name": name,
                "segment_id": int(part.segment_id),
                "vertex_start": int(vertex_start),
                "vertex_end": int(vertex_start + 7),
                "face_start": int(face_start),
                "face_end": int(face_start + 5),
                "vertex_count": 8,
                "edge_count": 12,
                "quad_count": 6,
                "closed": True,
                "independent_shell": True,
            }
        )
        vertex_offset += 8

    obj_path.parent.mkdir(parents=True, exist_ok=True)
    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return shell_records


def write_paper_model_glb(
    parts: Sequence[CuboidPart],
    uv_rects: dict[tuple[int, str], tuple[float, float, float, float]],
    texture: Image.Image,
    output_path: Path,
    *,
    object_name: str = "paper_model",
) -> None:
    """Write one textured GLB object containing all final cuboids.

    glTF stores one UV coordinate per position vertex.  A cuboid atlas therefore
    needs UV-seam vertex duplication (24 render vertices per cuboid) even though
    the canonical paper-unfolding OBJ keeps only 8 shared geometry vertices per
    cuboid.  This GLB is intended for convenient textured viewing in Blender;
    ``paper_model.obj`` remains the recommended input for topology-sensitive
    paper unfolding.
    """

    if not parts:
        raise ValueError("No parts were provided for paper-model GLB export")
    _assert_non_overlapping_parallel_cuboids(parts)

    vertices: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    uvs: list[tuple[float, float]] = []

    for part_index, part in enumerate(parts):
        for face in FACE_NAMES:
            base = len(vertices)
            corners = np.asarray(cuboid_face_corners(part, face), dtype=np.float64)
            u0, v0, u1, v1 = uv_rects[(part_index, face)]
            vertices.extend(corners)
            uvs.extend(((u0, v0), (u1, v0), (u1, v1), (u0, v1)))
            triangles.extend(((base, base + 1, base + 2), (base, base + 2, base + 3)))

    material = _make_material(texture.convert("RGB"))
    visual = trimesh.visual.texture.TextureVisuals(
        uv=np.asarray(uvs, dtype=np.float64),
        material=material,
    )
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(triangles, dtype=np.int64),
        visual=visual,
        process=False,
    )
    mesh.metadata.update(
        {
            "paper_model": True,
            "single_object": True,
            "independent_cuboids": True,
            "shell_count": int(len(parts)),
            "texture_embedded": True,
            "render_vertices_per_cuboid": 24,
            "recommended_blender_input_for_unfolding": "paper_model.obj",
            "recommended_blender_input_for_textured_preview": "paper_model.glb",
        }
    )
    scene = trimesh.Scene(base_frame="world")
    name = _safe_name(object_name, "paper_model")
    scene.add_geometry(mesh, node_name=name, geom_name=name, transform=np.eye(4, dtype=np.float64))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(scene.export(file_type="glb"))



def _points_inside_part(points: np.ndarray, part: CuboidPart, tolerance: float = 1e-8) -> np.ndarray:
    """Return a mask indicating which world-space points are inside a cuboid."""

    points = np.asarray(points, dtype=np.float64)
    local = (points - part.center) @ part.rotation
    half = np.asarray(part.size, dtype=np.float64) / 2.0 + tolerance
    return np.all(np.abs(local) <= half, axis=1)


def _cuboid_world_corners(part: CuboidPart) -> np.ndarray:
    half = np.asarray(part.size, dtype=np.float64) / 2.0
    local = np.asarray(
        [[x, y, z] for x in (-half[0], half[0]) for y in (-half[1], half[1]) for z in (-half[2], half[2])],
        dtype=np.float64,
    )
    return trimesh.transform_points(local, part.transform)


def _rect_union_covers(rects: list[tuple[float, float, float, float]], width: float, height: float, tol: float) -> bool:
    """Exactly test whether axis-aligned rectangles cover the whole target rectangle."""

    if not rects:
        return False
    xs = [0.0, width]
    ys = [0.0, height]
    clipped: list[tuple[float, float, float, float]] = []
    for x0, y0, x1, y1 in rects:
        x0, x1 = max(0.0, x0), min(width, x1)
        y0, y1 = max(0.0, y0), min(height, y1)
        if x1 - x0 <= tol or y1 - y0 <= tol:
            continue
        clipped.append((x0, y0, x1, y1))
        xs.extend((x0, x1))
        ys.extend((y0, y1))
    if not clipped:
        return False

    xs = sorted(set(round(v / tol) * tol if tol > 0 else v for v in xs))
    ys = sorted(set(round(v / tol) * tol if tol > 0 else v for v in ys))
    for xa, xb in zip(xs[:-1], xs[1:]):
        if xb - xa <= tol:
            continue
        cx = (xa + xb) * 0.5
        for ya, yb in zip(ys[:-1], ys[1:]):
            if yb - ya <= tol:
                continue
            cy = (ya + yb) * 0.5
            if not any(x0 - tol <= cx <= x1 + tol and y0 - tol <= cy <= y1 + tol for x0, y0, x1, y1 in clipped):
                return False
    return True


def _face_is_fully_hidden(
    part_index: int,
    part: CuboidPart,
    face: str,
    parts: Sequence[CuboidPart],
    samples_per_axis: int = 17,
) -> bool:
    """Return True only when the entire face is covered by other cuboids.

    For AABB/shared-axis cuboids this uses an exact 2D rectangle-union test, so a
    face covered by one or several neighbouring boxes is removed. For arbitrary
    OBB layouts it falls back to dense conservative sampling.
    """

    corners = np.asarray(cuboid_face_corners(part, face), dtype=np.float64)
    p0, p1, p2, p3 = corners
    u_vec = p1 - p0
    v_vec = p3 - p0
    width = float(np.linalg.norm(u_vec))
    height = float(np.linalg.norm(v_vec))
    if width <= 1e-12 or height <= 1e-12:
        return False
    u_hat = u_vec / width
    v_hat = v_vec / height
    normal = np.cross(u_hat, v_hat)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)

    model_scale = max(float(np.max(np.asarray(p.size, dtype=np.float64))) for p in parts)
    tol = max(model_scale * 1e-7, 1e-9)
    plane = float(np.dot(p0, normal))
    coverage: list[tuple[float, float, float, float]] = []
    exact_parallel = True

    for other_index, other in enumerate(parts):
        if other_index == part_index:
            continue
        other_axes = np.asarray(other.rotation, dtype=np.float64)
        align = np.abs(np.asarray([[np.dot(u_hat, other_axes[:, j]) for j in range(3)],
                                   [np.dot(v_hat, other_axes[:, j]) for j in range(3)],
                                   [np.dot(normal, other_axes[:, j]) for j in range(3)]]))
        if not np.all(np.max(align, axis=1) > 1.0 - 1e-6):
            exact_parallel = False
            continue

        oc = _cuboid_world_corners(other)
        nvals = oc @ normal
        # The other cuboid must touch/cross the face plane and extend to the
        # outward side. This identifies an internal interface rather than a
        # cuboid merely behind the current face.
        if float(np.min(nvals)) > plane + tol or float(np.max(nvals)) < plane + tol:
            continue
        x = (oc - p0) @ u_hat
        y = (oc - p0) @ v_hat
        x0, x1 = float(np.min(x)), float(np.max(x))
        y0, y1 = float(np.min(y)), float(np.max(y))
        if x1 >= -tol and x0 <= width + tol and y1 >= -tol and y0 <= height + tol:
            coverage.append((x0, y0, x1, y1))

    if coverage and _rect_union_covers(coverage, width, height, tol):
        return True

    # Generic OBB fallback: sample the face itself and a tiny point immediately
    # outside it. Both are checked to handle touching and slightly overlapping boxes.
    coords = np.linspace(0.0, 1.0, max(3, samples_per_axis))
    samples = np.asarray([
        (1.0-u)*(1.0-v)*p0 + u*(1.0-v)*p1 + u*v*p2 + (1.0-u)*v*p3
        for u in coords for v in coords
    ], dtype=np.float64)
    probes = np.vstack((samples, samples + normal[None, :] * tol))
    covered = np.zeros(len(samples), dtype=bool)
    for other_index, other in enumerate(parts):
        if other_index == part_index:
            continue
        on_plane = _points_inside_part(samples, other, tolerance=tol)
        outside = _points_inside_part(samples + normal[None, :] * tol, other, tolerance=tol)
        covered |= on_plane & outside
        if np.all(covered):
            return True
    return False

def _visible_faces(parts: Sequence[CuboidPart], surface_only: bool) -> dict[int, list[str]]:
    if not surface_only:
        return {index: list(FACE_NAMES) for index in range(len(parts))}
    return {
        index: [
            face for face in FACE_NAMES
            if not _face_is_fully_hidden(index, part, face, parts)
        ]
        for index, part in enumerate(parts)
    }



def _is_axis_aligned_part(part: CuboidPart, atol: float = 1e-6) -> bool:
    """Return True when the cuboid axes are aligned to world XYZ (sign/permutation allowed)."""
    r = np.asarray(part.rotation, dtype=np.float64)
    a = np.abs(r)
    return bool(
        np.allclose(a.sum(axis=0), 1.0, atol=atol)
        and np.allclose(a.sum(axis=1), 1.0, atol=atol)
        and np.all((a < atol) | (np.abs(a - 1.0) < atol))
    )


def _aabb_bounds(part: CuboidPart) -> tuple[np.ndarray, np.ndarray]:
    corners = _cuboid_world_corners(part)
    return corners.min(axis=0), corners.max(axis=0)


def _find_source_face_for_patch(
    parts: Sequence[CuboidPart],
    axis: int,
    sign: int,
    plane: float,
    center: np.ndarray,
    tol: float,
) -> tuple[int, str]:
    """Find a cuboid face that owns an exposed union patch."""
    normal = np.zeros(3, dtype=np.float64)
    normal[axis] = float(sign)
    best = None
    best_area = float('inf')
    for i, part in enumerate(parts):
        lo, hi = _aabb_bounds(part)
        boundary = hi[axis] if sign > 0 else lo[axis]
        if abs(boundary - plane) > tol:
            continue
        other_axes = [a for a in range(3) if a != axis]
        if not all(lo[a] - tol <= center[a] <= hi[a] + tol for a in other_axes):
            continue
        # Map world normal to the cuboid's named local face.
        dots = np.asarray(part.rotation, dtype=np.float64).T @ normal
        local_axis = int(np.argmax(np.abs(dots)))
        local_sign = 1 if dots[local_axis] >= 0 else -1
        face = ("+" if local_sign > 0 else "-") + "XYZ"[local_axis]
        area = float(np.prod(np.delete(hi - lo, axis)))
        if area < best_area:
            best = (i, face)
            best_area = area
    if best is None:
        raise RuntimeError(f"Unable to map exposed patch at axis={axis}, sign={sign}, plane={plane}")
    return best


def _patch_uvs(
    part: CuboidPart,
    face: str,
    points: np.ndarray,
    uv_rect: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    """Map split union-surface vertices into the original face atlas rectangle."""
    fc = np.asarray(cuboid_face_corners(part, face), dtype=np.float64)
    p0, p1, _, p3 = fc
    uvec, vvec = p1 - p0, p3 - p0
    ulen2, vlen2 = float(np.dot(uvec, uvec)), float(np.dot(vvec, vvec))
    u0, v0, u1, v1 = uv_rect
    result = []
    for point in points:
        rel = point - p0
        u = float(np.dot(rel, uvec) / max(ulen2, 1e-30))
        v = float(np.dot(rel, vvec) / max(vlen2, 1e-30))
        result.append((u0 + u * (u1 - u0), v0 + v * (v1 - v0)))
    return result


def _greedy_rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Cover a boolean 2D mask with non-overlapping axis-aligned rectangles.

    Returns (row0, row1, col0, col1), with the upper bounds exclusive.
    """
    work = np.asarray(mask, dtype=bool).copy()
    rectangles: list[tuple[int, int, int, int]] = []
    rows, cols = work.shape
    for r0 in range(rows):
        for c0 in range(cols):
            if not work[r0, c0]:
                continue
            c1 = c0
            while c1 < cols and work[r0, c1]:
                c1 += 1
            r1 = r0 + 1
            while r1 < rows and np.all(work[r1, c0:c1]):
                r1 += 1
            work[r0:r1, c0:c1] = False
            rectangles.append((r0, r1, c0, c1))
    return rectangles


def _merge_close_coordinates(values: Sequence[float], tolerance: float) -> np.ndarray:
    """Sort coordinates and merge values that differ only by numerical noise."""

    ordered = sorted(float(value) for value in values)
    merged: list[float] = []
    for value in ordered:
        if not merged or abs(value - merged[-1]) > tolerance:
            merged.append(value)
        else:
            merged[-1] = (merged[-1] + value) * 0.5
    return np.asarray(merged, dtype=np.float64)


def _face_contact_rectangles(
    part_index: int,
    part: CuboidPart,
    face: str,
    parts: Sequence[CuboidPart],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, list[tuple[float, float, float, float]]]:
    """Return exact 2D contact rectangles on one face of a parallel cuboid set.

    Coordinates are expressed in the face frame whose origin is ``p0`` and axes
    are ``p0->p1`` and ``p0->p3``.  A rectangle is reported only when another
    cuboid touches the face plane and extends to the outward side.  Mere edge or
    point contact is ignored because it has zero surface area.
    """

    corners = np.asarray(cuboid_face_corners(part, face), dtype=np.float64)
    p0, p1, _, p3 = corners
    u_vec = p1 - p0
    v_vec = p3 - p0
    width = float(np.linalg.norm(u_vec))
    height = float(np.linalg.norm(v_vec))
    if width <= 1e-12 or height <= 1e-12:
        return p0, u_vec, v_vec, width, height, 1e-12, []

    u_hat = u_vec / width
    v_hat = v_vec / height
    normal = np.cross(u_hat, v_hat)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    model_scale = max(float(np.max(np.asarray(value.size, dtype=np.float64))) for value in parts)
    tolerance = max(model_scale * 1e-7, 1e-9)
    plane = float(np.dot(p0, normal))
    rectangles: list[tuple[float, float, float, float]] = []

    for other_index, other in enumerate(parts):
        if other_index == part_index:
            continue

        # Surface mode only supports one common cuboid frame.  This alignment
        # check prevents an arbitrary OBB projection from being treated as an
        # exact rectangular contact region.
        other_axes = np.asarray(other.rotation, dtype=np.float64)
        align = np.abs(
            np.asarray(
                [
                    [np.dot(u_hat, other_axes[:, axis]) for axis in range(3)],
                    [np.dot(v_hat, other_axes[:, axis]) for axis in range(3)],
                    [np.dot(normal, other_axes[:, axis]) for axis in range(3)],
                ],
                dtype=np.float64,
            )
        )
        if not np.all(np.max(align, axis=1) > 1.0 - 1e-6):
            raise ValueError(
                "Partial untextured surface patches require AABB/shared-axis cuboids"
            )

        other_corners = _cuboid_world_corners(other)
        normal_values = other_corners @ normal
        other_min = float(np.min(normal_values))
        other_max = float(np.max(normal_values))

        # The other box must touch/cross the plane and have positive thickness
        # on the outward side.  A box entirely behind the face is not contact.
        if other_min > plane + tolerance:
            continue
        if other_max < plane + tolerance:
            continue

        x_values = (other_corners - p0) @ u_hat
        y_values = (other_corners - p0) @ v_hat
        x0 = max(0.0, float(np.min(x_values)))
        x1 = min(width, float(np.max(x_values)))
        y0 = max(0.0, float(np.min(y_values)))
        y1 = min(height, float(np.max(y_values)))
        if x1 - x0 <= tolerance or y1 - y0 <= tolerance:
            continue
        rectangles.append((x0, y0, x1, y1))

    return p0, u_vec, v_vec, width, height, tolerance, rectangles


def _face_surface_patches(
    part_index: int,
    part: CuboidPart,
    face: str,
    parts: Sequence[CuboidPart],
    uv_rects: dict[tuple[int, str], tuple[float, float, float, float]],
) -> list[FaceSurfacePatch]:
    """Split a face into visible textured and contact untextured rectangles."""

    p0, u_vec, v_vec, width, height, tolerance, contacts = _face_contact_rectangles(
        part_index,
        part,
        face,
        parts,
    )
    if width <= tolerance or height <= tolerance:
        return []

    x_coordinates: list[float] = [0.0, width]
    y_coordinates: list[float] = [0.0, height]
    for x0, y0, x1, y1 in contacts:
        x_coordinates.extend((x0, x1))
        y_coordinates.extend((y0, y1))
    xs = _merge_close_coordinates(x_coordinates, tolerance)
    ys = _merge_close_coordinates(y_coordinates, tolerance)

    hidden = np.zeros((len(ys) - 1, len(xs) - 1), dtype=bool)
    for row, (y0, y1) in enumerate(zip(ys[:-1], ys[1:])):
        if y1 - y0 <= tolerance:
            continue
        center_y = (float(y0) + float(y1)) * 0.5
        for col, (x0, x1) in enumerate(zip(xs[:-1], xs[1:])):
            if x1 - x0 <= tolerance:
                continue
            center_x = (float(x0) + float(x1)) * 0.5
            hidden[row, col] = any(
                rect_x0 - tolerance <= center_x <= rect_x1 + tolerance
                and rect_y0 - tolerance <= center_y <= rect_y1 + tolerance
                for rect_x0, rect_y0, rect_x1, rect_y1 in contacts
            )

    u_hat = u_vec / width
    v_hat = v_vec / height
    atlas_u0, atlas_v0, atlas_u1, atlas_v1 = uv_rects[(part_index, face)]
    patches: list[FaceSurfacePatch] = []

    def append_rectangles(mask: np.ndarray, textured: bool) -> None:
        for row0, row1, col0, col1 in _greedy_rectangles(mask):
            x0, x1 = float(xs[col0]), float(xs[col1])
            y0, y1 = float(ys[row0]), float(ys[row1])
            if x1 - x0 <= tolerance or y1 - y0 <= tolerance:
                continue
            corners = np.asarray(
                [
                    p0 + u_hat * x0 + v_hat * y0,
                    p0 + u_hat * x1 + v_hat * y0,
                    p0 + u_hat * x1 + v_hat * y1,
                    p0 + u_hat * x0 + v_hat * y1,
                ],
                dtype=np.float64,
            )
            patch_uvs: np.ndarray | None = None
            if textured:
                fu0, fu1 = x0 / width, x1 / width
                fv0, fv1 = y0 / height, y1 / height
                patch_uvs = np.asarray(
                    [
                        (
                            atlas_u0 + fu0 * (atlas_u1 - atlas_u0),
                            atlas_v0 + fv0 * (atlas_v1 - atlas_v0),
                        ),
                        (
                            atlas_u0 + fu1 * (atlas_u1 - atlas_u0),
                            atlas_v0 + fv0 * (atlas_v1 - atlas_v0),
                        ),
                        (
                            atlas_u0 + fu1 * (atlas_u1 - atlas_u0),
                            atlas_v0 + fv1 * (atlas_v1 - atlas_v0),
                        ),
                        (
                            atlas_u0 + fu0 * (atlas_u1 - atlas_u0),
                            atlas_v0 + fv1 * (atlas_v1 - atlas_v0),
                        ),
                    ],
                    dtype=np.float64,
                )
            patches.append(
                FaceSurfacePatch(
                    face=face,
                    textured=textured,
                    corners=corners,
                    uvs=patch_uvs,
                    local_rect=(x0, y0, x1, y1),
                )
            )

    append_rectangles(~hidden, textured=True)
    append_rectangles(hidden, textured=False)
    return patches


def _write_aabb_union_surface_obj(
    parts: Sequence[CuboidPart],
    uv_rects: dict[tuple[int, str], tuple[float, float, float, float]],
    obj_path: Path,
    mtl_filename: str,
) -> None:
    """Write a rectilinear quad-only external shell for AABB cuboids.

    The cuboids are unioned on their exact boundary-coordinate grid. Internal
    faces are removed. Adjacent coplanar exposed cells that use the same source
    texture face are greedily merged into larger rectangles. Every OBJ ``f``
    record has exactly four vertices, so no diagonal triangle edges are written.
    """
    if not parts:
        raise ValueError("No parts were provided for export")
    if not all(_is_axis_aligned_part(p) for p in parts):
        raise ValueError("Quad-only surface OBJ requires --fit-mode aabb")

    bounds = [_aabb_bounds(p) for p in parts]
    scale = max(float(np.max(hi - lo)) for lo, hi in bounds)
    tol = max(scale * 1e-7, 1e-9)

    def unique_coords(values: list[float]) -> np.ndarray:
        values = sorted(values)
        out: list[float] = []
        for value in values:
            if not out or abs(value - out[-1]) > tol:
                out.append(value)
            else:
                out[-1] = (out[-1] + value) * 0.5
        return np.asarray(out, dtype=np.float64)

    coords = [unique_coords([v for lo, hi in bounds for v in (lo[a], hi[a])]) for a in range(3)]
    nx, ny, nz = (len(c) - 1 for c in coords)
    dims = (nx, ny, nz)
    occupied = np.zeros(dims, dtype=bool)
    centers = [(c[:-1] + c[1:]) * 0.5 for c in coords]
    for ix, x in enumerate(centers[0]):
        for iy, y in enumerate(centers[1]):
            for iz, z in enumerate(centers[2]):
                point = np.array([x, y, z], dtype=np.float64)
                occupied[ix, iy, iz] = any(
                    np.all(point >= lo - tol) and np.all(point <= hi + tol)
                    for lo, hi in bounds
                )

    lines = [
        "# PartField exact AABB union surface",
        "# Quad-only rectilinear shell for Blender Export Paper Model.",
        "# Partial and fully covered internal faces removed; coplanar compatible cells merged.",
        f"mtllib {mtl_filename}",
        "s off",
        "o union_surface",
        "g union_surface",
        "usemtl mc_material",
    ]
    # OBJ topology must share identical world-space vertices globally.
    # UV coordinates remain face-local because atlas seams may differ across faces.
    vertex_indices: dict[tuple[int, int, int], int] = {}
    vertex_values: list[np.ndarray] = []
    uv_offset = 1
    emitted_quads = 0

    quant_step = max(tol, 1e-12)

    def vertex_index(point: np.ndarray) -> int:
        key = tuple(int(round(float(value) / quant_step)) for value in point)
        existing = vertex_indices.get(key)
        if existing is not None:
            return existing
        index = len(vertex_values) + 1
        vertex_indices[key] = index
        vertex_values.append(np.asarray(point, dtype=np.float64).copy())
        return index

    # Process one world-axis direction and one grid plane at a time.
    for axis in range(3):
        plane_axes = [a for a in range(3) if a != axis]
        a0, a1 = plane_axes
        for sign in (-1, +1):
            plane_count = dims[axis] + 1
            for plane_index in range(plane_count):
                # Cells on this plane are indexed on the other two axes.
                shape = (dims[a0], dims[a1])
                owner = np.full(shape + (2,), -1, dtype=np.int64)
                exposed = np.zeros(shape, dtype=bool)

                for i0 in range(dims[a0]):
                    for i1 in range(dims[a1]):
                        cell = [0, 0, 0]
                        cell[a0] = i0
                        cell[a1] = i1
                        if sign < 0:
                            inside_axis = plane_index
                            if inside_axis >= dims[axis]:
                                continue
                            cell[axis] = inside_axis
                            neighbor_axis = inside_axis - 1
                        else:
                            inside_axis = plane_index - 1
                            if inside_axis < 0:
                                continue
                            cell[axis] = inside_axis
                            neighbor_axis = inside_axis + 1
                        if not occupied[tuple(cell)]:
                            continue
                        neighbor_inside = 0 <= neighbor_axis < dims[axis]
                        if neighbor_inside:
                            neighbor = cell.copy()
                            neighbor[axis] = neighbor_axis
                            if occupied[tuple(neighbor)]:
                                continue

                        plane_value = coords[axis][plane_index]
                        center = np.zeros(3, dtype=np.float64)
                        center[axis] = plane_value
                        center[a0] = (coords[a0][i0] + coords[a0][i0 + 1]) * 0.5
                        center[a1] = (coords[a1][i1] + coords[a1][i1 + 1]) * 0.5
                        part_index, face = _find_source_face_for_patch(
                            parts, axis, sign, plane_value, center, tol
                        )
                        face_index = FACE_NAMES.index(face)
                        exposed[i0, i1] = True
                        owner[i0, i1] = (part_index, face_index)

                # Merge only cells that share the same source atlas face so UVs
                # remain continuous and one quad maps to one rectangular region.
                keys = sorted({tuple(v) for v in owner[exposed]})
                for part_index, face_index in keys:
                    key_mask = exposed & (owner[..., 0] == part_index) & (owner[..., 1] == face_index)
                    face = FACE_NAMES[int(face_index)]
                    for r0, r1, c0, c1 in _greedy_rectangles(key_mask):
                        plane_value = coords[axis][plane_index]
                        low0, high0 = coords[a0][r0], coords[a0][r1]
                        low1, high1 = coords[a1][c0], coords[a1][c1]
                        raw: list[np.ndarray] = []
                        for q0, q1 in ((0, 0), (1, 0), (1, 1), (0, 1)):
                            point = np.zeros(3, dtype=np.float64)
                            point[axis] = plane_value
                            point[a0] = high0 if q0 else low0
                            point[a1] = high1 if q1 else low1
                            raw.append(point)
                        points = np.asarray(raw, dtype=np.float64)
                        desired_normal = np.zeros(3, dtype=np.float64)
                        desired_normal[axis] = float(sign)
                        if float(np.dot(np.cross(points[1] - points[0], points[2] - points[0]), desired_normal)) < 0.0:
                            points = points[[0, 3, 2, 1]]

                        patch_uv = _patch_uvs(
                            parts[int(part_index)],
                            face,
                            points,
                            uv_rects[(int(part_index), face)],
                        )
                        lines.append(f"# source_part={part_index} source_face={face}")
                        v = [vertex_index(point) for point in points]
                        for uv in patch_uv:
                            lines.append(f"vt {uv[0]:.8f} {uv[1]:.8f}")
                        t = [uv_offset + i for i in range(4)]
                        lines.append(
                            f"f {v[0]}/{t[0]} {v[1]}/{t[1]} {v[2]}/{t[2]} {v[3]}/{t[3]}"
                        )
                        uv_offset += 4
                        emitted_quads += 1

    # Insert the pooled vertex table before texture coordinates and faces.
    vertex_lines = [
        f"v {point[0]:.8f} {point[1]:.8f} {point[2]:.8f}"
        for point in vertex_values
    ]
    first_payload = next((i for i, line in enumerate(lines) if line.startswith("# source_part=")), len(lines))
    lines[first_payload:first_payload] = vertex_lines
    lines.insert(3, f"# shared_vertices={len(vertex_values)} merged_quads={emitted_quads} triangles=0")
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _write_obj_file(
    parts: Sequence[CuboidPart],
    uv_rects: dict[tuple[int, str], tuple[float, float, float, float]],
    obj_path: Path,
    mtl_filename: str,
    *,
    surface_only: bool = False,
) -> None:
    names = _unique_part_names(parts)
    visible = _visible_faces(parts, surface_only=surface_only)
    lines = [
        "# PartField Minecraft-style cuboid model",
        "# Internal fully-covered cuboid faces removed." if surface_only else "# All cuboid faces retained.",
        f"mtllib {mtl_filename}",
        "s off",
    ]
    vertex_offset = 1
    uv_offset = 1

    for part_index, (part, name) in enumerate(zip(parts, names)):
        lines.extend(["", f"# segment_id={int(part.segment_id)} source_name={part.name}", f"o {name}", f"g {name}", "usemtl mc_material"])
        for face in visible[part_index]:
            corners = cuboid_face_corners(part, face)
            u0, v0, u1, v1 = uv_rects[(part_index, face)]
            face_uvs = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
            for vertex in corners:
                lines.append(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}")
            for uv in face_uvs:
                lines.append(f"vt {uv[0]:.8f} {uv[1]:.8f}")
            v = [vertex_offset + i for i in range(4)]
            t = [uv_offset + i for i in range(4)]
            lines.append(f"f {v[0]}/{t[0]} {v[1]}/{t[1]} {v[2]}/{t[2]}")
            lines.append(f"f {v[0]}/{t[0]} {v[2]}/{t[2]} {v[3]}/{t[3]}")
            vertex_offset += 4
            uv_offset += 4

    obj_path.parent.mkdir(parents=True, exist_ok=True)
    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def _assert_non_overlapping_parallel_cuboids(parts: Sequence[CuboidPart]) -> None:
    if len(parts) < 2:
        return
    rotation = np.asarray(parts[0].rotation, dtype=np.float64)
    for part in parts[1:]:
        if not np.allclose(part.rotation, rotation, atol=1e-6):
            raise ValueError(
                "Strict surface OBJ requires AABB or shared-axis cuboids; "
                "use --fit-mode aabb or shared"
            )
    model_extent = max(float(np.max(np.asarray(part.size, dtype=np.float64))) for part in parts)
    tolerance = max(model_extent * 1e-9, 1e-12)
    for i in range(len(parts)):
        center_i = parts[i].center @ rotation
        half_i = np.asarray(parts[i].size, dtype=np.float64) * 0.5
        min_i, max_i = center_i - half_i, center_i + half_i
        for j in range(i + 1, len(parts)):
            center_j = parts[j].center @ rotation
            half_j = np.asarray(parts[j].size, dtype=np.float64) * 0.5
            min_j, max_j = center_j - half_j, center_j + half_j
            depth = np.minimum(max_i, max_j) - np.maximum(min_i, min_j)
            if np.all(depth > tolerance):
                raise ValueError(
                    f"Strict surface OBJ received overlapping cuboids for segments "
                    f"{parts[i].segment_id} and {parts[j].segment_id}: overlap={depth.tolist()}"
                )


def _write_closed_cuboids_obj(
    parts: Sequence[CuboidPart],
    uv_rects: dict[tuple[int, str], tuple[float, float, float, float]],
    obj_path: Path,
    mtl_filename: str,
) -> None:
    """Write closed cuboids with exact untextured contact sub-regions.

    Each original cuboid face is partitioned into rectangular patches. Visible
    patches use ``mc_material`` and retain the correct sub-rectangle of the face
    UV atlas. Contact/intersection patches use ``mc_hidden`` and contain no
    texture-coordinate reference. Together all patches still cover every face,
    so each retained cuboid remains geometrically closed.
    """

    if not parts:
        raise ValueError("No parts were provided for surface OBJ export")
    _assert_non_overlapping_parallel_cuboids(parts)
    names = _unique_part_names(parts)
    model_extent = max(float(np.max(np.asarray(part.size, dtype=np.float64))) for part in parts)
    quant_step = max(model_extent * 1e-9, 1e-12)

    lines = [
        "# PartField large-first post-surface cuboid model",
        "# Contact/intersection sub-regions are geometry-only and have no texture UVs.",
        f"# part_count={len(parts)} overlaps_expected=0 triangles=0",
        f"mtllib {mtl_filename}",
        "s off",
    ]
    global_vertex_offset = 1
    global_uv_offset = 1
    total_quads = 0
    total_textured_quads = 0
    total_untextured_quads = 0

    for part_index, (part, name) in enumerate(zip(parts, names)):
        patches: list[FaceSurfacePatch] = []
        for face in FACE_NAMES:
            patches.extend(_face_surface_patches(part_index, part, face, parts, uv_rects))

        vertex_indices: dict[tuple[int, int, int], int] = {}
        vertex_values: list[np.ndarray] = []

        def local_vertex_index(point: np.ndarray) -> int:
            key = tuple(int(round(float(value) / quant_step)) for value in point)
            if key not in vertex_indices:
                vertex_indices[key] = len(vertex_values)
                vertex_values.append(np.asarray(point, dtype=np.float64))
            return global_vertex_offset + vertex_indices[key]

        patch_records: list[tuple[FaceSurfacePatch, list[int]]] = []
        for patch in patches:
            indices = [local_vertex_index(point) for point in patch.corners]
            patch_records.append((patch, indices))

        textured_area = float(sum(patch.area for patch in patches if patch.textured))
        untextured_area = float(sum(patch.area for patch in patches if not patch.textured))
        lines.extend(
            [
                "",
                f"# segment_id={int(part.segment_id)} source_name={part.name} closed=true",
                f"# surface_patches={len(patches)} textured_area={textured_area:.10f} "
                f"untextured_contact_area={untextured_area:.10f}",
                f"o {name}",
                f"g {name}",
            ]
        )
        for vertex in vertex_values:
            lines.append(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}")

        active_material: str | None = None
        for patch, vertices in patch_records:
            material = "mc_material" if patch.textured else "mc_hidden"
            if material != active_material:
                lines.append(f"usemtl {material}")
                active_material = material
            lines.append(
                f"# face={patch.face} patch_rect="
                f"{patch.local_rect[0]:.8f},{patch.local_rect[1]:.8f},"
                f"{patch.local_rect[2]:.8f},{patch.local_rect[3]:.8f} "
                f"textured={str(patch.textured).lower()}"
            )
            if patch.textured:
                if patch.uvs is None:
                    raise RuntimeError("Textured OBJ patch is missing UV coordinates")
                for uv in patch.uvs:
                    lines.append(f"vt {uv[0]:.8f} {uv[1]:.8f}")
                texcoords = [global_uv_offset + index for index in range(4)]
                lines.append(
                    f"f {vertices[0]}/{texcoords[0]} {vertices[1]}/{texcoords[1]} "
                    f"{vertices[2]}/{texcoords[2]} {vertices[3]}/{texcoords[3]}"
                )
                global_uv_offset += 4
                total_textured_quads += 1
            else:
                # No /vt indices: this patch deliberately has no image texture.
                lines.append(
                    f"f {vertices[0]} {vertices[1]} {vertices[2]} {vertices[3]}"
                )
                total_untextured_quads += 1
            total_quads += 1

        global_vertex_offset += len(vertex_values)

    lines.insert(
        3,
        f"# quads={total_quads} textured_quads={total_textured_quads} "
        f"untextured_contact_quads={total_untextured_quads}",
    )
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_obj(
    parts: Sequence[CuboidPart],
    uv_rects: dict[tuple[int, str], tuple[float, float, float, float]],
    obj_path: Path,
    mtl_filename: str,
    *,
    surface_only: bool = False,
) -> None:
    """Write one OBJ.

    ``surface_only`` writes the large-first post-surface cuboids. Fully occluded
    clusters may already have been dropped, and exact contact sub-regions are
    emitted with geometry-only ``mc_hidden`` material records.
    """

    if surface_only:
        _write_closed_cuboids_obj(parts, uv_rects, obj_path, mtl_filename)
    else:
        _write_obj_file(parts, uv_rects, obj_path, mtl_filename, surface_only=False)


def write_split_objs(
    parts: Sequence[CuboidPart],
    uv_rects: dict[tuple[int, str], tuple[float, float, float, float]],
    output_dir: Path,
    mtl_filename: str = "../mc_model.mtl",
) -> list[Path]:
    """Write one OBJ file per PartField cuboid."""

    output_dir.mkdir(parents=True, exist_ok=True)
    names = _unique_part_names(parts)
    outputs: list[Path] = []
    for index, (part, name) in enumerate(zip(parts, names)):
        path = output_dir / f"{name}.obj"
        local_uvs = {(0, face): uv_rects[(index, face)] for face in FACE_NAMES}
        _write_obj_file([part], local_uvs, path, mtl_filename, surface_only=False)
        outputs.append(path)
    return outputs


def write_mtl(output_path: Path, texture_filename: str) -> None:
    output_path.write_text(
        f"""# PartField MC material
newmtl mc_material
Ka 1.000000 1.000000 1.000000
Kd 1.000000 1.000000 1.000000
Ks 0.000000 0.000000 0.000000
d 1.000000
illum 1
map_Kd {texture_filename}

newmtl mc_hidden
Ka 0.650000 0.650000 0.650000
Kd 0.650000 0.650000 0.650000
Ks 0.000000 0.000000 0.000000
d 1.000000
illum 1
""",
        encoding="utf-8",
    )
