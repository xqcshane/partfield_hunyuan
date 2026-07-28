from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
import trimesh

from .primitive_fit import PrimitivePart, triangulate_polygons
from .texture import ColoredSurfacePointCloud


def _safe_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned.strip("._") or fallback


def _unique_names(parts: Sequence[PrimitivePart]) -> list[str]:
    used: dict[str, int] = {}
    names: list[str] = []
    for index, part in enumerate(parts):
        base = _safe_name(part.name, f"part_{index:03d}")
        count = used.get(base, 0)
        used[base] = count + 1
        names.append(base if count == 0 else f"{base}_{count + 1}")
    return names


def _polygon_normal(points: np.ndarray) -> np.ndarray:
    normal = np.zeros(3, dtype=np.float64)
    for index in range(len(points)):
        normal += np.cross(points[index], points[(index + 1) % len(points)])
    length = float(np.linalg.norm(normal))
    if length <= 1e-12:
        raise ValueError("Degenerate primitive polygon")
    return normal / length


def _face_projection(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    normal = _polygon_normal(points)
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    edge = edges[int(np.argmax(lengths))]
    edge_length = float(np.linalg.norm(edge))
    if edge_length <= 1e-12:
        raise ValueError("Degenerate primitive face edge")
    axis_u = edge / edge_length
    axis_v = np.cross(normal, axis_u)
    axis_v /= max(float(np.linalg.norm(axis_v)), 1e-12)
    origin = points.mean(axis=0)
    projected = np.column_stack(((points - origin) @ axis_u, (points - origin) @ axis_v))
    return origin, axis_u, axis_v, projected


def _shade_factor(normal: np.ndarray) -> float:
    light = np.asarray([0.45, 0.82, 0.35], dtype=np.float64)
    light /= np.linalg.norm(light)
    diffuse = abs(float(np.dot(normal, light)))
    return float(0.72 + 0.36 * diffuse)


def build_primitive_texture_atlas(
    parts: Sequence[PrimitivePart],
    sampler: ColoredSurfacePointCloud,
    *,
    face_resolution: int = 64,
    padding: int = 1,
    palette_size: int = 0,
    face_shading: bool = False,
) -> tuple[Image.Image, dict[tuple[int, int], np.ndarray]]:
    """Bake one planar texture tile for every canonical paper face."""

    if not parts:
        raise ValueError("No primitive parts were provided for texture baking")
    if face_resolution < 1:
        raise ValueError("face_resolution must be >= 1")
    total_faces = sum(part.face_count for part in parts)
    columns = max(1, int(math.ceil(math.sqrt(total_faces))))
    rows = max(1, int(math.ceil(total_faces / columns)))
    stride = face_resolution + 2 * max(int(padding), 0)
    width = columns * stride
    height = rows * stride
    atlas = np.zeros((height, width, 3), dtype=np.uint8)
    face_uvs: dict[tuple[int, int], np.ndarray] = {}

    tile_index = 0
    for part_index, part in enumerate(parts):
        vertices = np.asarray(part.vertices, dtype=np.float64)
        for face_index, polygon in enumerate(part.polygons):
            points = vertices[np.asarray(polygon, dtype=np.int64)]
            origin, axis_u, axis_v, projected = _face_projection(points)
            mins = projected.min(axis=0)
            maxs = projected.max(axis=0)
            extents = np.maximum(maxs - mins, 1e-9)

            columns_local = (np.arange(face_resolution, dtype=np.float64) + 0.5) / face_resolution
            rows_local = (np.arange(face_resolution, dtype=np.float64) + 0.5) / face_resolution
            grid_u, grid_v = np.meshgrid(columns_local, rows_local)
            plane_u = mins[0] + grid_u.reshape(-1) * extents[0]
            # Image rows increase downward.  Reverse the planar v direction so
            # the stored UV convention remains bottom-to-top.
            plane_v = maxs[1] - grid_v.reshape(-1) * extents[1]
            query = (
                origin[None, :]
                + plane_u[:, None] * axis_u[None, :]
                + plane_v[:, None] * axis_v[None, :]
            )
            untextured_faces = {
                int(value)
                for value in part.metadata.get("untextured_contact_face_indices", [])
            }
            if face_index in untextured_faces:
                # Connector end caps are hidden glue/contact surfaces.  Keeping
                # them neutral avoids sampling unrelated source texture through
                # the gap between two fitted parts.
                tile = np.full(
                    (face_resolution, face_resolution, 3),
                    255,
                    dtype=np.uint8,
                )
            else:
                colors = sampler.sample(query).reshape(
                    face_resolution, face_resolution, 3
                ).astype(np.float32)
                if face_shading:
                    colors *= _shade_factor(_polygon_normal(points))
                tile = np.clip(colors, 0, 255).astype(np.uint8)

            tile_x = tile_index % columns
            tile_y = tile_index // columns
            x0 = tile_x * stride + padding
            y0 = tile_y * stride + padding
            x1 = x0 + face_resolution
            y1 = y0 + face_resolution
            atlas[y0:y1, x0:x1] = tile
            if padding:
                atlas[y0:y1, x0 - padding : x0] = tile[:, :1]
                atlas[y0:y1, x1 : x1 + padding] = tile[:, -1:]
                atlas[y0 - padding : y0, x0 - padding : x1 + padding] = atlas[
                    y0 : y0 + 1, x0 - padding : x1 + padding
                ]
                atlas[y1 : y1 + padding, x0 - padding : x1 + padding] = atlas[
                    y1 - 1 : y1, x0 - padding : x1 + padding
                ]

            local_uv = (projected - mins[None, :]) / extents[None, :]
            atlas_uv = np.column_stack(
                (
                    (x0 + local_uv[:, 0] * face_resolution) / width,
                    1.0 - (y1 - local_uv[:, 1] * face_resolution) / height,
                )
            )
            face_uvs[(part_index, face_index)] = np.asarray(atlas_uv, dtype=np.float64)
            tile_index += 1
            print(
                f"[PrimitiveTexture] {tile_index}/{total_faces}: {part.name} face_{face_index:03d}",
                flush=True,
            )

    image = Image.fromarray(atlas, mode="RGB")
    if 2 <= palette_size <= 256:
        image = image.quantize(colors=palette_size, method=Image.Quantize.MEDIANCUT).convert("RGB")
    return image, face_uvs


def _validate_shell(part: PrimitivePart) -> dict[str, int | bool]:
    edge_counts: dict[tuple[int, int], int] = {}
    for polygon in part.polygons:
        for a, b in zip(polygon, polygon[1:] + polygon[:1]):
            edge = tuple(sorted((int(a), int(b))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    closed = bool(edge_counts) and set(edge_counts.values()) == {2}
    if not closed:
        invalid = [edge for edge, count in edge_counts.items() if count != 2]
        raise ValueError(
            f"Primitive segment {part.segment_id} is not watertight; invalid edges={invalid[:5]}"
        )
    return {
        "vertex_count": int(len(part.vertices)),
        "edge_count": int(len(edge_counts)),
        "paper_face_count": int(part.face_count),
        "triangle_count": int(part.triangle_count),
        "closed": True,
    }


def write_primitive_obj(
    parts: Sequence[PrimitivePart],
    face_uvs: dict[tuple[int, int], np.ndarray],
    obj_path: Path,
    mtl_filename: str,
    *,
    object_name: str = "paper_model",
) -> list[dict[str, object]]:
    """Write one OBJ object containing watertight shells in one connected assembly."""

    if not parts:
        raise ValueError("No primitive parts were provided for OBJ export")
    names = _unique_names(parts)
    object_safe = _safe_name(object_name, "paper_model")
    total_vertices = sum(len(part.vertices) for part in parts)
    total_faces = sum(part.face_count for part in parts)
    lines = [
        "# PartField automatic primitive paper model",
        "# One OBJ object; each segment is a closed shell and fitted contacts keep the assembly connected.",
        f"# shell_count={len(parts)} total_vertices={total_vertices} total_paper_faces={total_faces}",
        f"mtllib {mtl_filename}",
        "s off",
        f"o {object_safe}",
        "usemtl mc_material",
    ]
    vertex_offset = 1
    uv_offset = 1
    face_cursor = 1
    records: list[dict[str, object]] = []

    for part_index, (part, name) in enumerate(zip(parts, names)):
        topology = _validate_shell(part)
        vertices = np.asarray(part.vertices, dtype=np.float64)
        vertex_start = vertex_offset
        face_start = face_cursor
        lines.extend(
            [
                "",
                f"# shell_begin index={part_index} name={name} segment_id={int(part.segment_id)}",
                f"# primitive_type={part.primitive_type} source_faces={part.source_face_count} "
                f"target_faces={part.target_face_count} paper_faces={part.face_count}",
            ]
        )
        for vertex in vertices:
            lines.append(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}")

        for face_index, polygon in enumerate(part.polygons):
            uvs = np.asarray(face_uvs[(part_index, face_index)], dtype=np.float64)
            if len(uvs) != len(polygon):
                raise ValueError("Primitive face UV count does not match polygon vertex count")
            for uv in uvs:
                lines.append(f"vt {uv[0]:.8f} {uv[1]:.8f}")
            geometry_ids = [vertex_offset + int(local_id) for local_id in polygon]
            texture_ids = [uv_offset + index for index in range(len(polygon))]
            lines.append(
                "f "
                + " ".join(
                    f"{geometry_id}/{texture_id}"
                    for geometry_id, texture_id in zip(geometry_ids, texture_ids)
                )
            )
            uv_offset += len(polygon)
            face_cursor += 1

        records.append(
            {
                "shell_index": int(part_index),
                "name": name,
                "segment_id": int(part.segment_id),
                "primitive_type": part.primitive_type,
                "source_face_count": int(part.source_face_count),
                "target_face_count": int(part.target_face_count),
                "vertex_start": int(vertex_start),
                "vertex_end": int(vertex_start + len(vertices) - 1),
                "face_start": int(face_start),
                "face_end": int(face_cursor - 1),
                "independent_shell": True,
                **topology,
            }
        )
        lines.append(f"# shell_end index={part_index}")
        vertex_offset += len(vertices)

    obj_path.parent.mkdir(parents=True, exist_ok=True)
    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return records


def write_split_primitive_objs(
    parts: Sequence[PrimitivePart],
    face_uvs: dict[tuple[int, int], np.ndarray],
    output_dir: Path,
    *,
    mtl_filename: str = "../mc_model.mtl",
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = _unique_names(parts)
    paths: list[Path] = []
    for part_index, (part, name) in enumerate(zip(parts, names)):
        local_uvs = {
            (0, face_index): face_uvs[(part_index, face_index)]
            for face_index in range(part.face_count)
        }
        path = output_dir / f"{name}.obj"
        write_primitive_obj([part], local_uvs, path, mtl_filename, object_name=name)
        paths.append(path)
    return paths


def _make_material(texture: Image.Image) -> trimesh.visual.material.SimpleMaterial:
    return trimesh.visual.material.SimpleMaterial(
        image=texture.convert("RGBA"),
        diffuse=np.asarray([255, 255, 255, 255], dtype=np.uint8),
    )


def write_primitive_glb(
    parts: Sequence[PrimitivePart],
    face_uvs: dict[tuple[int, int], np.ndarray],
    texture: Image.Image,
    output_path: Path,
    *,
    object_name: str = "paper_model",
) -> None:
    """Write one textured GLB preview with UV-seam vertex duplication."""

    if not parts:
        raise ValueError("No primitive parts were provided for GLB export")
    vertices: list[np.ndarray] = []
    uvs: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    for part_index, part in enumerate(parts):
        part_vertices = np.asarray(part.vertices, dtype=np.float64)
        for face_index, polygon in enumerate(part.polygons):
            base = len(vertices)
            polygon_points = part_vertices[np.asarray(polygon, dtype=np.int64)]
            polygon_uvs = np.asarray(face_uvs[(part_index, face_index)], dtype=np.float64)
            vertices.extend(polygon_points)
            uvs.extend(polygon_uvs)
            for offset in range(1, len(polygon) - 1):
                triangles.append((base, base + offset, base + offset + 1))

    material = _make_material(texture)
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
            "automatic_primitive_fit": True,
            "single_object": True,
            "independent_closed_shells": True,
            "shell_count": int(len(parts)),
            "texture_embedded": True,
            "recommended_blender_input_for_unfolding": "paper_model.obj",
        }
    )
    scene = trimesh.Scene(base_frame="world")
    name = _safe_name(object_name, "paper_model")
    scene.add_geometry(mesh, node_name=name, geom_name=name, transform=np.eye(4, dtype=np.float64))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(scene.export(file_type="glb"))
