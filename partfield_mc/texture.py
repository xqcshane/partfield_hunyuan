from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image
import trimesh

from .models import CuboidPart

FACE_NAMES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
FACE_BRIGHTNESS = {
    "+X": 0.94,
    "-X": 0.86,
    "+Y": 1.08,
    "-Y": 0.72,
    "+Z": 1.00,
    "-Z": 0.90,
}


def cuboid_face_corners(part: CuboidPart, face: str) -> np.ndarray:
    sx, sy, sz = np.asarray(part.size, dtype=np.float64)
    x0, x1 = -sx / 2.0, sx / 2.0
    y0, y1 = -sy / 2.0, sy / 2.0
    z0, z1 = -sz / 2.0, sz / 2.0
    local = {
        "+X": [(x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1)],
        "-X": [(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)],
        "+Y": [(x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0)],
        "-Y": [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        "+Z": [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        "-Z": [(x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0)],
    }[face]
    return trimesh.transform_points(np.asarray(local, dtype=np.float64), part.transform)


def face_sample_points(corners: np.ndarray, resolution: int) -> np.ndarray:
    p0, p1, p2, p3 = np.asarray(corners, dtype=np.float64)
    rows = np.arange(resolution, dtype=np.float64)
    cols = np.arange(resolution, dtype=np.float64)
    u, image_v = np.meshgrid((cols + 0.5) / resolution, (rows + 0.5) / resolution)
    v = 1.0 - image_v
    u = u.reshape(-1, 1)
    v = v.reshape(-1, 1)
    return (
        (1.0 - u) * (1.0 - v) * p0
        + u * (1.0 - v) * p1
        + u * v * p2
        + (1.0 - u) * v * p3
    )


@dataclass
class TextureSource:
    uv: np.ndarray
    image: np.ndarray
    factor: np.ndarray


def _as_rgba_image(value: object) -> Image.Image | None:
    if value is None:
        return None
    try:
        if isinstance(value, Image.Image):
            return value.convert("RGBA")
        arr = np.asarray(value)
        if arr.ndim in (2, 3):
            return Image.fromarray(arr).convert("RGBA")
    except Exception:
        return None
    return None


def _extract_texture_source(mesh: trimesh.Trimesh) -> TextureSource | None:
    uv = getattr(mesh.visual, "uv", None)
    if uv is None:
        return None
    uv_array = np.asarray(uv, dtype=np.float64)
    if uv_array.ndim != 2 or uv_array.shape[1] < 2 or len(uv_array) != len(mesh.vertices):
        return None

    material = getattr(mesh.visual, "material", None)
    if material is None:
        return None
    image = _as_rgba_image(getattr(material, "image", None))
    if image is None:
        image = _as_rgba_image(getattr(material, "baseColorTexture", None))
    if image is None:
        return None

    factor_value = getattr(material, "baseColorFactor", None)
    if factor_value is None:
        factor_value = getattr(material, "diffuse", None)
    if factor_value is None:
        factor = np.ones(4, dtype=np.float64)
    else:
        factor = np.asarray(factor_value, dtype=np.float64).reshape(-1)
        factor = np.pad(factor, (0, max(0, 4 - factor.size)), constant_values=1.0)[:4]
        if np.max(factor) > 1.0:
            factor = factor / 255.0
        factor = np.clip(factor, 0.0, 1.0)
    return TextureSource(uv=uv_array[:, :2], image=np.asarray(image, dtype=np.float64), factor=factor)


def _material_color(mesh: trimesh.Trimesh) -> np.ndarray:
    material = getattr(mesh.visual, "material", None)
    if material is not None:
        for key in ("main_color", "diffuse", "baseColorFactor"):
            value = getattr(material, key, None)
            if value is None:
                continue
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
            if arr.size >= 3:
                rgb = arr[:3]
                if np.max(rgb) <= 1.0:
                    rgb *= 255.0
                return np.clip(rgb, 0, 255)
    return np.array([170.0, 170.0, 170.0], dtype=np.float64)


def _vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray | None:
    if getattr(mesh.visual, "kind", None) != "vertex":
        return None
    colors = np.asarray(getattr(mesh.visual, "vertex_colors", []), dtype=np.float64)
    return colors if len(colors) == len(mesh.vertices) else None


def _wrap_uv(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "clamp":
        return np.clip(values, 0.0, 1.0)
    wrapped = np.mod(values, 1.0)
    edge = (values > 0) & np.isclose(wrapped, 0.0, atol=1e-10)
    wrapped[edge] = 1.0
    return wrapped


def _sample_image(source: TextureSource, uv: np.ndarray, filter_mode: str, wrap_mode: str) -> np.ndarray:
    image = source.image
    h, w = image.shape[:2]
    u = _wrap_uv(np.asarray(uv[:, 0], dtype=np.float64), wrap_mode)
    v = _wrap_uv(np.asarray(uv[:, 1], dtype=np.float64), wrap_mode)
    x = u * max(w - 1, 0)
    y = (1.0 - v) * max(h - 1, 0)
    if filter_mode == "nearest" or w == 1 or h == 1:
        xi = np.clip(np.rint(x).astype(np.int64), 0, w - 1)
        yi = np.clip(np.rint(y).astype(np.int64), 0, h - 1)
        rgba = image[yi, xi]
    else:
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        wx = (x - x0)[:, None]
        wy = (y - y0)[:, None]
        top = image[y0, x0] * (1 - wx) + image[y0, x1] * wx
        bottom = image[y1, x0] * (1 - wx) + image[y1, x1] * wx
        rgba = top * (1 - wy) + bottom * wy
    return np.clip(rgba * source.factor[None, :], 0, 255)


class ColoredSurfacePointCloud:
    """Low-memory UV-aware source sampler using a colored surface cloud."""

    def __init__(
        self,
        meshes: Sequence[trimesh.Trimesh],
        surface_samples: int = 500_000,
        texture_filter: str = "bilinear",
        uv_wrap: str = "repeat",
        seed: int = 12345,
    ):
        from scipy.spatial import cKDTree

        if surface_samples < 1000:
            raise ValueError("surface_samples must be >= 1000")
        self.meshes = list(meshes)
        self.texture_sources = [_extract_texture_source(mesh) for mesh in self.meshes]
        self.vertex_colors = [_vertex_colors(mesh) for mesh in self.meshes]
        self.fallback = [_material_color(mesh) for mesh in self.meshes]
        self.texture_filter = texture_filter
        self.uv_wrap = uv_wrap
        self.rng = np.random.default_rng(seed)

        areas = np.asarray([max(float(np.sum(mesh.area_faces)), 1e-12) for mesh in meshes])
        weights = areas / areas.sum()
        counts = np.maximum(500, np.floor(weights * surface_samples).astype(np.int64))

        points_all: list[np.ndarray] = []
        colors_all: list[np.ndarray] = []
        for index, (mesh, count) in enumerate(zip(meshes, counts.tolist())):
            print(f"[texture] sampling source mesh {index + 1}/{len(meshes)}: {count:,} points", flush=True)
            points, face_ids, bary = self._sample_surface(mesh, count)
            colors = self._colors(index, face_ids, bary)
            points_all.append(points.astype(np.float32))
            colors_all.append(colors.astype(np.uint8))

        self.points = np.vstack(points_all)
        self.colors = np.vstack(colors_all)
        print(f"[texture] building KD-tree: {len(self.points):,} points", flush=True)
        self.tree = cKDTree(self.points.astype(np.float64), compact_nodes=True, balanced_tree=True)

    def _sample_surface(self, mesh: trimesh.Trimesh, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        areas = np.asarray(mesh.area_faces, dtype=np.float64)
        valid_ids = np.flatnonzero(np.isfinite(areas) & (areas > 0))
        if len(valid_ids) == 0:
            raise ValueError("Source mesh has no non-degenerate faces")
        probabilities = areas[valid_ids] / areas[valid_ids].sum()
        face_ids = self.rng.choice(valid_ids, size=count, replace=True, p=probabilities)
        tri_ids = np.asarray(mesh.faces[face_ids], dtype=np.int64)
        triangles = np.asarray(mesh.vertices[tri_ids], dtype=np.float64)
        r1 = np.sqrt(self.rng.random(count))
        r2 = self.rng.random(count)
        bary = np.column_stack((1 - r1, r1 * (1 - r2), r1 * r2))
        points = np.einsum("ni,nic->nc", bary, triangles)
        return points, face_ids.astype(np.int64), bary

    def _colors(self, mesh_index: int, face_ids: np.ndarray, bary: np.ndarray) -> np.ndarray:
        mesh = self.meshes[mesh_index]
        tri_vertices = np.asarray(mesh.faces[face_ids], dtype=np.int64)
        texture = self.texture_sources[mesh_index]
        if texture is not None:
            tri_uv = texture.uv[tri_vertices]
            uv = np.einsum("ni,nic->nc", bary, tri_uv)
            rgba = _sample_image(texture, uv, self.texture_filter, self.uv_wrap)
            alpha = rgba[:, 3:4] / 255.0
            return np.clip(rgba[:, :3] * alpha + self.fallback[mesh_index][None, :] * (1 - alpha), 0, 255)
        colors = self.vertex_colors[mesh_index]
        if colors is not None:
            return np.clip(np.einsum("ni,nic->nc", bary, colors[tri_vertices, :3]), 0, 255)
        return np.repeat(self.fallback[mesh_index][None, :], len(face_ids), axis=0)

    def sample(self, query_points: np.ndarray, chunk_size: int = 65_536) -> np.ndarray:
        query_points = np.asarray(query_points, dtype=np.float64)
        output = np.empty((len(query_points), 3), dtype=np.uint8)
        for start in range(0, len(query_points), chunk_size):
            end = min(start + chunk_size, len(query_points))
            _, indices = self.tree.query(query_points[start:end], k=1, workers=-1)
            output[start:end] = self.colors[np.asarray(indices, dtype=np.int64)]
        return output


def build_texture_atlas(
    parts: Sequence[CuboidPart],
    sampler: ColoredSurfacePointCloud,
    face_resolution: int = 64,
    padding: int = 1,
    palette_size: int = 0,
    face_shading: bool = False,
) -> tuple[Image.Image, dict[tuple[int, str], tuple[float, float, float, float]]]:
    if face_resolution < 1:
        raise ValueError("face_resolution must be >= 1")
    stride = face_resolution + 2 * padding
    width = len(FACE_NAMES) * stride
    height = len(parts) * stride
    atlas = np.zeros((height, width, 3), dtype=np.uint8)
    uv_rects: dict[tuple[int, str], tuple[float, float, float, float]] = {}

    total = len(parts) * len(FACE_NAMES)
    done = 0
    for part_index, part in enumerate(parts):
        for face_index, face in enumerate(FACE_NAMES):
            points = face_sample_points(cuboid_face_corners(part, face), face_resolution)
            colors = sampler.sample(points).reshape(face_resolution, face_resolution, 3).astype(np.float32)
            if face_shading:
                colors *= FACE_BRIGHTNESS[face]
            tile = np.clip(colors, 0, 255).astype(np.uint8)

            x0 = face_index * stride + padding
            y0 = part_index * stride + padding
            x1 = x0 + face_resolution
            y1 = y0 + face_resolution
            atlas[y0:y1, x0:x1] = tile
            if padding:
                atlas[y0:y1, x0 - padding:x0] = tile[:, :1]
                atlas[y0:y1, x1:x1 + padding] = tile[:, -1:]
                atlas[y0 - padding:y0, x0 - padding:x1 + padding] = atlas[y0:y0 + 1, x0 - padding:x1 + padding]
                atlas[y1:y1 + padding, x0 - padding:x1 + padding] = atlas[y1 - 1:y1, x0 - padding:x1 + padding]

            uv_rects[(part_index, face)] = (
                x0 / width,
                1.0 - y1 / height,
                x1 / width,
                1.0 - y0 / height,
            )
            done += 1
            print(f"[texture] {done}/{total}: {part.name} {face}", file=sys.stderr, flush=True)

    image = Image.fromarray(atlas, mode="RGB")
    if 2 <= palette_size <= 256:
        image = image.quantize(colors=palette_size, method=Image.Quantize.MEDIANCUT).convert("RGB")
    return image, uv_rects
