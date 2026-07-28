from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

from partfield_mc.cuboid_fit import FitConfig, fit_cuboids_from_labels
from partfield_mc.exporters import write_glb
from partfield_mc.texture import ColoredSurfacePointCloud, build_texture_atlas


def textured_box() -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=[2.0, 1.0, 1.0])
    uv = np.zeros((len(mesh.vertices), 2), dtype=float)
    vertices = np.asarray(mesh.vertices)
    uv[:, 0] = (vertices[:, 0] - vertices[:, 0].min()) / np.ptp(vertices[:, 0])
    uv[:, 1] = (vertices[:, 1] - vertices[:, 1].min()) / np.ptp(vertices[:, 1])
    image = np.zeros((32, 32, 4), dtype=np.uint8)
    image[..., 0] = np.arange(32, dtype=np.uint8)[None, :] * 8
    image[..., 1] = np.arange(32, dtype=np.uint8)[:, None] * 8
    image[..., 2] = 80
    image[..., 3] = 255
    material = trimesh.visual.material.SimpleMaterial(image=Image.fromarray(image))
    mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)
    return mesh


def test_cuboid_fit_and_export(tmp_path: Path) -> None:
    mesh = textured_box()
    labels = np.zeros(len(mesh.faces), dtype=np.int64)
    centroids = mesh.triangles_center
    labels[centroids[:, 0] > 0] = 1
    parts = fit_cuboids_from_labels(mesh, labels, FitConfig(fit_mode="obb", min_faces=2))
    assert len(parts) == 2
    sampler = ColoredSurfacePointCloud([mesh], surface_samples=5000)
    texture, uv_rects = build_texture_atlas(parts, sampler, face_resolution=8)
    output = tmp_path / "model.glb"
    write_glb(parts, uv_rects, texture, output)
    assert output.exists() and output.stat().st_size > 0
