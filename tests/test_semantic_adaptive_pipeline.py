from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

from partfield_mc.mesh_io import normalize_meshes, partfield_normalization
from partfield_mc.pipeline import PipelineConfig, run_postprocess


def test_pipeline_exports_face_first_and_split_torso(tmp_path: Path) -> None:
    lower = trimesh.creation.box(extents=[2.0, 0.8, 1.5])
    lower.apply_translation([1.0, 0.4, 0.75])
    upper = trimesh.creation.box(extents=[1.2, 1.2, 1.5])
    upper.apply_translation([0.6, 1.4, 0.75])
    face = trimesh.creation.box(extents=[1.2, 1.3, 1.1])
    face.apply_translation([1.95, 1.6, 0.8])
    tail = trimesh.creation.box(extents=[1.5, 0.5, 0.4])
    tail.apply_translation([-0.55, 1.3, 0.75])
    meshes = [lower, upper, face, tail]

    source_path = tmp_path / "source.glb"
    source_path.write_bytes(trimesh.Scene(meshes).export(file_type="glb"))
    center, scale = partfield_normalization(meshes)
    normalized = normalize_meshes(meshes, center, scale)
    segmented = trimesh.util.concatenate(normalized)
    normalized_path = tmp_path / "normalized.ply"
    segmented.export(normalized_path)
    labels = np.concatenate(
        [
            np.zeros(len(normalized[0].faces), dtype=np.int64),
            np.zeros(len(normalized[1].faces), dtype=np.int64),
            np.ones(len(normalized[2].faces), dtype=np.int64),
            np.full(len(normalized[3].faces), 2, dtype=np.int64),
        ]
    )
    labels_path = tmp_path / "labels.npy"
    np.save(labels_path, labels)

    output_dir = tmp_path / "out"
    result = run_postprocess(
        source_path,
        output_dir,
        normalized_path,
        labels_path,
        PipelineConfig(
            partfield_repo=tmp_path,
            checkpoint=tmp_path / "unused.ckpt",
            clusters=3,
            fit_mode="aabb",
            obj_mode="surface",
            face_resolution=4,
            surface_samples=1000,
            padding=0,
            semantic_refit="animal",
            adaptive_split=True,
            max_extra_cuboids=2,
            protected_min_coverage=0.90,
            split_min_coverage_gain=0.04,
        ),
    )

    data = json.loads((output_dir / "parts_after_surface.json").read_text())
    assert result.part_count == 4
    assert data["requested_cluster_count"] == 3
    assert data["unique_segment_count"] == 3
    assert data["cuboid_count"] == 4
    names = [part["name"] for part in data["parts"]]
    assert "face_head" in names
    assert sum(name.startswith("body_") for name in names) == 2
    face_part = next(part for part in data["parts"] if part["name"] == "face_head")
    assert face_part["metadata"]["protected_visual_region"] is True
