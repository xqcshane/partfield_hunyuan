from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

from partfield_mc.mesh_io import normalize_meshes, partfield_normalization
from partfield_mc.pipeline import PipelineConfig, run_postprocess


def _positive_overlap(a: dict, b: dict, tol: float = 1e-8) -> bool:
    ac = np.asarray(a["center"], dtype=float)
    bc = np.asarray(b["center"], dtype=float)
    ah = np.asarray(a["size"], dtype=float) * 0.5
    bh = np.asarray(b["size"], dtype=float) * 0.5
    depth = np.minimum(ac + ah, bc + bh) - np.maximum(ac - ah, bc - bh)
    return bool(np.all(depth > tol))


def test_surface_mode_saves_simplified_before_and_after_models(tmp_path: Path) -> None:
    large = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    small = trimesh.creation.box(extents=[1.2, 1.0, 1.0])
    small.apply_translation([0.8, 0.0, 0.0])
    source_scene = trimesh.Scene([large, small])
    source_path = tmp_path / "source.glb"
    source_path.write_bytes(source_scene.export(file_type="glb"))

    center, scale = partfield_normalization([large, small])
    normalized = normalize_meshes([large, small], center, scale)
    segmented = trimesh.util.concatenate(normalized)
    normalized_path = tmp_path / "normalized.ply"
    segmented.export(normalized_path)

    labels = np.concatenate(
        [
            np.zeros(len(normalized[0].faces), dtype=np.int64),
            np.ones(len(normalized[1].faces), dtype=np.int64),
        ]
    )
    labels_path = tmp_path / "labels.npy"
    np.save(labels_path, labels)

    simplified_path = tmp_path / "partfield_input_simplified.glb"
    simplified_path.write_bytes(source_path.read_bytes())
    output_dir = tmp_path / "out"
    config = PipelineConfig(
        partfield_repo=tmp_path,
        checkpoint=tmp_path / "unused.ckpt",
        clusters=2,
        fit_mode="aabb",
        obj_mode="surface",
        face_resolution=4,
        surface_samples=1000,
        padding=0,
    )

    result = run_postprocess(
        source_path,
        output_dir,
        normalized_path,
        labels_path,
        config,
        simplified_mesh_path=simplified_path,
    )

    expected = [
        simplified_path,
        output_dir / "mc_model_raw_aabb.glb",
        output_dir / "mc_model_raw_aabb.obj",
        output_dir / "parts_raw_aabb.json",
        output_dir / "mc_model_before_surface.glb",
        output_dir / "mc_model_before_surface.obj",
        output_dir / "parts_before_surface.json",
        output_dir / "mc_model_after_surface.glb",
        output_dir / "mc_model_after_surface.obj",
        output_dir / "parts_after_surface.json",
        output_dir / "mc_model.glb",
        output_dir / "mc_model.obj",
        output_dir / "paper_model.glb",
        output_dir / "paper_model.obj",
        output_dir / "paper_model.mtl",
        output_dir / "paper_model_texture.png",
        output_dir / "paper_model_parts.json",
    ]
    for path in expected:
        assert path.exists() and path.stat().st_size > 0

    assert Path(result.simplified_mesh) == simplified_path.resolve()
    assert Path(result.before_surface_glb).name == "mc_model_before_surface.glb"
    assert Path(result.after_surface_glb).name == "mc_model_after_surface.glb"
    assert Path(result.paper_model_obj).name == "paper_model.obj"
    assert Path(result.paper_model_glb).name == "paper_model.glb"

    paper_glb_scene = trimesh.load(output_dir / "paper_model.glb", force="scene")
    assert len(paper_glb_scene.geometry) == 1
    paper_glb_mesh = next(iter(paper_glb_scene.geometry.values()))
    assert getattr(paper_glb_mesh.visual, "kind", None) == "texture"
    assert getattr(paper_glb_mesh.visual.material, "baseColorTexture", None) is not None

    raw = json.loads((output_dir / "parts_raw_aabb.json").read_text())
    before = json.loads((output_dir / "parts_before_surface.json").read_text())
    after = json.loads((output_dir / "parts_after_surface.json").read_text())
    paper = json.loads((output_dir / "paper_model_parts.json").read_text())
    assert paper["single_object"] is True
    assert paper["glb_texture_embedded"] is True
    assert paper["independent_closed_shells"] is True
    assert paper["shell_count"] == 2
    assert paper["vertex_count"] == 16
    assert paper["quad_count"] == 12
    assert raw["part_count"] == 2
    assert before["part_count"] == 2
    assert after["part_count"] == 2
    assert _positive_overlap(raw["parts"][0], raw["parts"][1])
    assert not _positive_overlap(before["parts"][0], before["parts"][1])
    assert not _positive_overlap(after["parts"][0], after["parts"][1])

    # Surface processing changes only material/UV treatment, never cuboid bounds.
    before_by_id = {part["segment_id"]: part for part in before["parts"]}
    after_by_id = {part["segment_id"]: part for part in after["parts"]}
    for segment_id in before_by_id:
        assert np.allclose(before_by_id[segment_id]["center"], after_by_id[segment_id]["center"])
        assert np.allclose(before_by_id[segment_id]["size"], after_by_id[segment_id]["size"])
