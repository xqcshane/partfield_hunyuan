from __future__ import annotations

import copy
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .cuboid_fit import FitConfig, fit_cuboids_from_labels
from .exporters import (
    build_paper_model_texture,
    write_glb,
    write_mtl,
    write_obj,
    write_paper_model_glb,
    write_paper_model_obj,
    write_split_objs,
)
from .mesh_io import (
    load_triangle_mesh,
    load_world_meshes,
    normalize_meshes,
    partfield_normalization,
    simplify_mesh_for_partfield,
)
from .models import CuboidPart, ExportPaths, PartFieldArtifacts
from .partfield_runner import PartFieldRunConfig, run_partfield
from .primitive_exporters import (
    build_primitive_texture_atlas,
    write_primitive_glb,
    write_primitive_obj,
    write_split_primitive_objs,
)
from .primitive_fit import (
    PrimitiveFitConfig,
    fit_primitives_from_labels,
    parse_primitive_types,
)
from .texture import ColoredSurfacePointCloud, build_texture_atlas


@dataclass
class PipelineConfig:
    partfield_repo: Path
    checkpoint: Path
    clusters: int = 10
    clustering: str = "agglo"
    adjacency: str = "mst"
    n_point_per_face: int = 500
    n_sample_each: int = 5000
    preprocess_mesh: bool = False
    up_axis: str = "y"
    fit_mode: str = "obb"
    min_area_ratio: float = 0.0
    min_faces: int = 4
    grid_divisions: int = 0
    category: str = "generic"
    forward_axis: str = "auto"
    resolve_overlaps: bool = False
    part_gap_ratio: float = 0.0
    face_resolution: int = 64
    surface_samples: int = 500_000
    palette_size: int = 0
    texture_filter: str = "bilinear"
    uv_wrap: str = "repeat"
    padding: int = 1
    face_shading: bool = False
    obj_mode: str = "merged"
    surface_fit_strategy: str = "refit"
    refit_min_coverage: float = 0.05
    refit_beam_width: int = 64
    refit_preserve_contact: bool = True
    semantic_refit: str = "auto"
    adaptive_split: bool = True
    max_extra_cuboids: int = 1
    protected_min_coverage: float = 0.85
    split_min_coverage_gain: float = 0.05
    primitive_types: str = "auto"
    primitive_target_faces: int = 0
    primitive_max_faces: int = 48
    primitive_max_sides: int = 24
    primitive_fit_samples: int = 2500
    primitive_complexity_weight: float = 0.025
    primitive_resolve_overlaps: bool = True
    primitive_preserve_contacts: bool = True
    primitive_contact_overlap_ratio: float = 0.0
    primitive_contact_mode: str = "fixed"
    primitive_connector_sides: int = 4
    primitive_connector_radius_ratio: float = 0.028
    primitive_connector_inset_ratio: float = 0.28
    primitive_connector_min_length_ratio: float = 0.002
    primitive_interface_max_sides: int = 8
    primitive_interface_min_width_ratio: float = 0.006
    primitive_interface_plane_tolerance_ratio: float = 1e-6
    primitive_part_mode: str = "auto"
    primitive_patch_min_segment_area_ratio: float = 0.10
    primitive_patch_min_area_balance: float = 0.30
    primitive_patch_min_interface_area_ratio: float = 0.14
    primitive_patch_min_seam_length_ratio: float = 0.75
    primitive_surface_main_body_min_area_ratio: float = 0.35
    primitive_surface_boundary_rings: int = 0
    primitive_surface_search_steps: int = 18
    primitive_surface_min_reduction_ratio: float = 0.15
    primitive_surface_hard_max_faces: int = 512
    primitive_validation_policy: str = "repair"
    primitive_contact_weak_threshold: float = 0.20
    primitive_contact_strong_threshold: float = 0.55
    primitive_contact_min_edge_count: int = 6
    primitive_contact_medium_mode: str = "connector"
    simplify_faces: int = 50_000
    force: bool = False


@dataclass
class PipelineResult:
    output_dir: str
    glb: str
    obj: str
    obj_parts_dir: str | None
    texture: str
    parts_json: str
    normalized_mesh: str
    labels: str
    part_count: int
    job_id: str
    simplified_mesh: str = ""
    raw_aabb_glb: str = ""
    raw_aabb_obj: str = ""
    raw_aabb_parts_json: str = ""
    before_surface_glb: str = ""
    before_surface_obj: str = ""
    before_surface_parts_json: str = ""
    after_surface_glb: str = ""
    after_surface_obj: str = ""
    after_surface_parts_json: str = ""
    paper_model_glb: str = ""
    paper_model_obj: str = ""
    paper_model_texture: str = ""
    paper_model_parts_json: str = ""


def _export_paths(output_dir: Path, segmented_source: Path | None) -> ExportPaths:
    return ExportPaths(
        output_dir=output_dir,
        glb=output_dir / "mc_model.glb",
        obj=output_dir / "mc_model.obj",
        mtl=output_dir / "mc_model.mtl",
        texture=output_dir / "mc_texture.png",
        parts_json=output_dir / "parts.json",
        segmented_ply=(output_dir / "partfield_segmented.ply") if segmented_source else None,
    )


def _resolved_simplified_mesh(
    output_dir: Path,
    simplified_mesh_path: str | Path | None,
) -> str:
    if simplified_mesh_path is not None:
        candidate = Path(simplified_mesh_path).expanduser().resolve()
        if candidate.exists():
            return str(candidate)
    cached = output_dir / "partfield_input_simplified.glb"
    if cached.exists():
        return str(cached.resolve())
    return ""


def _run_primitive_postprocess(
    *,
    input_path: Path,
    output_dir: Path,
    normalized_mesh_path: Path,
    labels_path: Path,
    normalized_sources: list,
    segmented_mesh,
    labels: np.ndarray,
    center: np.ndarray,
    scale: float,
    config: PipelineConfig,
    artifacts: PartFieldArtifacts | None,
    simplified_mesh_path: str | Path | None,
) -> PipelineResult:
    """Automatic per-cluster primitive fitting and paper-model export."""

    preserve_all_clusters = config.obj_mode in {"surface", "all"}
    primitive_config = PrimitiveFitConfig(
        min_area_ratio=0.0 if preserve_all_clusters else config.min_area_ratio,
        min_faces=1 if preserve_all_clusters else config.min_faces,
        target_faces=config.primitive_target_faces,
        max_faces=config.primitive_max_faces,
        max_sides=config.primitive_max_sides,
        fit_samples=config.primitive_fit_samples,
        complexity_weight=config.primitive_complexity_weight,
        allowed_types=parse_primitive_types(config.primitive_types),
        resolve_overlaps=config.primitive_resolve_overlaps,
        overlap_gap_ratio=config.part_gap_ratio,
        preserve_contacts=config.primitive_preserve_contacts,
        contact_overlap_ratio=config.primitive_contact_overlap_ratio,
        contact_mode=config.primitive_contact_mode,
        connector_sides=config.primitive_connector_sides,
        connector_radius_ratio=config.primitive_connector_radius_ratio,
        connector_inset_ratio=config.primitive_connector_inset_ratio,
        connector_min_length_ratio=config.primitive_connector_min_length_ratio,
        interface_max_sides=config.primitive_interface_max_sides,
        interface_min_width_ratio=config.primitive_interface_min_width_ratio,
        interface_plane_tolerance_ratio=config.primitive_interface_plane_tolerance_ratio,
        part_mode=config.primitive_part_mode,
        patch_min_segment_area_ratio=config.primitive_patch_min_segment_area_ratio,
        patch_min_area_balance=config.primitive_patch_min_area_balance,
        patch_min_interface_area_ratio=config.primitive_patch_min_interface_area_ratio,
        patch_min_seam_length_ratio=config.primitive_patch_min_seam_length_ratio,
        surface_main_body_min_area_ratio=config.primitive_surface_main_body_min_area_ratio,
        surface_boundary_rings=config.primitive_surface_boundary_rings,
        surface_search_steps=config.primitive_surface_search_steps,
        surface_min_reduction_ratio=config.primitive_surface_min_reduction_ratio,
        surface_hard_max_faces=config.primitive_surface_hard_max_faces,
        validation_policy=config.primitive_validation_policy,
        contact_weak_threshold=config.primitive_contact_weak_threshold,
        contact_strong_threshold=config.primitive_contact_strong_threshold,
        contact_min_edge_count=config.primitive_contact_min_edge_count,
        contact_medium_mode=config.primitive_contact_medium_mode,
        category=config.category,
        forward_axis=config.forward_axis,
    )
    parts = fit_primitives_from_labels(segmented_mesh, labels, primitive_config)

    sampler = ColoredSurfacePointCloud(
        normalized_sources,
        surface_samples=config.surface_samples,
        texture_filter=config.texture_filter,
        uv_wrap=config.uv_wrap,
    )
    texture, face_uvs = build_primitive_texture_atlas(
        parts,
        sampler,
        face_resolution=config.face_resolution,
        padding=config.padding,
        palette_size=config.palette_size,
        face_shading=config.face_shading,
    )

    segmented_source = artifacts.colored_segmentation if artifacts else None
    paths = _export_paths(output_dir, segmented_source)
    texture.save(paths.texture)
    write_mtl(paths.mtl, paths.texture.name)
    write_primitive_glb(parts, face_uvs, texture, paths.glb, object_name="mc_model")

    obj_parts_dir: Path | None = None
    if config.obj_mode in {"merged", "surface", "all"}:
        write_primitive_obj(parts, face_uvs, paths.obj, paths.mtl.name, object_name="mc_model")
    if config.obj_mode in {"separate", "all"}:
        obj_parts_dir = output_dir / "obj_parts"
        write_split_primitive_objs(parts, face_uvs, obj_parts_dir)
    if config.obj_mode == "all":
        write_primitive_obj(
            parts,
            face_uvs,
            output_dir / "mc_model_surface.obj",
            paths.mtl.name,
            object_name="mc_model_surface",
        )
    if config.obj_mode not in {"merged", "surface", "separate", "all"}:
        raise ValueError(f"Unsupported obj_mode: {config.obj_mode}")

    paper_requested = config.obj_mode in {"surface", "all"}
    paper_model_glb = output_dir / "paper_model.glb"
    paper_model_obj = output_dir / "paper_model.obj"
    paper_model_mtl = output_dir / "paper_model.mtl"
    paper_model_texture = output_dir / "paper_model_texture.png"
    paper_model_parts_json = output_dir / "paper_model_parts.json"
    paper_shells: list[dict[str, object]] = []
    if paper_requested:
        texture.save(paper_model_texture)
        write_mtl(paper_model_mtl, paper_model_texture.name)
        paper_shells = write_primitive_obj(
            parts,
            face_uvs,
            paper_model_obj,
            paper_model_mtl.name,
            object_name="paper_model",
        )
        write_primitive_glb(
            parts,
            face_uvs,
            texture,
            paper_model_glb,
            object_name="paper_model",
        )
        allowed_separated_edges = next(
            (
                part.metadata.get("allowed_separated_contact_edges", [])
                for part in parts
                if part.metadata.get("allowed_separated_contact_edges") is not None
            ),
            [],
        )
        required_contact_graph_connected = bool(
            all(
                part.metadata.get("contact_required_graph_connected",
                                  part.metadata.get("contact_graph_connected")) is True
                for part in parts
            )
        )
        spatially_connected_assembly = bool(
            required_contact_graph_connected and not allowed_separated_edges
        )
        paper_model_parts_json.write_text(
            json.dumps(
                {
                    "stage": "paper_model_primitive_fit",
                    "description": (
                        "Hybrid primitive fitting first classifies broad seams between similarly sized "
                        "PartField regions as internal surface-patch seams and merges those regions into one "
                        "physical body. Narrow seams remain independent paper parts. Each resulting body is "
                        "fitted as one closed low-face primitive, while the remaining source seams are "
                        "reconstructed as immutable shared interface polygons. This prevents one apple body "
                        "from becoming several complete ellipsoids while preserving fox heads, legs, ears, "
                        "and tails as separate closed paper shells."
                    ),
                    "recommended_blender_input": paper_model_obj.name,
                    "textured_preview_glb": paper_model_glb.name,
                    "object_name": "paper_model",
                    "single_object": True,
                    "boolean_union": False,
                    "cross_shell_vertex_welding": False,
                    "independent_closed_shells": True,
                    "spatially_connected_assembly": spatially_connected_assembly,
                    "required_contact_graph_connected": required_contact_graph_connected,
                    "allowed_separated_contact_edges": allowed_separated_edges,
                    "contact_method": (
                        "In auto contact mode, source seams are classified by scale-normalised interface "
                        "area, seam length, edge count, and boundary point count. Strong seams use fixed "
                        "interfaces, medium seams use stationary connector patches by default, and weak "
                        "seams are allowed to separate without forcing an intersection."
                    ),
                    "primitive_part_mode": str(config.primitive_part_mode),
                    "surface_patch_groups": [
                        part.metadata.get("source_segment_ids", [int(part.segment_id)])
                        for part in parts
                        if not bool(part.metadata.get("connector_part"))
                    ],
                    "source_part_count": int(
                        sum(not bool(part.metadata.get("connector_part")) for part in parts)
                    ),
                    "connector_count": int(
                        sum(bool(part.metadata.get("connector_part")) for part in parts)
                    ),
                    "shell_count": len(parts),
                    "paper_face_count": int(sum(part.face_count for part in parts)),
                    "triangle_count": int(sum(part.triangle_count for part in parts)),
                    "shells": paper_shells,
                    "parts": [part.to_json() for part in parts],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[PrimitivePaper] Saved canonical Blender OBJ: {paper_model_obj}", flush=True)
        print(f"[PrimitivePaper] Saved textured preview GLB: {paper_model_glb}", flush=True)

    if segmented_source and paths.segmented_ply:
        shutil.copy2(segmented_source, paths.segmented_ply)
    simplified_mesh = _resolved_simplified_mesh(output_dir, simplified_mesh_path)

    metadata = {
        "source": str(input_path),
        "partfield": {
            "normalized_mesh": str(normalized_mesh_path),
            "labels": str(labels_path),
            "job_id": artifacts.job_id if artifacts else None,
            "clusters_requested": config.clusters,
            "clustering": config.clustering,
            "adjacency": config.adjacency,
            "class_agnostic": True,
            "semantic_names_are_heuristic": config.category != "generic",
        },
        "normalization": {"center": center.tolist(), "scale": float(scale)},
        "artifacts": {
            "simplified_mesh": simplified_mesh or None,
            "paper_model": (
                {
                    "glb": str(paper_model_glb),
                    "obj": str(paper_model_obj),
                    "mtl": str(paper_model_mtl),
                    "texture": str(paper_model_texture),
                    "parts_json": str(paper_model_parts_json),
                    "recommended_blender_input": str(paper_model_obj),
                    "single_object": True,
                    "independent_closed_shells": True,
                    "spatially_connected_assembly": spatially_connected_assembly,
                    "required_contact_graph_connected": required_contact_graph_connected,
                    "allowed_separated_contact_edges": allowed_separated_edges,
                    "source_part_count": int(
                        sum(not bool(part.metadata.get("connector_part")) for part in parts)
                    ),
                    "connector_count": int(
                        sum(bool(part.metadata.get("connector_part")) for part in parts)
                    ),
                }
                if paper_requested
                else None
            ),
        },
        "conversion": {
            "fit_mode": "primitive",
            "primitive_types": list(primitive_config.allowed_types),
            "primitive_target_faces": int(config.primitive_target_faces),
            "primitive_target_faces_auto": config.primitive_target_faces <= 0,
            "primitive_max_faces": int(config.primitive_max_faces),
            "primitive_max_sides": int(config.primitive_max_sides),
            "primitive_fit_samples": int(config.primitive_fit_samples),
            "primitive_complexity_weight": float(config.primitive_complexity_weight),
            "primitive_resolve_overlaps": bool(config.primitive_resolve_overlaps),
            "primitive_preserve_contacts": bool(config.primitive_preserve_contacts),
            "primitive_contact_overlap_ratio": float(config.primitive_contact_overlap_ratio),
            "primitive_contact_mode": str(config.primitive_contact_mode),
            "primitive_connector_sides": int(config.primitive_connector_sides),
            "primitive_connector_radius_ratio": float(
                config.primitive_connector_radius_ratio
            ),
            "primitive_connector_inset_ratio": float(
                config.primitive_connector_inset_ratio
            ),
            "primitive_connector_min_length_ratio": float(
                config.primitive_connector_min_length_ratio
            ),
            "primitive_interface_max_sides": int(config.primitive_interface_max_sides),
            "primitive_interface_min_width_ratio": float(
                config.primitive_interface_min_width_ratio
            ),
            "primitive_interface_plane_tolerance_ratio": float(
                config.primitive_interface_plane_tolerance_ratio
            ),
            "primitive_part_mode": str(config.primitive_part_mode),
            "primitive_patch_min_segment_area_ratio": float(
                config.primitive_patch_min_segment_area_ratio
            ),
            "primitive_patch_min_area_balance": float(
                config.primitive_patch_min_area_balance
            ),
            "primitive_patch_min_interface_area_ratio": float(
                config.primitive_patch_min_interface_area_ratio
            ),
            "primitive_patch_min_seam_length_ratio": float(
                config.primitive_patch_min_seam_length_ratio
            ),
            "primitive_surface_main_body_min_area_ratio": float(
                config.primitive_surface_main_body_min_area_ratio
            ),
            "primitive_surface_boundary_rings": int(
                config.primitive_surface_boundary_rings
            ),
            "primitive_surface_search_steps": int(
                config.primitive_surface_search_steps
            ),
            "primitive_surface_min_reduction_ratio": float(
                config.primitive_surface_min_reduction_ratio
            ),
            "primitive_surface_hard_max_faces": int(
                config.primitive_surface_hard_max_faces
            ),
            "primitive_validation_policy": str(config.primitive_validation_policy),
            "primitive_contact_weak_threshold": float(
                config.primitive_contact_weak_threshold
            ),
            "primitive_contact_strong_threshold": float(
                config.primitive_contact_strong_threshold
            ),
            "primitive_contact_min_edge_count": int(
                config.primitive_contact_min_edge_count
            ),
            "primitive_contact_medium_mode": str(
                config.primitive_contact_medium_mode
            ),
            "surface_patch_groups": [
                part.metadata.get("source_segment_ids", [int(part.segment_id)])
                for part in parts
                if not bool(part.metadata.get("connector_part"))
            ],
            "part_gap_ratio": float(config.part_gap_ratio),
            "face_resolution": int(config.face_resolution),
            "surface_samples": int(config.surface_samples),
            "palette_size": int(config.palette_size),
            "texture_filter": config.texture_filter,
            "uv_wrap": config.uv_wrap,
            "face_shading": bool(config.face_shading),
            "obj_mode": config.obj_mode,
            "paper_safe_closed_shells": True,
            "preserve_all_partfield_clusters": preserve_all_clusters,
        },
        "parts": [part.to_json() for part in parts],
    }
    paths.parts_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return PipelineResult(
        output_dir=str(output_dir),
        glb=str(paths.glb),
        obj=str(paths.obj) if paths.obj.exists() else "",
        obj_parts_dir=str(obj_parts_dir) if obj_parts_dir else None,
        texture=str(paths.texture),
        parts_json=str(paths.parts_json),
        normalized_mesh=str(normalized_mesh_path),
        labels=str(labels_path),
        part_count=sum(not bool(part.metadata.get("connector_part")) for part in parts),
        job_id=artifacts.job_id if artifacts else "postprocess_only",
        simplified_mesh=simplified_mesh,
        paper_model_glb=str(paper_model_glb) if paper_requested else "",
        paper_model_obj=str(paper_model_obj) if paper_requested else "",
        paper_model_texture=str(paper_model_texture) if paper_requested else "",
        paper_model_parts_json=str(paper_model_parts_json) if paper_requested else "",
    )


def run_postprocess(
    input_path: str | Path,
    output_dir: str | Path,
    normalized_mesh_path: str | Path,
    labels_path: str | Path,
    config: PipelineConfig,
    artifacts: PartFieldArtifacts | None = None,
    simplified_mesh_path: str | Path | None = None,
) -> PipelineResult:
    """Convert existing PartField labels into textured MC cuboids."""

    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_mesh_path = Path(normalized_mesh_path).expanduser().resolve()
    labels_path = Path(labels_path).expanduser().resolve()

    # The canonical source passed to PartField is Y-up. Reproduce that same
    # source transform and PartField's exact normalization for texture baking.
    source_meshes = load_world_meshes(input_path, up_axis=config.up_axis)
    center, scale = partfield_normalization(source_meshes)
    normalized_sources = normalize_meshes(source_meshes, center, scale)

    segmented_mesh = load_triangle_mesh(normalized_mesh_path)
    labels = np.load(labels_path)
    if config.fit_mode == "primitive":
        return _run_primitive_postprocess(
            input_path=input_path,
            output_dir=output_dir,
            normalized_mesh_path=normalized_mesh_path,
            labels_path=labels_path,
            normalized_sources=normalized_sources,
            segmented_mesh=segmented_mesh,
            labels=labels,
            center=center,
            scale=scale,
            config=config,
            artifacts=artifacts,
            simplified_mesh_path=simplified_mesh_path,
        )
    strict_cluster_surface = config.obj_mode in {"surface", "all"}
    if strict_cluster_surface and config.fit_mode not in {"aabb", "shared"}:
        raise ValueError(
            "--obj-mode surface requires --fit-mode aabb or shared so every cluster can be "
            "made into a non-overlapping closed cuboid"
        )
    raw_aabb_parts: list[CuboidPart] | None = None
    before_surface_parts: list[CuboidPart] | None = None

    if strict_cluster_surface:
        # Stage 1: direct one-box-per-cluster AABBs. This diagnostic artifact
        # intentionally preserves the original overlaps.
        raw_config = FitConfig(
            fit_mode=config.fit_mode,
            min_area_ratio=0.0,
            min_faces=1,
            grid_divisions=config.grid_divisions,
            category=config.category,
            forward_axis=config.forward_axis,
            resolve_overlaps=False,
            part_gap_ratio=0.0,
            overlap_strategy="refit",
            preserve_all_labels=True,
            expected_parts=config.clusters,
        )
        raw_aabb_parts = fit_cuboids_from_labels(segmented_mesh, labels, raw_config)
        for part in raw_aabb_parts:
            part.metadata["surface_stage"] = "raw_aabb"
            part.metadata["raw_aabb_may_overlap"] = True

        # Stage 2: refit from each cluster's own source faces while enforcing
        # zero positive-volume overlap. Adjacent source labels are kept in
        # face-to-face contact whenever a valid candidate exists.
        fit_config = FitConfig(
            fit_mode=config.fit_mode,
            min_area_ratio=0.0,
            min_faces=1,
            grid_divisions=config.grid_divisions,
            category=config.category,
            forward_axis=config.forward_axis,
            resolve_overlaps=True,
            part_gap_ratio=config.part_gap_ratio,
            overlap_strategy=config.surface_fit_strategy,
            preserve_all_labels=True,
            expected_parts=config.clusters,
            refit_min_coverage=config.refit_min_coverage,
            refit_beam_width=config.refit_beam_width,
            refit_preserve_contact=config.refit_preserve_contact,
            semantic_refit=config.semantic_refit,
            adaptive_split=config.adaptive_split,
            max_extra_cuboids=config.max_extra_cuboids,
            protected_min_coverage=config.protected_min_coverage,
            split_min_coverage_gain=config.split_min_coverage_gain,
        )
        parts = fit_cuboids_from_labels(segmented_mesh, labels, fit_config)

        # Keep semantic names tied to PartField segment IDs, regardless of
        # which small segments the constrained fit has to drop.
        names_by_segment = {part.segment_id: part.name for part in raw_aabb_parts}
        for part in parts:
            semantic_name = part.metadata.get("semantic_name")
            if semantic_name:
                part.name = str(semantic_name)
            else:
                part.name = names_by_segment.get(part.segment_id, part.name)
            part.metadata["surface_stage"] = "constrained_before_surface"

        # Stage 3 uses exactly the same cuboid bounds. Surface export changes
        # only the UV/material assignment on contact rectangles, never geometry.
        before_surface_parts = copy.deepcopy(parts)
        for part in before_surface_parts:
            part.metadata["surface_stage"] = "before_surface"
        for part in parts:
            part.metadata["surface_stage"] = "after_surface"
    else:
        fit_config = FitConfig(
            fit_mode=config.fit_mode,
            min_area_ratio=config.min_area_ratio,
            min_faces=config.min_faces,
            grid_divisions=config.grid_divisions,
            category=config.category,
            forward_axis=config.forward_axis,
            resolve_overlaps=config.resolve_overlaps,
            part_gap_ratio=config.part_gap_ratio,
            overlap_strategy="move",
        )
        parts = fit_cuboids_from_labels(segmented_mesh, labels, fit_config)

    raw_segment_ids = (
        sorted(int(part.segment_id) for part in raw_aabb_parts)
        if raw_aabb_parts is not None
        else []
    )
    after_surface_segment_ids = sorted(int(part.segment_id) for part in parts)
    dropped_surface_segment_ids = sorted(
        set(raw_segment_ids) - set(after_surface_segment_ids)
    )

    sampler = ColoredSurfacePointCloud(
        normalized_sources,
        surface_samples=config.surface_samples,
        texture_filter=config.texture_filter,
        uv_wrap=config.uv_wrap,
    )
    texture, uv_rects = build_texture_atlas(
        parts,
        sampler,
        face_resolution=config.face_resolution,
        padding=config.padding,
        palette_size=config.palette_size,
        face_shading=config.face_shading,
    )

    segmented_source = artifacts.colored_segmentation if artifacts else None
    paths = _export_paths(output_dir, segmented_source)

    raw_aabb_glb = output_dir / "mc_model_raw_aabb.glb"
    raw_aabb_obj = output_dir / "mc_model_raw_aabb.obj"
    raw_aabb_mtl = output_dir / "mc_model_raw_aabb.mtl"
    raw_aabb_texture = output_dir / "mc_texture_raw_aabb.png"
    raw_aabb_parts_json = output_dir / "parts_raw_aabb.json"
    before_surface_glb = output_dir / "mc_model_before_surface.glb"
    before_surface_obj = output_dir / "mc_model_before_surface.obj"
    before_surface_mtl = output_dir / "mc_model_before_surface.mtl"
    before_surface_texture = output_dir / "mc_texture_before_surface.png"
    before_surface_parts_json = output_dir / "parts_before_surface.json"
    after_surface_glb = output_dir / "mc_model_after_surface.glb"
    after_surface_obj = output_dir / "mc_model_after_surface.obj"
    after_surface_mtl = output_dir / "mc_model_after_surface.mtl"
    after_surface_texture = output_dir / "mc_texture_after_surface.png"
    after_surface_parts_json = output_dir / "parts_after_surface.json"
    paper_model_glb = output_dir / "paper_model.glb"
    paper_model_obj = output_dir / "paper_model.obj"
    paper_model_mtl = output_dir / "paper_model.mtl"
    paper_model_texture = output_dir / "paper_model_texture.png"
    paper_model_parts_json = output_dir / "paper_model_parts.json"

    if raw_aabb_parts is not None:
        raw_texture, raw_uv_rects = build_texture_atlas(
            raw_aabb_parts,
            sampler,
            face_resolution=config.face_resolution,
            padding=config.padding,
            palette_size=config.palette_size,
            face_shading=config.face_shading,
        )
        raw_texture.save(raw_aabb_texture)
        write_glb(raw_aabb_parts, raw_uv_rects, raw_texture, raw_aabb_glb)
        write_mtl(raw_aabb_mtl, raw_aabb_texture.name)
        write_obj(
            raw_aabb_parts,
            raw_uv_rects,
            raw_aabb_obj,
            raw_aabb_mtl.name,
            surface_only=False,
        )
        raw_aabb_parts_json.write_text(
            json.dumps(
                {
                    "stage": "raw_aabb",
                    "description": (
                        "Direct one-AABB-per-PartField-cluster fit. Cuboids are "
                        "closed and preserve all requested labels, but may overlap."
                    ),
                    "requested_part_count": int(config.clusters),
                    "part_count": len(raw_aabb_parts),
                    "parts": [part.to_json() for part in raw_aabb_parts],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[Refit] Saved raw overlapping AABBs: {raw_aabb_glb}", flush=True)

    if before_surface_parts is not None:
        before_texture, before_uv_rects = build_texture_atlas(
            before_surface_parts,
            sampler,
            face_resolution=config.face_resolution,
            padding=config.padding,
            palette_size=config.palette_size,
            face_shading=config.face_shading,
        )
        before_texture.save(before_surface_texture)
        write_glb(before_surface_parts, before_uv_rects, before_texture, before_surface_glb)
        write_mtl(before_surface_mtl, before_surface_texture.name)
        write_obj(
            before_surface_parts,
            before_uv_rects,
            before_surface_obj,
            before_surface_mtl.name,
            surface_only=False,
        )
        before_surface_parts_json.write_text(
            json.dumps(
                {
                    "stage": "before_surface",
                    "description": (
                        "Non-overlapping constrained AABB refit before surface material processing. "
                        "Geometry is already final; adjacent source segments are kept in face-to-face "
                        "contact whenever feasible."
                    ),
                    "requested_part_count": int(config.clusters),
                    "requested_cluster_count": int(config.clusters),
                    "part_count": len(before_surface_parts),
                    "cuboid_count": len(before_surface_parts),
                    "unique_segment_count": len({int(part.segment_id) for part in before_surface_parts}),
                    "dropped_segment_ids": dropped_surface_segment_ids,
                    "parts": [part.to_json() for part in before_surface_parts],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[Surface] Saved before-surface model: {before_surface_glb}", flush=True)
        print(f"[Surface] Saved before-surface OBJ: {before_surface_obj}", flush=True)

    texture.save(paths.texture)
    write_glb(parts, uv_rects, texture, paths.glb, surface_only=strict_cluster_surface)
    write_mtl(paths.mtl, paths.texture.name)
    obj_parts_dir: Path | None = None
    if config.obj_mode == "merged":
        write_obj(parts, uv_rects, paths.obj, paths.mtl.name, surface_only=False)
    elif config.obj_mode == "surface":
        write_obj(parts, uv_rects, paths.obj, paths.mtl.name, surface_only=True)
    elif config.obj_mode == "separate":
        obj_parts_dir = output_dir / "obj_parts"
        write_split_objs(parts, uv_rects, obj_parts_dir)
    elif config.obj_mode == "all":
        write_obj(parts, uv_rects, paths.obj, paths.mtl.name, surface_only=False)
        write_obj(parts, uv_rects, output_dir / "mc_model_surface.obj", paths.mtl.name, surface_only=True)
        obj_parts_dir = output_dir / "obj_parts"
        write_split_objs(parts, uv_rects, obj_parts_dir)
    else:
        raise ValueError(f"Unsupported obj_mode: {config.obj_mode}")
    if strict_cluster_surface:
        # Keep the legacy mc_model.* names as the final result, and also create
        # explicit after-surface filenames for side-by-side inspection.
        texture.save(after_surface_texture)
        shutil.copy2(paths.glb, after_surface_glb)
        write_mtl(after_surface_mtl, after_surface_texture.name)
        write_obj(
            parts,
            uv_rects,
            after_surface_obj,
            after_surface_mtl.name,
            surface_only=True,
        )
        after_surface_parts_json.write_text(
            json.dumps(
                {
                    "stage": "after_surface",
                    "description": (
                        "Same constrained-refit cuboid geometry as before_surface. Surface processing "
                        "does not trim or move parts; it only leaves exact face-to-face contact "
                        "rectangles untextured while visible regions retain UVs."
                    ),
                    "requested_part_count": int(config.clusters),
                    "requested_cluster_count": int(config.clusters),
                    "part_count": len(parts),
                    "cuboid_count": len(parts),
                    "unique_segment_count": len({int(part.segment_id) for part in parts}),
                    "kept_segment_ids": after_surface_segment_ids,
                    "dropped_segment_ids": dropped_surface_segment_ids,
                    "parts": [part.to_json() for part in parts],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[Surface] Saved after-surface model: {after_surface_glb}", flush=True)
        print(f"[Surface] Saved after-surface OBJ: {after_surface_obj}", flush=True)

        # Stage 4: canonical Blender/Paper Model artifact.  All final cuboids are
        # stored in one OBJ object, but each remains an independent watertight
        # shell with eight shared geometry vertices and six quad faces.  Contact
        # regions are blanked in the atlas rather than splitting cuboid faces.
        paper_texture, paper_texture_stats = build_paper_model_texture(
            parts,
            uv_rects,
            texture,
        )
        paper_texture.save(paper_model_texture)
        write_mtl(paper_model_mtl, paper_model_texture.name)
        paper_shells = write_paper_model_obj(
            parts,
            uv_rects,
            paper_model_obj,
            paper_model_mtl.name,
        )
        write_paper_model_glb(parts, uv_rects, paper_texture, paper_model_glb)
        paper_model_parts_json.write_text(
            json.dumps(
                {
                    "stage": "paper_model",
                    "description": (
                        "Canonical Blender paper-unfolding model. All cuboids are merged into one "
                        "OBJ object without Boolean union or cross-cuboid vertex welding. Each "
                        "cuboid remains an independent closed shell with 8 vertices, 12 edges and "
                        "6 quad faces. Contact rectangles are painted paper-white in the texture "
                        "atlas so no face subdivision is required."
                    ),
                    "recommended_blender_input": paper_model_obj.name,
                    "textured_preview_glb": paper_model_glb.name,
                    "glb_texture_embedded": True,
                    "object_name": "paper_model",
                    "single_object": True,
                    "boolean_union": False,
                    "cross_shell_vertex_welding": False,
                    "independent_closed_shells": True,
                    "shell_count": len(parts),
                    "vertex_count": len(parts) * 8,
                    "edge_count": len(parts) * 12,
                    "quad_count": len(parts) * 6,
                    "triangles_in_obj": 0,
                    "geometry_matches_after_surface": True,
                    "contact_texture_mask": paper_texture_stats,
                    "shells": paper_shells,
                    "parts": [part.to_json() for part in parts],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[Paper] Saved canonical Blender OBJ: {paper_model_obj}", flush=True)
        print(f"[Paper] Saved textured preview GLB: {paper_model_glb}", flush=True)

    if segmented_source and paths.segmented_ply:
        shutil.copy2(segmented_source, paths.segmented_ply)

    simplified_mesh = ""
    if simplified_mesh_path is not None:
        candidate = Path(simplified_mesh_path).expanduser().resolve()
        if candidate.exists():
            simplified_mesh = str(candidate)
    elif (output_dir / "partfield_input_simplified.glb").exists():
        simplified_mesh = str((output_dir / "partfield_input_simplified.glb").resolve())

    metadata = {
        "source": str(input_path),
        "partfield": {
            "normalized_mesh": str(normalized_mesh_path),
            "labels": str(labels_path),
            "job_id": artifacts.job_id if artifacts else None,
            "clusters_requested": config.clusters,
            "clustering": config.clustering,
            "adjacency": config.adjacency,
            "class_agnostic": True,
            "semantic_names_are_heuristic": config.category != "generic",
        },
        "normalization": {"center": center.tolist(), "scale": float(scale)},
        "artifacts": {
            "simplified_mesh": simplified_mesh or None,
            "raw_aabb": (
                {
                    "glb": str(raw_aabb_glb),
                    "obj": str(raw_aabb_obj),
                    "mtl": str(raw_aabb_mtl),
                    "texture": str(raw_aabb_texture),
                    "parts_json": str(raw_aabb_parts_json),
                }
                if raw_aabb_parts is not None
                else None
            ),
            "before_surface": (
                {
                    "glb": str(before_surface_glb),
                    "obj": str(before_surface_obj),
                    "mtl": str(before_surface_mtl),
                    "texture": str(before_surface_texture),
                    "parts_json": str(before_surface_parts_json),
                }
                if before_surface_parts is not None
                else None
            ),
            "after_surface": (
                {
                    "glb": str(after_surface_glb),
                    "obj": str(after_surface_obj),
                    "mtl": str(after_surface_mtl),
                    "texture": str(after_surface_texture),
                    "parts_json": str(after_surface_parts_json),
                    "legacy_glb": str(paths.glb),
                    "legacy_obj": str(paths.obj) if paths.obj.exists() else None,
                    "kept_segment_ids": after_surface_segment_ids,
                    "dropped_segment_ids": dropped_surface_segment_ids,
                }
                if strict_cluster_surface
                else None
            ),
            "paper_model": (
                {
                    "glb": str(paper_model_glb),
                    "obj": str(paper_model_obj),
                    "mtl": str(paper_model_mtl),
                    "texture": str(paper_model_texture),
                    "parts_json": str(paper_model_parts_json),
                    "recommended_blender_input": str(paper_model_obj),
                    "single_object": True,
                    "independent_closed_shells": True,
                }
                if strict_cluster_surface
                else None
            ),
        },
        "conversion": {
            "fit_mode": config.fit_mode,
            "grid_divisions": config.grid_divisions,
            "category": config.category,
            "forward_axis": config.forward_axis,
            "resolve_overlaps": strict_cluster_surface or config.resolve_overlaps,
            "overlap_strategy": (
                config.surface_fit_strategy
                if strict_cluster_surface
                else ("move" if config.resolve_overlaps else "none")
            ),
            "refit_min_coverage": config.refit_min_coverage if strict_cluster_surface else None,
            "refit_beam_width": config.refit_beam_width if strict_cluster_surface else None,
            "refit_preserve_contact": config.refit_preserve_contact if strict_cluster_surface else None,
            "semantic_refit": config.semantic_refit if strict_cluster_surface else None,
            "adaptive_split": config.adaptive_split if strict_cluster_surface else None,
            "max_extra_cuboids": config.max_extra_cuboids if strict_cluster_surface else None,
            "protected_min_coverage": config.protected_min_coverage if strict_cluster_surface else None,
            "split_min_coverage_gain": config.split_min_coverage_gain if strict_cluster_surface else None,
            "strict_cluster_surface": strict_cluster_surface,
            "surface_requested_part_count": int(config.clusters) if strict_cluster_surface else None,
            "surface_kept_part_count": len(parts) if strict_cluster_surface else None,
            "surface_unique_segment_count": (
                len({int(part.segment_id) for part in parts}) if strict_cluster_surface else None
            ),
            "surface_dropped_segment_ids": dropped_surface_segment_ids if strict_cluster_surface else [],
            "part_gap_ratio": config.part_gap_ratio,
            "face_resolution": config.face_resolution,
            "surface_samples": config.surface_samples,
            "palette_size": config.palette_size,
            "texture_filter": config.texture_filter,
            "uv_wrap": config.uv_wrap,
            "face_shading": config.face_shading,
            "obj_mode": config.obj_mode,
        },
        "parts_raw_aabb": (
            [part.to_json() for part in raw_aabb_parts]
            if raw_aabb_parts is not None
            else None
        ),
        "parts_before_surface": (
            [part.to_json() for part in before_surface_parts]
            if before_surface_parts is not None
            else None
        ),
        "parts": [part.to_json() for part in parts],
    }
    paths.parts_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return PipelineResult(
        output_dir=str(output_dir),
        glb=str(paths.glb),
        obj=str(paths.obj) if paths.obj.exists() else "",
        obj_parts_dir=str(obj_parts_dir) if obj_parts_dir else None,
        texture=str(paths.texture),
        parts_json=str(paths.parts_json),
        normalized_mesh=str(normalized_mesh_path),
        labels=str(labels_path),
        part_count=len(parts),
        job_id=artifacts.job_id if artifacts else "postprocess_only",
        simplified_mesh=simplified_mesh,
        raw_aabb_glb=str(raw_aabb_glb) if raw_aabb_parts is not None else "",
        raw_aabb_obj=str(raw_aabb_obj) if raw_aabb_parts is not None else "",
        raw_aabb_parts_json=(
            str(raw_aabb_parts_json) if raw_aabb_parts is not None else ""
        ),
        before_surface_glb=str(before_surface_glb) if before_surface_parts is not None else "",
        before_surface_obj=str(before_surface_obj) if before_surface_parts is not None else "",
        before_surface_parts_json=(
            str(before_surface_parts_json) if before_surface_parts is not None else ""
        ),
        after_surface_glb=str(after_surface_glb) if strict_cluster_surface else "",
        after_surface_obj=str(after_surface_obj) if strict_cluster_surface else "",
        after_surface_parts_json=(
            str(after_surface_parts_json) if strict_cluster_surface else ""
        ),
        paper_model_glb=str(paper_model_glb) if strict_cluster_surface else "",
        paper_model_obj=str(paper_model_obj) if strict_cluster_surface else "",
        paper_model_texture=str(paper_model_texture) if strict_cluster_surface else "",
        paper_model_parts_json=(
            str(paper_model_parts_json) if strict_cluster_surface else ""
        ),
    )



def run_single_aabb(
    input_path: str | Path, output_dir: str | Path, config: PipelineConfig
) -> PipelineResult:
    """Skip PartField and fit one world-axis-aligned cuboid to the whole model."""
    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_meshes = load_world_meshes(input_path, up_axis=config.up_axis)
    center, scale = partfield_normalization(source_meshes)
    normalized_sources = normalize_meshes(source_meshes, center, scale)
    vertices = np.vstack([np.asarray(mesh.vertices, dtype=np.float64) for mesh in normalized_sources])
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    size = np.maximum(maxs - mins, 1e-5)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = (mins + maxs) * 0.5
    total_area = float(sum(float(mesh.area) for mesh in normalized_sources))
    total_faces = int(sum(len(mesh.faces) for mesh in normalized_sources))
    part = CuboidPart(
        name="whole_model", segment_id=0, size=size, transform=transform,
        face_count=total_faces, surface_area=total_area, source_center=vertices.mean(axis=0),
        metadata={"single_aabb": True, "shared_orientation": True},
    )
    parts = [part]

    sampler = ColoredSurfacePointCloud(
        normalized_sources, surface_samples=config.surface_samples,
        texture_filter=config.texture_filter, uv_wrap=config.uv_wrap,
    )
    texture, uv_rects = build_texture_atlas(
        parts, sampler, face_resolution=config.face_resolution, padding=config.padding,
        palette_size=config.palette_size, face_shading=config.face_shading,
    )
    paths = _export_paths(output_dir, None)
    texture.save(paths.texture)
    write_glb(parts, uv_rects, texture, paths.glb, surface_only=(config.obj_mode == "surface"))
    write_mtl(paths.mtl, paths.texture.name)
    obj_parts_dir = None
    if config.obj_mode in {"merged", "surface"}:
        write_obj(parts, uv_rects, paths.obj, paths.mtl.name, surface_only=(config.obj_mode == "surface"))
    elif config.obj_mode == "separate":
        obj_parts_dir = output_dir / "obj_parts"
        write_split_objs(parts, uv_rects, obj_parts_dir)
    elif config.obj_mode == "all":
        write_obj(parts, uv_rects, paths.obj, paths.mtl.name, surface_only=False)
        write_obj(parts, uv_rects, output_dir / "mc_model_surface.obj", paths.mtl.name, surface_only=True)
        obj_parts_dir = output_dir / "obj_parts"
        write_split_objs(parts, uv_rects, obj_parts_dir)
    else:
        raise ValueError(f"Unsupported obj_mode: {config.obj_mode}")

    metadata = {
        "source": str(input_path),
        "partfield": {"skipped": True, "reason": "clusters=1 single AABB mode"},
        "normalization": {"center": center.tolist(), "scale": float(scale)},
        "conversion": {"fit_mode": "aabb", "clusters": 1, "obj_mode": config.obj_mode},
        "parts": [part.to_json()],
    }
    paths.parts_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return PipelineResult(
        output_dir=str(output_dir), glb=str(paths.glb),
        obj=str(paths.obj) if paths.obj.exists() else "",
        obj_parts_dir=str(obj_parts_dir) if obj_parts_dir else None,
        texture=str(paths.texture), parts_json=str(paths.parts_json),
        normalized_mesh="", labels="", part_count=1, job_id="single_aabb",
    )

def run_pipeline(input_path: str | Path, output_dir: str | Path, config: PipelineConfig) -> PipelineResult:
    if config.clusters < 1:
        raise ValueError("--clusters must be at least 1")
    if config.clusters == 1:
        return run_single_aabb(input_path, output_dir, config)

    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Simplify only the geometry passed into PartField to avoid CUDA OOM.
    # Keep the original detailed input for texture baking in run_postprocess.
    partfield_input = input_path
    if config.simplify_faces and config.simplify_faces > 0:
        simplified_path = output_dir / "partfield_input_simplified.glb"
        print(f"[Mesh] Simplifying PartField input to ~{config.simplify_faces} faces...", flush=True)
        partfield_input = simplify_mesh_for_partfield(input_path, simplified_path, config.simplify_faces, up_axis=config.up_axis)
        print(f"[Mesh] Saved simplified PartField input: {partfield_input}", flush=True)

    runner_config = PartFieldRunConfig(
        repo=Path(config.partfield_repo).expanduser().resolve(),
        checkpoint=Path(config.checkpoint).expanduser().resolve(),
        clusters=config.clusters,
        clustering=config.clustering,
        adjacency=config.adjacency,
        n_point_per_face=config.n_point_per_face,
        n_sample_each=config.n_sample_each,
        preprocess_mesh=config.preprocess_mesh,
        up_axis=config.up_axis,
        force=config.force,
    )
    artifacts = run_partfield(partfield_input, output_dir, runner_config)
    return run_postprocess(
        input_path=input_path,
        output_dir=output_dir,
        normalized_mesh_path=artifacts.normalized_mesh,
        labels_path=artifacts.labels_path,
        config=config,
        artifacts=artifacts,
        simplified_mesh_path=(
            partfield_input if partfield_input != input_path else None
        ),
    )
