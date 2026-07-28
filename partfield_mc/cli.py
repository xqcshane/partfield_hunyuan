from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from .pipeline import PipelineConfig, run_pipeline, run_postprocess
from .hunyuan3d import generate_mesh


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run NVIDIA PartField on a mesh, fit either one cuboid or one automatic "
            "low-face paper primitive per segment, and bake the original detailed texture."
        )
    )
    p.add_argument("input", nargs="?", help="Input GLB/OBJ model (optional when using image arguments)")
    p.add_argument("--image", help="Single image input for Hunyuan3D image-to-model generation")
    p.add_argument("--image-front", help="Front-view image for true multi-view generation")
    p.add_argument("--image-side", help="Side-view image alias (mapped to left view)")
    p.add_argument("--image-left", help="Left-view image for true multi-view generation")
    p.add_argument("--image-right", help="Right-view image for true multi-view generation")
    p.add_argument("--image-back", help="Back-view image for true multi-view generation")
    p.add_argument("--multiview", action="store_true", help="Use true multi-view generation when 2+ view images are provided")
    p.add_argument("--texture", action="store_true", help="Generate a textured mesh instead of geometry only")
    p.add_argument("--hunyuan-repo", help="Path to Hunyuan3D repository")
    p.add_argument("-o", "--output", default="partfield_mc_output")
    p.add_argument("--partfield-repo", default="./PartField")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--clusters", type=int, default=10, help="1: skip PartField and fit one global AABB; >=2: requested PartField part count")
    p.add_argument("--clustering", choices=("agglo", "kmeans"), default="agglo")
    p.add_argument("--adjacency", choices=("naive", "mst"), default="mst")
    p.add_argument("--n-point-per-face", type=int, default=500)
    p.add_argument("--n-sample-each", type=int, default=5000)
    p.add_argument("--preprocess-mesh", action="store_true")
    p.add_argument("--up-axis", choices=("x", "y", "z"), default="y")
    p.add_argument(
        "--fit-mode",
        choices=("obb", "aabb", "shared", "primitive"),
        default="obb",
        help=(
            "obb: each part rotates independently; aabb: all parts use world axes; "
            "shared: all parts share one model-level rotation; primitive: after PartField, "
            "automatically choose a closed low-face box/prism/frustum/cone/ellipsoid/convex "
            "proxy per cluster, then reconnect source-adjacent clusters for Blender Paper Model export"
        ),
    )
    p.add_argument("--min-area-ratio", type=float, default=0.0)
    p.add_argument("--min-faces", type=int, default=4)
    p.add_argument("--grid-divisions", type=int, default=0, help="0 disables MC grid snapping; try 32")
    p.add_argument("--category", choices=("generic", "auto", "animal", "person"), default="generic")
    p.add_argument("--forward-axis", choices=("auto", "+x", "-x", "+z", "-z"), default="auto")
    p.add_argument(
        "--resolve-overlaps",
        action="store_true",
        help="Move smaller parallel cuboids outward until they no longer intersect",
    )
    p.add_argument(
        "--part-gap-ratio",
        type=float,
        default=0.0,
        help="Gap as a fraction of the model longest side; try 0.003 to 0.01",
    )
    p.add_argument("--face-resolution", type=int, default=64)
    p.add_argument("--surface-samples", type=int, default=500000)
    p.add_argument("--palette-size", type=int, default=0)
    p.add_argument("--texture-filter", choices=("nearest", "bilinear"), default="bilinear")
    p.add_argument("--uv-wrap", choices=("repeat", "clamp"), default="repeat")
    p.add_argument("--padding", type=int, default=1)
    p.add_argument("--face-shading", action="store_true")
    p.add_argument(
        "--surface-fit-strategy",
        choices=("refit", "trim"),
        default="refit",
        help=(
            "Geometry strategy for obj-mode surface/all. refit searches source-face-supported "
            "non-overlapping AABBs and preserves label adjacency; trim uses the legacy "
            "large-first post-fit cut."
        ),
    )
    p.add_argument(
        "--refit-min-coverage",
        type=float,
        default=0.05,
        help="Minimum source surface-area coverage required to keep a constrained-refit cuboid",
    )
    p.add_argument(
        "--refit-beam-width",
        type=int,
        default=64,
        help="Beam width for non-overlapping constrained AABB search",
    )
    p.add_argument(
        "--no-refit-preserve-contact",
        action="store_true",
        help="Do not prioritise face-to-face contact for source-adjacent PartField labels",
    )
    p.add_argument(
        "--semantic-refit",
        choices=("off", "auto", "animal", "person"),
        default="auto",
        help=(
            "Deterministic geometry-based refit priority. auto detects animal/person, protects "
            "the head/face before the torso, and does not use an LLM agent."
        ),
    )
    p.add_argument(
        "--no-adaptive-split",
        action="store_true",
        help="Disable automatic splitting of a torso cluster into two touching cuboids",
    )
    p.add_argument(
        "--max-extra-cuboids",
        type=int,
        default=1,
        help="Maximum cuboids added beyond the PartField cluster count by adaptive splitting",
    )
    p.add_argument(
        "--protected-min-coverage",
        type=float,
        default=0.85,
        help=(
            "Target source-area coverage for the torso after protecting the face; below this, "
            "a two-cuboid torso is considered"
        ),
    )
    p.add_argument(
        "--split-min-coverage-gain",
        type=float,
        default=0.05,
        help="Minimum combined source-area coverage improvement required to accept a split",
    )
    p.add_argument(
        "--obj-mode",
        choices=("merged", "separate", "surface", "all"),
        default="merged",
        help=(
            "OBJ output: merged keeps all cuboid faces in one OBJ; separate writes one OBJ per part; "
            "surface uses non-overlapping constrained AABB refitting by default, then leaves exact "
            "contact sub-regions untextured and also writes canonical paper_model.obj. With "
            "--fit-mode primitive, surface writes closed primitive shells whose source PartField "
            "adjacency is restored as a connected assembly. all writes merged, surface, separate, "
            "and paper-model outputs"
        ),
    )
    p.add_argument(
        "--primitive-part-mode",
        choices=("auto", "closed", "surface-patch"),
        default="auto",
        help=(
            "How PartField labels become physical paper parts. auto merges broad main-body "
            "surface patches and simplifies that source surface with fixed boundaries, while "
            "thin/elongated appendages remain closed primitives. closed preserves the V26/V27 "
            "one-label-one-primitive behaviour. surface-patch uses a more permissive main-body "
            "classifier but still protects thin appendages."
        ),
    )
    p.add_argument(
        "--primitive-patch-min-segment-area-ratio",
        type=float,
        default=0.10,
        help="Minimum smaller-segment area fraction required for an automatic surface-patch merge",
    )
    p.add_argument(
        "--primitive-patch-min-area-balance",
        type=float,
        default=0.30,
        help="Minimum smaller/larger source-area ratio required for an automatic patch merge",
    )
    p.add_argument(
        "--primitive-patch-min-interface-area-ratio",
        type=float,
        default=0.14,
        help="Minimum reconstructed seam area divided by the smaller segment area",
    )
    p.add_argument(
        "--primitive-patch-min-seam-length-ratio",
        type=float,
        default=0.75,
        help="Minimum seam length divided by sqrt(smaller segment area)",
    )
    p.add_argument(
        "--primitive-surface-main-body-min-area-ratio",
        type=float,
        default=0.35,
        help=(
            "Minimum total source-area fraction for the largest non-thin group to use "
            "boundary-locked constrained mesh simplification in auto mode."
        ),
    )
    p.add_argument(
        "--primitive-surface-boundary-rings",
        type=int,
        default=0,
        help=(
            "Number of extra source vertex rings frozen around each PartField attachment boundary "
            "during constrained surface simplification (0-4)."
        ),
    )
    p.add_argument(
        "--primitive-surface-search-steps",
        type=int,
        default=18,
        help="Number of grid resolutions evaluated by the constrained surface simplifier.",
    )
    p.add_argument(
        "--primitive-surface-min-reduction-ratio",
        type=float,
        default=0.15,
        help=(
            "Minimum required constrained-surface face reduction fraction. The default 0.15 "
            "requires at least 15%% fewer outer triangles; robust fallback keeps the most reduced "
            "topology-safe result instead of silently returning the original dense patch."
        ),
    )
    p.add_argument(
        "--primitive-surface-hard-max-faces",
        type=int,
        default=512,
        help=(
            "Hard maximum faces for one constrained-surface paper part. If source-preserving "
            "simplification cannot satisfy this ceiling, the pipeline uses a low-face source-derived "
            "convex or closed-primitive fallback before texture atlas generation."
        ),
    )
    p.add_argument(
        "--primitive-validation-policy",
        choices=("strict", "repair", "warn"),
        default="repair",
        help=(
            "Fixed-interface failure policy. repair (default) accepts exact shared geometry when "
            "side tests are ambiguous, attempts canonical interface repair, and continues with "
            "warnings; warn never aborts; strict preserves fail-fast validation."
        ),
    )
    p.add_argument(
        "--primitive-types",
        default="auto",
        help=(
            "Comma-separated candidates used by --fit-mode primitive: "
            "box,prism,frustum,cone,ellipsoid,convex. auto enables all."
        ),
    )
    p.add_argument(
        "--primitive-target-faces",
        type=int,
        default=0,
        help=(
            "Paper-face target per PartField cluster in primitive mode. 0 derives a target "
            "from each simplified cluster's source triangle count."
        ),
    )
    p.add_argument(
        "--primitive-max-faces",
        type=int,
        default=48,
        help="Maximum canonical paper faces allowed for one fitted primitive",
    )
    p.add_argument(
        "--primitive-max-sides",
        type=int,
        default=24,
        help="Maximum ring side count for prism/frustum/cone/ellipsoid candidates",
    )
    p.add_argument(
        "--primitive-fit-samples",
        type=int,
        default=2500,
        help="Surface samples per cluster and candidate used for geometric fit scoring",
    )
    p.add_argument(
        "--primitive-complexity-weight",
        type=float,
        default=0.025,
        help="Penalty weight for paper face-count complexity during primitive selection",
    )
    p.add_argument(
        "--no-primitive-resolve-overlaps",
        action="store_true",
        help=(
            "Do not remove accidental overlaps between non-adjacent primitive shells. "
            "Source-adjacent joints are never separated by this pass."
        ),
    )
    p.add_argument(
        "--no-primitive-preserve-contacts",
        action="store_true",
        help="Disable source-boundary contact restoration between fitted primitive parts",
    )
    p.add_argument(
        "--primitive-contact-overlap-ratio",
        type=float,
        default=0.0,
        help=(
            "Optional hidden insertion as a fraction of the model longest side in primitive mode. "
            "The default 0 keeps exact coplanar face-to-face paper joints."
        ),
    )
    p.add_argument(
        "--primitive-contact-mode",
        choices=("auto", "fixed", "connector", "move"),
        default="auto",
        help=(
            "Contact strategy for primitive mode. auto classifies each source seam by contact strength: "
            "strong=fixed shared interface, medium=stationary connector by default, weak=allowed to "
            "separate. fixed forces every seam, connector adds stationary joints, and move is legacy."
        ),
    )
    p.add_argument(
        "--primitive-contact-weak-threshold",
        type=float,
        default=0.20,
        help="Auto-contact score below which the fitted parts may remain separated.",
    )
    p.add_argument(
        "--primitive-contact-strong-threshold",
        type=float,
        default=0.55,
        help="Auto-contact score at or above which an immutable shared interface is required.",
    )
    p.add_argument(
        "--primitive-contact-min-edge-count",
        type=int,
        default=6,
        help=(
            "Source boundary edges required before a seam can be medium/strong. Fewer edges always "
            "classify as weak, preventing accidental point/short-edge contacts from being forced."
        ),
    )
    p.add_argument(
        "--primitive-contact-medium-mode",
        choices=("connector", "separate"),
        default="connector",
        help=(
            "Handling for medium-strength seams in auto mode. connector preserves placement and adds "
            "a small paper joint; separate permits a gap without moving the main parts."
        ),
    )
    p.add_argument(
        "--primitive-connector-sides",
        type=int,
        default=4,
        help="Number of sides for each closed connector patch (3-12; 4 is paper-friendly)",
    )
    p.add_argument(
        "--primitive-connector-radius-ratio",
        type=float,
        default=0.028,
        help=(
            "Preferred connector patch radius as a fraction of the model longest side. "
            "The actual radius is clamped so the patch remains inside both selected faces."
        ),
    )
    p.add_argument(
        "--primitive-connector-inset-ratio",
        type=float,
        default=0.28,
        help=(
            "Move each interface centre from the source-boundary projection toward the face "
            "centroid before sizing the connector; prevents edge/point contacts."
        ),
    )
    p.add_argument(
        "--primitive-connector-min-length-ratio",
        type=float,
        default=0.002,
        help=(
            "Minimum connector length as a fraction of the model longest side. Used only when "
            "selected faces nearly intersect but do not form a valid coplanar joint."
        ),
    )
    p.add_argument(
        "--primitive-interface-max-sides",
        type=int,
        default=8,
        help=(
            "Maximum polygon sides used to reconstruct each immutable source interface in fixed "
            "contact mode (3-32)."
        ),
    )
    p.add_argument(
        "--primitive-interface-min-width-ratio",
        type=float,
        default=0.006,
        help=(
            "Minimum half-width of a degenerate source seam fallback patch as a fraction of the "
            "model longest side. Normal closed boundary loops keep their measured footprint."
        ),
    )
    p.add_argument(
        "--primitive-interface-plane-tolerance-ratio",
        type=float,
        default=1e-6,
        help="Numerical plane tolerance for validating unchanged shared interface polygons.",
    )
    p.add_argument("--simplify-faces", type=int, default=50000, help="Target face count for the mesh passed into PartField; 0 disables simplification")
    p.add_argument("--force", action="store_true", help="Delete cached PartField outputs and rerun")
    p.add_argument("--postprocess-only", action="store_true")
    p.add_argument("--normalized-mesh", help="PartField exp_results/.../input_UID_0.ply")
    p.add_argument("--labels", help="PartField cluster_out/UID_0_NN.npy")
    return p


def _collect_mv_images(args: argparse.Namespace) -> dict[str, str]:
    mv = {}
    if args.image_front:
        mv["front"] = args.image_front
    if args.image_left:
        mv["left"] = args.image_left
    if args.image_side and "left" not in mv:
        mv["left"] = args.image_side
    if args.image_right:
        mv["right"] = args.image_right
    if args.image_back:
        mv["back"] = args.image_back
    return mv


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mv_images = _collect_mv_images(args)
    generated_from_image = False
    if args.image or mv_images:
        output_dir = Path(args.output)
        suffix = "hunyuan3d_generated"
        if args.texture and mv_images:
            suffix = "hunyuan3d_textured_multiview"
        elif args.texture:
            suffix = "hunyuan3d_textured"
        elif mv_images:
            suffix = "hunyuan3d_multiview"
        generated = output_dir / f"{suffix}.glb"
        generate_mesh(
            image=args.image,
            images=mv_images or None,
            output=generated,
            hunyuan_repo=args.hunyuan_repo,
            texture=args.texture,
            multiview=args.multiview or len(mv_images) >= 2,
        )
        args.input = str(generated)
        generated_from_image = True
    if not args.input:
        raise ValueError("input mesh or image arguments are required")

    partfield_repo = Path(args.partfield_repo).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else (partfield_repo / "model" / "model_objaverse.ckpt")

    # Hunyuan3D meshes are usually too dense for PartField. Lower defaults automatically
    # when the user didn't override them.
    n_point_per_face = args.n_point_per_face
    n_sample_each = args.n_sample_each
    if generated_from_image and args.n_point_per_face == 500 and args.n_sample_each == 5000:
        n_point_per_face = 50
        n_sample_each = 1000
        print("[PartField] Auto-adjusting sampling for Hunyuan meshes: n_point_per_face=50, n_sample_each=1000", flush=True)

    config = PipelineConfig(
        partfield_repo=partfield_repo,
        checkpoint=checkpoint,
        clusters=args.clusters,
        clustering=args.clustering,
        adjacency=args.adjacency,
        n_point_per_face=n_point_per_face,
        n_sample_each=n_sample_each,
        preprocess_mesh=args.preprocess_mesh,
        up_axis=args.up_axis,
        fit_mode=args.fit_mode,
        min_area_ratio=args.min_area_ratio,
        min_faces=args.min_faces,
        grid_divisions=args.grid_divisions,
        category=args.category,
        forward_axis=args.forward_axis,
        resolve_overlaps=args.resolve_overlaps,
        part_gap_ratio=args.part_gap_ratio,
        face_resolution=args.face_resolution,
        surface_samples=args.surface_samples,
        palette_size=args.palette_size,
        texture_filter=args.texture_filter,
        uv_wrap=args.uv_wrap,
        padding=args.padding,
        face_shading=args.face_shading,
        obj_mode=args.obj_mode,
        surface_fit_strategy=args.surface_fit_strategy,
        refit_min_coverage=args.refit_min_coverage,
        refit_beam_width=args.refit_beam_width,
        refit_preserve_contact=not args.no_refit_preserve_contact,
        semantic_refit=args.semantic_refit,
        adaptive_split=not args.no_adaptive_split,
        max_extra_cuboids=args.max_extra_cuboids,
        protected_min_coverage=args.protected_min_coverage,
        split_min_coverage_gain=args.split_min_coverage_gain,
        primitive_types=args.primitive_types,
        primitive_target_faces=args.primitive_target_faces,
        primitive_max_faces=args.primitive_max_faces,
        primitive_max_sides=args.primitive_max_sides,
        primitive_fit_samples=args.primitive_fit_samples,
        primitive_complexity_weight=args.primitive_complexity_weight,
        primitive_resolve_overlaps=not args.no_primitive_resolve_overlaps,
        primitive_preserve_contacts=not args.no_primitive_preserve_contacts,
        primitive_contact_overlap_ratio=args.primitive_contact_overlap_ratio,
        primitive_contact_mode=args.primitive_contact_mode,
        primitive_connector_sides=args.primitive_connector_sides,
        primitive_connector_radius_ratio=args.primitive_connector_radius_ratio,
        primitive_connector_inset_ratio=args.primitive_connector_inset_ratio,
        primitive_connector_min_length_ratio=args.primitive_connector_min_length_ratio,
        primitive_interface_max_sides=args.primitive_interface_max_sides,
        primitive_interface_min_width_ratio=args.primitive_interface_min_width_ratio,
        primitive_interface_plane_tolerance_ratio=args.primitive_interface_plane_tolerance_ratio,
        primitive_part_mode=args.primitive_part_mode,
        primitive_patch_min_segment_area_ratio=args.primitive_patch_min_segment_area_ratio,
        primitive_patch_min_area_balance=args.primitive_patch_min_area_balance,
        primitive_patch_min_interface_area_ratio=args.primitive_patch_min_interface_area_ratio,
        primitive_patch_min_seam_length_ratio=args.primitive_patch_min_seam_length_ratio,
        primitive_surface_main_body_min_area_ratio=args.primitive_surface_main_body_min_area_ratio,
        primitive_surface_boundary_rings=args.primitive_surface_boundary_rings,
        primitive_surface_search_steps=args.primitive_surface_search_steps,
        primitive_surface_min_reduction_ratio=args.primitive_surface_min_reduction_ratio,
        primitive_surface_hard_max_faces=args.primitive_surface_hard_max_faces,
        primitive_validation_policy=args.primitive_validation_policy,
        primitive_contact_weak_threshold=args.primitive_contact_weak_threshold,
        primitive_contact_strong_threshold=args.primitive_contact_strong_threshold,
        primitive_contact_min_edge_count=args.primitive_contact_min_edge_count,
        primitive_contact_medium_mode=args.primitive_contact_medium_mode,
        simplify_faces=args.simplify_faces,
        force=args.force,
    )
    try:
        if args.postprocess_only:
            if not args.normalized_mesh or not args.labels:
                raise ValueError("--postprocess-only requires --normalized-mesh and --labels")
            result = run_postprocess(
                args.input,
                args.output,
                args.normalized_mesh,
                args.labels,
                config,
            )
        else:
            result = run_pipeline(args.input, args.output, config)
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: external command failed with code {exc.returncode}", file=sys.stderr)
        return exc.returncode or 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
