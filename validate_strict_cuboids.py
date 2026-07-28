#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate large-first PartField cuboids from parts_after_surface.json "
            "or parts.json and the post-surface OBJ."
        )
    )
    parser.add_argument("parts_json", type=Path)
    parser.add_argument("obj", type=Path)
    parser.add_argument(
        "--expected",
        type=int,
        required=True,
        help="Requested cluster count before fully enclosed parts are dropped.",
    )
    parser.add_argument(
        "--require-exact",
        action="store_true",
        help="Fail if any requested cluster was dropped.",
    )
    return parser.parse_args()


def _dropped_ids(data: dict, parts: list[dict]) -> list[int]:
    direct = data.get("dropped_segment_ids")
    if direct is not None:
        return sorted(int(value) for value in direct)

    conversion = data.get("conversion", {})
    indirect = conversion.get("surface_dropped_segment_ids")
    if indirect is not None:
        return sorted(int(value) for value in indirect)

    artifacts = data.get("artifacts", {}).get("after_surface") or {}
    artifact_ids = artifacts.get("dropped_segment_ids")
    if artifact_ids is not None:
        return sorted(int(value) for value in artifact_ids)

    for part in parts:
        values = part.get("metadata", {}).get("dropped_segment_ids_during_overlap_resolution")
        if values is not None:
            return sorted({int(value) for value in values})
    return []


def _obj_stats(path: Path) -> dict:
    object_names: list[str] = []
    vertices = 0
    faces = 0
    textured_faces = 0
    untextured_faces = 0
    non_quad_faces = 0
    current_object: str | None = None
    faces_per_object: dict[str, int] = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("o "):
            current_object = line[2:].strip()
            object_names.append(current_object)
            faces_per_object.setdefault(current_object, 0)
        elif line.startswith("v "):
            vertices += 1
        elif line.startswith("f "):
            tokens = line.split()[1:]
            faces += 1
            if len(tokens) != 4:
                non_quad_faces += 1
            if all("/" in token for token in tokens):
                textured_faces += 1
            elif all("/" not in token for token in tokens):
                untextured_faces += 1
            else:
                raise SystemExit(f"FAIL: mixed textured/untextured indices in face: {line}")
            if current_object is not None:
                faces_per_object[current_object] += 1

    return {
        "objects": object_names,
        "vertices": vertices,
        "faces": faces,
        "textured_faces": textured_faces,
        "untextured_faces": untextured_faces,
        "non_quad_faces": non_quad_faces,
        "faces_per_object": faces_per_object,
    }


def main() -> int:
    args = _parse_args()
    data = json.loads(args.parts_json.read_text(encoding="utf-8"))
    parts = data.get("parts", [])
    dropped = _dropped_ids(data, parts)
    kept = len(parts)

    if kept + len(dropped) != args.expected:
        raise SystemExit(
            f"FAIL: requested {args.expected}, but kept {kept} and recorded {len(dropped)} dropped"
        )
    if args.require_exact and dropped:
        raise SystemExit(f"FAIL: dropped segments are not allowed: {dropped}")
    if kept == 0:
        raise SystemExit("FAIL: no cuboids remain after surface processing")

    rotations = [np.asarray(part["rotation"], dtype=np.float64) for part in parts]
    base_rotation = rotations[0]
    if any(not np.allclose(rotation, base_rotation, atol=1e-6) for rotation in rotations[1:]):
        raise SystemExit("FAIL: cuboids do not share one AABB/shared-axis frame")

    centers = [np.asarray(part["center"], dtype=np.float64) @ base_rotation for part in parts]
    sizes = [np.asarray(part["size"], dtype=np.float64) for part in parts]
    if any(np.any(size <= 0.0) for size in sizes):
        raise SystemExit("FAIL: at least one retained cuboid has a non-positive side length")

    max_extent = max(float(np.max(size)) for size in sizes)
    tolerance = max(max_extent * 1e-9, 1e-12)
    for i in range(kept):
        min_i = centers[i] - sizes[i] * 0.5
        max_i = centers[i] + sizes[i] * 0.5
        for j in range(i + 1, kept):
            min_j = centers[j] - sizes[j] * 0.5
            max_j = centers[j] + sizes[j] * 0.5
            overlap = np.minimum(max_i, max_j) - np.maximum(min_i, min_j)
            if np.all(overlap > tolerance):
                raise SystemExit(
                    f"FAIL: segments {parts[i]['segment_id']} and {parts[j]['segment_id']} "
                    f"overlap by {overlap.tolist()}"
                )

    stats = _obj_stats(args.obj)
    if len(stats["objects"]) != kept:
        raise SystemExit(
            f"FAIL: expected {kept} retained OBJ objects, found {len(stats['objects'])}"
        )
    if stats["non_quad_faces"]:
        raise SystemExit(f"FAIL: found {stats['non_quad_faces']} non-quad OBJ faces")
    if stats["faces"] < kept * 6:
        raise SystemExit(
            f"FAIL: expected at least {kept * 6} face patches, found {stats['faces']}"
        )
    if any(count < 6 for count in stats["faces_per_object"].values()):
        raise SystemExit(
            f"FAIL: at least one retained object has fewer than six surface patches: "
            f"{stats['faces_per_object']}"
        )

    print(
        "PASS: "
        f"requested={args.expected}, kept={kept}, dropped={dropped}, "
        f"OBJ_objects={len(stats['objects'])}, quad_patches={stats['faces']}, "
        f"textured={stats['textured_faces']}, contact_untextured={stats['untextured_faces']}, "
        "no positive-volume overlap."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
