#!/usr/bin/env python3
"""Validate a V24 primitive Paper Model OBJ and its optional metadata.

Checks:
- one OBJ object and no groups that split Blender import;
- every connected geometry component is an independent closed 2-manifold shell;
- every polygon is planar within a scale-aware tolerance;
- optional ``paper_model_parts.json`` shell/contact counts are consistent;
- fixed-interface mode reports unchanged, exactly shared source joint polygons;
- legacy connector mode reports a connected assembly without moving fitted source parts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_obj(path: Path) -> tuple[np.ndarray, list[list[int]], list[str], list[str]]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    objects: list[str] = []
    groups: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("v "):
            values = line.split()
            vertices.append([float(values[1]), float(values[2]), float(values[3])])
        elif line.startswith("f "):
            ids: list[int] = []
            for token in line.split()[1:]:
                value = int(token.split("/")[0])
                ids.append(value - 1 if value > 0 else len(vertices) + value)
            faces.append(ids)
        elif line.startswith("o "):
            objects.append(line.split(maxsplit=1)[1])
        elif line.startswith("g "):
            groups.append(line.split(maxsplit=1)[1])
    return np.asarray(vertices, dtype=np.float64), faces, objects, groups


def face_components(faces: list[list[int]]) -> list[list[int]]:
    vertex_to_faces: dict[int, list[int]] = {}
    for face_index, face in enumerate(faces):
        for vertex in face:
            vertex_to_faces.setdefault(vertex, []).append(face_index)
    neighbours = [set() for _ in faces]
    for indices in vertex_to_faces.values():
        for index in indices:
            neighbours[index].update(indices)

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


def polygon_planarity_error(points: np.ndarray) -> float:
    if len(points) <= 3:
        return 0.0
    centered = points - points.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return float("inf")
    normal = vh[-1]
    return float(np.max(np.abs(centered @ normal)))


def validate(
    path: Path,
    expected_shells: int | None,
    parts_json: Path | None,
) -> None:
    vertices, faces, objects, groups = parse_obj(path)
    errors: list[str] = []
    if objects != ["paper_model"]:
        errors.append(f"expected one object named paper_model, got {objects}")
    if groups:
        errors.append(f"expected no OBJ groups, got {groups}")
    if not len(vertices) or not faces:
        errors.append("OBJ has no vertices or faces")
    if any(len(face) < 3 for face in faces):
        errors.append("every paper face must contain at least three vertices")

    components = face_components(faces)
    if expected_shells is not None and len(components) != expected_shells:
        errors.append(f"expected {expected_shells} shells, got {len(components)}")

    if len(vertices):
        extent = max(float(np.max(vertices.max(axis=0) - vertices.min(axis=0))), 1e-8)
    else:
        extent = 1.0
    planarity_tolerance = extent * 1e-6

    for shell_index, component in enumerate(components):
        shell_faces = [faces[index] for index in component]
        shell_vertices = {vertex for face in shell_faces for vertex in face}
        edge_counts: dict[tuple[int, int], int] = {}
        for face in shell_faces:
            for a, b in zip(face, face[1:] + face[:1]):
                edge = tuple(sorted((a, b)))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
            error = polygon_planarity_error(vertices[np.asarray(face, dtype=np.int64)])
            if error > planarity_tolerance:
                errors.append(
                    f"shell {shell_index}: non-planar face error={error:.8g} "
                    f"> tolerance={planarity_tolerance:.8g}"
                )
        bad_edges = [edge for edge, count in edge_counts.items() if count != 2]
        if bad_edges:
            errors.append(
                f"shell {shell_index}: {len(bad_edges)} non-manifold/boundary edges; "
                f"examples={bad_edges[:5]}"
            )
        if len(shell_vertices) < 4:
            errors.append(f"shell {shell_index}: fewer than four unique vertices")

    metadata: dict[str, object] | None = None
    if parts_json is None:
        candidate = path.with_name("paper_model_parts.json")
        if candidate.exists():
            parts_json = candidate
    if parts_json is not None:
        metadata = json.loads(parts_json.read_text(encoding="utf-8"))
        shell_count = int(metadata.get("shell_count", -1))
        if shell_count != len(components):
            errors.append(
                f"metadata shell_count={shell_count} but OBJ contains {len(components)} shells"
            )
        if metadata.get("spatially_connected_assembly") is not True:
            errors.append("metadata does not report spatially_connected_assembly=true")
        parts = metadata.get("parts", [])
        if not isinstance(parts, list):
            errors.append("metadata parts must be a list")
        else:
            connector_count = sum(
                bool(part.get("metadata", {}).get("connector_part"))
                for part in parts
                if isinstance(part, dict)
            )
            reported_connectors = int(metadata.get("connector_count", connector_count))
            if connector_count != reported_connectors:
                errors.append(
                    f"metadata connector_count={reported_connectors}, actual={connector_count}"
                )
            source_by_id = {
                int(part.get("segment_id")): part
                for part in parts
                if isinstance(part, dict)
                and not bool(part.get("metadata", {}).get("connector_part"))
            }
            checked_interfaces: set[tuple[int, int]] = set()
            for part in parts:
                if not isinstance(part, dict):
                    continue
                part_meta = part.get("metadata", {})
                if not isinstance(part_meta, dict):
                    continue
                if part_meta.get("connector_part"):
                    continue
                mode = part_meta.get("contact_mode")
                if mode == "connector" and part_meta.get(
                    "main_part_rigid_transform_applied"
                ) is not False:
                    errors.append(
                        f"source segment {part.get('segment_id')} was moved in connector mode"
                    )
                if mode != "fixed":
                    continue
                segment_id = int(part.get("segment_id"))
                if part_meta.get("main_part_rigid_transform_applied") is not False:
                    errors.append(f"source segment {segment_id} moved in fixed mode")
                if part_meta.get("source_interface_geometry_changed") is not False:
                    errors.append(f"source segment {segment_id} changed a frozen interface")
                indices = part_meta.get("frozen_interface_face_indices", {})
                if not isinstance(indices, dict):
                    errors.append(f"source segment {segment_id} has invalid frozen interface map")
                    continue
                vertices_json = np.asarray(part.get("vertices", []), dtype=np.float64)
                polygons_json = part.get("polygons", [])
                for neighbour_text, face_index_value in indices.items():
                    neighbour = int(neighbour_text)
                    pair = tuple(sorted((segment_id, neighbour)))
                    if pair in checked_interfaces:
                        continue
                    checked_interfaces.add(pair)
                    other = source_by_id.get(neighbour)
                    if other is None:
                        errors.append(f"frozen interface {pair} references missing source part")
                        continue
                    other_meta = other.get("metadata", {})
                    other_indices = other_meta.get("frozen_interface_face_indices", {})
                    if str(segment_id) not in other_indices:
                        errors.append(f"frozen interface {pair} is not reciprocal")
                        continue
                    face_index = int(face_index_value)
                    other_face_index = int(other_indices[str(segment_id)])
                    try:
                        face_points = vertices_json[
                            np.asarray(polygons_json[face_index], dtype=np.int64)
                        ]
                        other_vertices = np.asarray(other.get("vertices", []), dtype=np.float64)
                        other_polygons = other.get("polygons", [])
                        other_points = other_vertices[
                            np.asarray(other_polygons[other_face_index], dtype=np.int64)
                        ]
                    except (IndexError, TypeError, ValueError):
                        errors.append(f"frozen interface {pair} has invalid face indices")
                        continue
                    if len(face_points) != len(other_points):
                        errors.append(f"frozen interface {pair} has different vertex counts")
                        continue
                    if len(face_points):
                        distances_ab = np.min(
                            np.linalg.norm(
                                face_points[:, None, :] - other_points[None, :, :], axis=2
                            ),
                            axis=1,
                        )
                        distances_ba = np.min(
                            np.linalg.norm(
                                other_points[:, None, :] - face_points[None, :, :], axis=2
                            ),
                            axis=1,
                        )
                        if max(float(np.max(distances_ab)), float(np.max(distances_ba))) > planarity_tolerance * 8.0:
                            errors.append(f"frozen interface {pair} does not share exact vertices")

    if errors:
        raise SystemExit("FAIL:\n- " + "\n- ".join(errors))

    connector_text = ""
    if metadata is not None:
        connector_text = f", connectors={int(metadata.get('connector_count', 0))}"
    print(
        f"PASS: one Blender object, {len(components)} independent closed planar shells, "
        f"{len(vertices)} vertices, {len(faces)} paper faces{connector_text}."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("obj", type=Path)
    parser.add_argument("--expected-shells", type=int)
    parser.add_argument("--parts-json", type=Path)
    args = parser.parse_args()
    validate(args.obj, args.expected_shells, args.parts_json)


if __name__ == "__main__":
    main()
