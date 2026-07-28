from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import trimesh


def rotation_to_y_up(up_axis: str) -> np.ndarray:
    axis = up_axis.lower()
    if axis == "y":
        return np.eye(4, dtype=np.float64)
    if axis == "z":
        return trimesh.transformations.rotation_matrix(-math.pi / 2.0, [1, 0, 0])
    if axis == "x":
        return trimesh.transformations.rotation_matrix(math.pi / 2.0, [0, 0, 1])
    raise ValueError(f"Unsupported up axis: {up_axis}")


def load_world_meshes(path: str | Path, up_axis: str = "y") -> list[trimesh.Trimesh]:
    """Load all scene meshes in world coordinates and convert to Y-up.

    Materials and UVs are retained on each geometry whenever trimesh supports
    the source format.
    """

    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input model does not exist: {source}")

    loaded = trimesh.load(source, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        meshes = [loaded.copy()]
    elif isinstance(loaded, trimesh.Scene):
        # Scene.dump applies node transforms and returns world-space meshes.
        dumped = loaded.dump(concatenate=False)
        meshes = [item.copy() for item in dumped if isinstance(item, trimesh.Trimesh)]
    else:
        raise ValueError(f"Unsupported object returned by trimesh: {type(loaded)!r}")

    rot = rotation_to_y_up(up_axis)
    valid: list[trimesh.Trimesh] = []
    for mesh in meshes:
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            continue
        mesh.apply_transform(rot)
        mesh.remove_unreferenced_vertices()
        valid.append(mesh)

    if not valid:
        raise ValueError("No triangle mesh was found in the input model.")
    return valid


def combined_bounds(meshes: Sequence[trimesh.Trimesh]) -> tuple[np.ndarray, np.ndarray]:
    mins = np.vstack([mesh.bounds[0] for mesh in meshes]).min(axis=0)
    maxs = np.vstack([mesh.bounds[1] for mesh in meshes]).max(axis=0)
    if np.any((maxs - mins) <= 1e-12):
        raise ValueError(f"Degenerate input bounds: min={mins}, max={maxs}")
    return mins.astype(np.float64), maxs.astype(np.float64)


def partfield_normalization(meshes: Sequence[trimesh.Trimesh]) -> tuple[np.ndarray, float]:
    """Return the exact center and scale used by PartField Demo_Dataset.

    PartField normalizes the longest bounding-box side to 1.8, i.e. the model
    fits in approximately [-0.9, 0.9].
    """

    bbmin, bbmax = combined_bounds(meshes)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * 0.9 / float((bbmax - bbmin).max())
    return center, scale


def normalize_meshes(meshes: Sequence[trimesh.Trimesh], center: np.ndarray, scale: float) -> list[trimesh.Trimesh]:
    normalized: list[trimesh.Trimesh] = []
    for source in meshes:
        mesh = source.copy()
        mesh.vertices = (np.asarray(mesh.vertices, dtype=np.float64) - center[None, :]) * scale
        normalized.append(mesh)
    return normalized


def export_canonical_glb(meshes: Sequence[trimesh.Trimesh], output_path: str | Path) -> Path:
    """Export the Y-up world-space source to a single GLB for PartField."""

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    for index, mesh in enumerate(meshes):
        scene.add_geometry(mesh, node_name=f"mesh_{index:03d}", geom_name=f"mesh_{index:03d}")
    data = scene.export(file_type="glb")
    output.write_bytes(data)
    return output


def load_triangle_mesh(path: str | Path) -> trimesh.Trimesh:
    mesh = trimesh.load(Path(path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected a triangle mesh: {path}")
    if len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no faces: {path}")
    return mesh



def simplify_mesh_for_partfield(input_path: str | Path, output_path: str | Path, target_faces: int, up_axis: str = "y") -> Path:
    """Create a simplified canonical GLB used only for PartField segmentation.

    The original detailed input is still used later for texture baking, so this
    simplification step only affects the geometry fed into PartField.
    """
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if target_faces <= 0:
        return export_canonical_glb(load_world_meshes(input_path, up_axis=up_axis), output_path)

    meshes = load_world_meshes(input_path, up_axis=up_axis)
    total_faces = sum(len(m.faces) for m in meshes)
    if total_faces <= target_faces:
        return export_canonical_glb(meshes, output_path)

    simplified = []
    for mesh in meshes:
        current_faces = len(mesh.faces)
        portion = max(4, int(target_faces * (current_faces / total_faces)))
        candidate = mesh.copy()
        simplified_mesh = None
        try:
            import fast_simplification

            vertices, faces = fast_simplification.simplify(
                candidate.vertices,
                candidate.faces,
                target_count=portion,
            )

            simplified_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                process=False,
            )

        except Exception:
            simplified_mesh = candidate
        if simplified_mesh is None:
            simplified_mesh = candidate
        simplified_mesh.remove_unreferenced_vertices()
        simplified.append(simplified_mesh)
    return export_canonical_glb(simplified, output_path)
