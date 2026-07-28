from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .mesh_io import export_canonical_glb, load_world_meshes
from .models import PartFieldArtifacts


@dataclass
class PartFieldRunConfig:
    repo: Path
    checkpoint: Path
    clusters: int = 10
    clustering: str = "agglo"
    adjacency: str = "mst"
    n_point_per_face: int = 500
    n_sample_each: int = 5000
    preprocess_mesh: bool = False
    up_axis: str = "y"
    force: bool = False


def _run(command: Sequence[str], cwd: Path) -> None:
    printable = " ".join(str(value) for value in command)
    print(f"\n[run] cwd={cwd}\n{printable}\n", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(list(command), cwd=str(cwd), env=env, check=True)


def _job_id(input_path: Path, config: PartFieldRunConfig) -> str:
    stat = input_path.stat()
    payload = "|".join(
        [
            str(input_path.resolve()),
            str(stat.st_size),
            str(stat.st_mtime_ns),
            str(config.clusters),
            config.clustering,
            config.adjacency,
            str(config.n_point_per_face),
            str(config.n_sample_each),
            str(config.preprocess_mesh),
            config.up_axis,
        ]
    )
    suffix = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in input_path.stem)
    return f"{safe_stem}_{suffix}"


def validate_partfield_install(config: PartFieldRunConfig) -> None:
    required = [
        config.repo / "partfield_inference.py",
        config.repo / "run_part_clustering.py",
        config.repo / "configs" / "final" / "demo.yaml",
        config.checkpoint,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"PartField installation is incomplete:\n{formatted}")
    if config.clusters < 2:
        raise ValueError("clusters must be >= 2")
    if config.clustering not in {"agglo", "kmeans"}:
        raise ValueError("clustering must be agglo or kmeans")
    if config.adjacency not in {"naive", "mst"}:
        raise ValueError("adjacency must be naive or mst")


def run_partfield(
    input_path: str | Path,
    output_dir: str | Path,
    config: PartFieldRunConfig,
) -> PartFieldArtifacts:
    """Run official PartField feature extraction and clustering scripts."""

    validate_partfield_install(config)
    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    job_id = _job_id(input_path, config)
    work_dir = output_dir / "partfield_work" / job_id
    data_dir = work_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    canonical_input = data_dir / f"{job_id}.glb"
    if config.force or not canonical_input.exists():
        meshes = load_world_meshes(input_path, up_axis=config.up_axis)
        export_canonical_glb(meshes, canonical_input)

    # result_name is intentionally relative because official PartField writes to
    # f"exp_results/{result_name}" from the repository working directory.
    result_name = f"partfield_features/mc_pipeline/{job_id}"
    feature_dir = config.repo / "exp_results" / result_name
    cluster_dir = output_dir / "partfield_clusters" / job_id

    uid = job_id
    normalized_mesh = feature_dir / f"input_{uid}_0.ply"
    feature_batch = feature_dir / f"part_feat_{uid}_0_batch.npy"
    feature_plain = feature_dir / f"part_feat_{uid}_0.npy"

    if config.force:
        if feature_dir.exists():
            shutil.rmtree(feature_dir)
        if cluster_dir.exists():
            shutil.rmtree(cluster_dir)

    if not (normalized_mesh.exists() and (feature_batch.exists() or feature_plain.exists())):
        inference_command = [
            sys.executable,
            str(config.repo / "partfield_inference.py"),
            "-c",
            str(config.repo / "configs" / "final" / "demo.yaml"),
            "--opts",
            "continue_ckpt",
            str(config.checkpoint),
            "result_name",
            result_name,
            "dataset.data_path",
            str(data_dir),
            "n_point_per_face",
            str(config.n_point_per_face),
            "n_sample_each",
            str(config.n_sample_each),
            "preprocess_mesh",
            str(bool(config.preprocess_mesh)),
        ]
        _run(inference_command, cwd=config.repo)

    if config.clustering == "agglo":
        clustering_max = config.clusters
        clustering_command = [
            sys.executable,
            str(config.repo / "run_part_clustering.py"),
            "--root",
            str(feature_dir),
            "--dump_dir",
            str(cluster_dir),
            "--source_dir",
            str(data_dir),
            "--use_agglo",
            "True",
            "--max_num_clusters",
            str(clustering_max),
            "--option",
            "0" if config.adjacency == "naive" else "1",
        ]
        if config.adjacency == "mst":
            clustering_command.extend(["--with_knn", "True"])
    else:
        # Official KMeans loop is range(2, max_num_clusters), so use N+1 to
        # obtain exactly N clusters.
        clustering_max = config.clusters + 1
        clustering_command = [
            sys.executable,
            str(config.repo / "run_part_clustering.py"),
            "--root",
            str(feature_dir),
            "--dump_dir",
            str(cluster_dir),
            "--source_dir",
            str(data_dir),
            "--max_num_clusters",
            str(clustering_max),
        ]

    labels_path = cluster_dir / "cluster_out" / f"{uid}_0_{config.clusters:02d}.npy"
    colored_path = cluster_dir / "ply" / f"{uid}_0_{config.clusters:02d}.ply"
    if not labels_path.exists():
        _run(clustering_command, cwd=config.repo)

    if not normalized_mesh.exists():
        raise FileNotFoundError(f"PartField did not create normalized mesh: {normalized_mesh}")
    if not labels_path.exists():
        raise FileNotFoundError(f"PartField did not create requested labels: {labels_path}")

    return PartFieldArtifacts(
        job_id=job_id,
        canonical_input=canonical_input,
        feature_dir=feature_dir,
        cluster_dir=cluster_dir,
        normalized_mesh=normalized_mesh,
        labels_path=labels_path,
        colored_segmentation=colored_path if colored_path.exists() else None,
    )
