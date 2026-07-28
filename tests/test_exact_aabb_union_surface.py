from pathlib import Path

import numpy as np
import pytest

from partfield_mc.cuboid_fit import _trim_parallel_overlaps
from partfield_mc.exporters import write_obj
from partfield_mc.models import CuboidPart
from partfield_mc.texture import FACE_NAMES


def _part(name, segment_id, center, size):
    transform = np.eye(4)
    transform[:3, 3] = center
    return CuboidPart(
        name,
        segment_id,
        np.asarray(size, float),
        transform,
        0,
        0.0,
        np.asarray(center, float),
    )


def test_surface_rejects_overlapping_cluster_boxes(tmp_path: Path):
    parts = [_part("a", 0, [0, 0, 0], [2, 2, 2]), _part("b", 1, [1, 0, 0], [2, 2, 2])]
    uv = {(i, face): (0.0, 0.0, 1.0, 1.0) for i in range(2) for face in FACE_NAMES}
    with pytest.raises(ValueError, match="overlapping cuboids"):
        write_obj(parts, uv, tmp_path / "surface.obj", "mc_model.mtl", surface_only=True)


def test_trimmed_surface_preserves_two_closed_boxes(tmp_path: Path):
    parts = [_part("a", 0, [0, 0, 0], [2, 2, 2]), _part("b", 1, [1, 0, 0], [2, 2, 2])]
    _trim_parallel_overlaps(parts, gap=0.0)
    uv = {(i, face): (0.0, 0.0, 1.0, 1.0) for i in range(2) for face in FACE_NAMES}
    target = tmp_path / "surface.obj"
    write_obj(parts, uv, target, "mc_model.mtl", surface_only=True)

    text = target.read_text()
    assert "part_count=2 overlaps_expected=0" in text
    assert len([line for line in text.splitlines() if line.startswith("o ")]) == 2
    assert len([line for line in text.splitlines() if line.startswith("v ")]) == 16
    assert len([line for line in text.splitlines() if line.startswith("f ")]) == 12
    assert len(
        [line for line in text.splitlines() if line.startswith("f ") and "/" not in line]
    ) == 2

    face_lines = [line.split()[1:] for line in text.splitlines() if line.startswith("f ")]
    for part_index in range(2):
        faces = face_lines[part_index * 6 : (part_index + 1) * 6]
        edge_counts = {}
        for face in faces:
            ids = [int(token.split("/")[0]) for token in face]
            for a, b in zip(ids, ids[1:] + ids[:1]):
                edge = tuple(sorted((a, b)))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        assert len(edge_counts) == 12
        assert set(edge_counts.values()) == {2}
