# v12: strict cluster-preserving cuboids

`--obj-mode surface` now means:

- preserve exactly the requested PartField cluster count;
- never silently discard a small non-empty AABB cluster;
- convert every cluster to one closed cuboid;
- automatically trim intersecting AABBs at a low-loss split plane;
- allow touching faces, but reject positive-volume overlaps;
- write one OBJ file containing one named object/group per cluster;
- use 8 shared geometry vertices and 6 quad faces per cuboid;
- retain the original `segment_id` in OBJ comments and `parts.json`.

The old union-shell behaviour is no longer used by `surface`, because it merged and subdivided cluster geometry and therefore did not preserve `--clusters N` as N closed boxes.

Validation example:

```bash
python validate_strict_cuboids.py \
  fox_mc_result/parts.json \
  fox_mc_result/mc_model.obj \
  --expected 8
```
