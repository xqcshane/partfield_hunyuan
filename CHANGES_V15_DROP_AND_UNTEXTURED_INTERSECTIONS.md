# v15 changes

1. In strict `--obj-mode surface`, if a small cuboid is fully enclosed by a larger protected cuboid and no positive outside slab remains after trimming, the small cuboid is dropped instead of being moved outside.
2. In after-surface OBJ/GLB exports, faces hidden by cuboid-cuboid contact are exported without texture. OBJ uses `mc_hidden`; GLB splits hidden faces into separate untextured meshes.
3. Before-surface exports are unchanged and still preserve the original fitted AABBs.
