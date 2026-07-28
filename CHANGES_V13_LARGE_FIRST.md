# v13: large-first strict cuboids

`--obj-mode surface` keeps the strict eight-cluster behaviour from v12, but overlap resolution is now priority-based:

- rank cuboids once by original fitted AABB volume;
- preserve larger cuboids exactly;
- trim only the lower-volume cuboid in an overlapping pair;
- prefer a retained slab containing the cluster source centroid;
- if the smaller cuboid is fully enclosed, move only that smaller cuboid outside instead of collapsing it;
- keep exactly one closed cuboid per PartField cluster;
- reject any remaining positive-volume overlap.

The largest/body-like cuboid therefore remains unchanged even when several smaller clusters intersect it.
