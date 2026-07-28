# V24: Fit outer primitives around frozen source interfaces

V23 fitted every PartField segment independently and repaired the assembly only
after fitting. V24 makes the source seam a hard geometric constraint before
candidate scoring.

## Geometry pipeline

1. Recover every face-label seam from the original simplified PartField mesh.
2. Project the seam points onto a stable interface plane and build one low-face
   convex interface polygon.
3. Reuse the exact same 3D polygon for both adjacent segments.
4. Restrict every primitive candidate to the segment side of every interface
   plane, add the immutable interface vertices, and rebuild a closed convex shell.
5. Score the constrained candidate against the source cluster and select the
   best primitive type/face count.
6. Validate identical parent/child interface vertices, opposite normals, equal
   area, and watertight topology.

No source-adjacent part is translated, rotated, or uniformly scaled. The overlap
resolver is disabled for frozen-interface parts because such a transform would
change the joint. Non-neighbour overlaps are reported in metadata instead.

Disconnected source components have no original interface to preserve. For those
only, the V23 connector fallback remains available so the exported assembly can
still be connected.

## CLI

```bash
--primitive-contact-mode fixed
--primitive-interface-max-sides 8
--primitive-interface-min-width-ratio 0.006
--primitive-interface-plane-tolerance-ratio 1e-6
```

Legacy modes remain available:

```bash
--primitive-contact-mode connector
--primitive-contact-mode move
```
