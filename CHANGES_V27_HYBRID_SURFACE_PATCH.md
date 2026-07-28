# V27 Hybrid Surface-Patch Primitive Fit

V27 adds a hybrid interpretation layer between PartField labels and primitive fitting.

## Problem fixed

PartField labels are surface regions, not guaranteed semantic solid parts. In V26, every label became a complete closed primitive. When one apple body was split into two large surface regions, both regions became complete ellipsoids and the apple separated into multiple bodies.

## New behaviour

- `--primitive-part-mode auto` measures every original label seam.
- Broad seams between similarly sized, volumetric regions are classified as internal surface-patch seams.
- Those labels are merged before primitive fitting, so one physical body receives one closed primitive.
- Narrow seams and thin/elongated appendages remain independent closed paper parts.
- External joints still use the V26 exact frozen-interface solver.
- `--primitive-part-mode closed` reproduces V26 one-label-one-shell behaviour.
- `--primitive-part-mode surface-patch` uses a more permissive patch classifier.

## Diagnostics

Each source part records `source_segment_ids`, `surface_patch_group`, and the seam metrics used by the classifier. The console prints each accepted merge and the final fitted groups.
