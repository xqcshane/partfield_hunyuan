# V25 — Fixed-interface local patch fitting

V24 rebuilt every constrained primitive as one convex hull. A torso-like segment
with several PartField seams can contain interface polygons that cannot all be
facets of the same convex body, causing:

```text
No primitive candidate can preserve the frozen interfaces
```

V25 keeps the V24 convex solver when it is valid and adds a deterministic
non-convex local-patch fallback:

1. The primitive body remains in its fitted position and orientation.
2. Every source interface polygon remains numerically unchanged.
3. A distinct local primitive triangle is selected for each interface by joint
   optimisation of normal, plane, position, and usable area.
4. Only that local triangle is replaced by a triangulated transition strip that
   terminates at the immutable interface polygon.
5. The result is one watertight, orientable paper shell, not a detached connector.
6. Adjacent parts still contain the exact same interface vertex set with opposite
   face orientation.

The fallback supports multiple mutually non-convex attachment interfaces and no
longer aborts the complete PartField model when the convex constraint is
impossible.
