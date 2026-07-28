# V26 — Exact frozen-interface validation and deterministic part sides

V25 could still fail after all primitive candidates were fitted with:

```text
Frozen source-interface validation failed for edges: [[A, B]]
```

Increasing `--primitive-interface-plane-tolerance-ratio` did not solve this,
because the final validation also compared the two interface vertex sets and
face orientation.  A convex-hull candidate could merge the frozen polygon with
extra coplanar support vertices, and centroid-based side inference could place
curved parts on the same logical half-space.

V26 changes fixed-interface mode as follows:

1. The lower label (`segment_a`) always occupies the negative side of the
   source `a -> b` interface normal; the higher label occupies the positive
   side.  This no longer depends on primitive or cluster centroids.
2. A convex constrained candidate is accepted only when every contact face has
   exactly the same vertex set and area as the frozen source polygon.
3. If a convex face is enlarged or merged, fitting automatically falls back to
   the local non-convex adapter while keeping the primitive body in place.
4. Final connection validation checks exact polygon geometry and opposite body
   half-spaces.  Face winding is recorded as a diagnostic instead of rejecting
   an otherwise exact paper joint.
5. Failed edges now print plane, vertex, expected-polygon, area, normal, and
   body-side diagnostics.

The Hunyuan3D and PartField stages are unchanged. Existing GLB, normalized PLY,
and label NPY artifacts can be reused with `--postprocess-only`.
