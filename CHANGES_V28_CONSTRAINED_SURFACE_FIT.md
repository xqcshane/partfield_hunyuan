# V28 Constrained Main-Body Surface Fit

V28 replaces the incorrect "merge surface labels, then fit one closed PCA primitive" behaviour for the main body.

## New hybrid geometry path

- `--primitive-part-mode auto`
  - broad PartField surface regions belonging to one main body are merged;
  - the largest attached non-thin body is fitted with constrained mesh simplification;
  - legs, tails, ears, stems, leaves and other appendages remain independent closed primitives.
- `--primitive-part-mode closed`
  - preserves the V26/V27 one-label-one-closed-primitive path for reproducibility.
- `--primitive-part-mode surface-patch`
  - uses more permissive body-patch classification but the same constrained simplifier.

## Constrained simplification

The main body now:

1. keeps source coordinates instead of using a PCA ellipsoid/convex replacement;
2. collapses a high-resolution PartField seam onto the immutable low-sided joint polygon before simplification;
3. locks the joint polygon during source-vertex clustering;
4. chooses representatives from original source vertices to avoid shrinkage on round fruit;
5. caps the exact same interface polygon used by the adjacent closed primitive;
6. exports one watertight, shared-vertex paper shell.

Metadata for a constrained body includes:

```json
{
  "primitive_type": "constrained_surface",
  "fitting_strategy": "constrained_mesh_simplification",
  "surface_fit_solver": "boundary_locked_source_vertex_clustering",
  "source_surface_geometry_preserved": true,
  "pca_primitive_replacement_applied": false,
  "fixed_boundary_collapse_applied": true
}
```

## New options

```bash
--primitive-surface-main-body-min-area-ratio 0.35
--primitive-surface-boundary-rings 0
--primitive-surface-search-steps 18
```

`boundary-rings=0` still freezes every interface vertex. Higher values preserve additional source rings but may substantially increase the paper face count.

## Compatibility

- AABB/OBB modes are unchanged.
- `--primitive-part-mode closed` keeps the previous fox behaviour.
- Fixed-interface, connector fallback, OBJ/GLB export and texture baking remain available.

## Verification

V28 includes tests for:

- exact fixed interfaces between a constrained body and a closed primitive;
- preservation of asymmetric source bounds;
- watertight paper topology;
- V26/V27 closed-mode compatibility;
- end-to-end OBJ, GLB and texture export.
