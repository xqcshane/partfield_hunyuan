# V21 Automatic Primitive Paper Fit

V21 adds `--fit-mode primitive` after PartField segmentation.

For every PartField label, the fitter reads the simplified cluster triangle
count, derives an automatic paper-face budget, and evaluates multiple closed
polyhedral candidates:

- oriented box;
- n-sided prism;
- n-sided frustum;
- n-sided cone;
- faceted ellipsoid;
- adaptive convex polyhedron.

Selection uses bidirectional sampled surface distance, 95th-percentile error,
face-budget deviation, volume deviation, primitive prior, and paper complexity.
The selected primitive is exported as an independent watertight shell with
shared geometry vertices.

New CLI options:

```text
--fit-mode primitive
--primitive-types auto
--primitive-target-faces 0
--primitive-max-faces 48
--primitive-max-sides 24
--primitive-fit-samples 2500
--primitive-complexity-weight 0.025
--no-primitive-resolve-overlaps
```

With `--obj-mode surface` or `all`, the pipeline writes a canonical
`paper_model.obj`, material, texture atlas, textured GLB preview, and detailed
candidate diagnostics in `paper_model_parts.json`.
