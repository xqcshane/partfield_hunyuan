# v14: simplified / before-surface / after-surface stage exports

When `--obj-mode surface` or `--obj-mode all` is used, one run now preserves all three geometry stages:

1. `partfield_input_simplified.glb` — the simplified triangle mesh passed to PartField.
2. `mc_model_before_surface.glb/.obj` — one direct closed AABB per cluster before overlap removal. These boxes may overlap.
3. `mc_model_after_surface.glb/.obj` — the final large-first, non-overlapping closed cuboids.

The legacy `mc_model.glb/.obj` filenames remain the final after-surface result for backwards compatibility.

Each cuboid stage has its own texture, MTL and JSON metadata files. `parts.json` contains an artifact manifest and both the before-surface and after-surface part records.
