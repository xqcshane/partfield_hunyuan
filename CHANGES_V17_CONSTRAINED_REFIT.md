# v17 — Non-overlap constrained AABB refit

- Added a source-face-supported beam search that refits AABBs under a no-positive-volume-overlap constraint.
- Prioritises parts by source surface area, preventing curved-tail AABB volume from outranking the body.
- Uses PartField face-label adjacency to preserve face-to-face body/limb/tail connections whenever feasible.
- Fully enclosed clusters with no valid candidate are dropped, never relocated.
- Added raw diagnostic exports: `mc_model_raw_aabb.*` and `parts_raw_aabb.json`.
- `before_surface` and `after_surface` now have identical geometry; surface processing changes only contact-region materials/UVs.
- Added `--surface-fit-strategy`, `--refit-min-coverage`, `--refit-beam-width`, and `--no-refit-preserve-contact`.
