# V18: Semantic face-first constrained refit and adaptive torso splitting

## Why

V17 ranked clusters mainly by source surface area. A large torso was therefore
locked before the head/face. The later head box had to shrink to satisfy the
non-overlap constraint, which removed visible face texture and geometry.

## New deterministic behaviour

No LLM agent is used. A geometry classifier runs on the PartField cuboids and
label adjacency graph:

1. Infer animal/person layout (`--semantic-refit auto`).
2. Detect the main torso from true source surface area.
3. Detect the primary head/face as a compact, elevated endpoint; tail-like
   elongated endpoints are penalised.
4. Lock the face/head before the torso.
5. Refit the torso around the protected face.
6. If one torso AABB retains less than the requested target coverage, search
   for two touching, non-overlapping torso AABBs and accept the split only when
   it improves source surface coverage.
7. Fit tail, limbs and auxiliary pieces afterwards.
8. Surface processing still changes only contact-region materials, not geometry.

`--clusters N` remains the PartField semantic cluster count. The output cuboid
count may be larger because one source segment can own two cuboids. Every split
cuboid records `parent_segment_id` and `cuboid_index_within_segment`.

## New CLI options

```bash
--semantic-refit auto|animal|person|off
--no-adaptive-split
--max-extra-cuboids 1
--protected-min-coverage 0.85
--split-min-coverage-gain 0.05
```

Recommended for the fox/dog case:

```bash
--semantic-refit animal \
--max-extra-cuboids 1 \
--protected-min-coverage 0.85 \
--split-min-coverage-gain 0.05
```

## Output metadata

Each part now records fields including:

- `semantic_role`: `face`, `body`, `head_aux`, `tail`, `limb`, ...
- `protected_visual_region`
- `semantic_plan_confidence`
- `adaptive_split`
- `parent_segment_id`
- `cuboid_index_within_segment`
- `split_axis`
- `split_plane_local`
- `split_combined_coverage_ratio`
- `single_box_coverage_before_split`
- `split_coverage_gain`

`parts_before_surface.json` and `parts_after_surface.json` separately report
`requested_cluster_count`, `unique_segment_count`, and `cuboid_count`.
