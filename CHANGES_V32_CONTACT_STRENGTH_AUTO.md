# V32 — Contact-strength auto classification

V32 replaces the all-or-nothing primitive contact constraint with a per-edge contact-strength policy.

## New mode

```bash
--primitive-contact-mode auto
```

For each PartField label seam, V32 measures:

- reconstructed interface area divided by the smaller part area;
- seam length divided by `sqrt(smaller part area)`;
- boundary edge count normalised by source face count;
- unique boundary point count normalised by source face count.

The weighted score classifies the seam as:

- **strong**: insert and preserve the immutable shared interface before fitting;
- **medium**: keep both main parts stationary and add a small connector by default;
- **weak**: do not create a fixed interface or connector; accidental fitted overlap is separated when this does not break another strong joint.

This allows apple stems/leaves, fox ear tips, tail tips, and other very small source contacts to remain separate instead of forcing a large artificial intersection.

## New CLI parameters

```text
--primitive-contact-weak-threshold 0.20
--primitive-contact-strong-threshold 0.55
--primitive-contact-min-edge-count 6
--primitive-contact-medium-mode connector|separate
```

The thresholds satisfy `0 <= weak < strong <= 1`.

## Mandatory paper face ceiling

```text
--primitive-surface-hard-max-faces 512
```

A constrained surface cannot proceed into texture-atlas generation above this ceiling. V32 first closes unfrozen weak-contact seams before simplification, preventing all seam vertices from becoming locked. If the source-preserving simplifier still exceeds the ceiling, it tries a source-support convex fallback and finally a bounded closed primitive. This prevents 30,000-face paper atlases.

## Metadata

`paper_model_parts.json` now records:

- `contact_strength_classification`;
- per-edge score and component metrics;
- `allowed_separated_contact_edges`;
- `weak_contact_overlap_separation`;
- `required_contact_graph_connected`;
- `spatially_connected_assembly` separately from required-contact validity;
- constrained-surface hard-cap status and fallback information.

## Compatibility

- `--primitive-contact-mode fixed` retains the V31 all-seams-fixed path.
- `--primitive-contact-mode connector` retains the stationary connector path.
- `--primitive-contact-mode move` retains the legacy rigid relocation path.
- `--primitive-part-mode closed` remains available for the previous fox one-label-one-closed-primitive workflow.

## Primitive source

V32 still uses the existing programmatically generated `box/prism/frustum/cone/ellipsoid/convex` candidates. It does **not** yet load an external OBJ/GLB preset-template library.
