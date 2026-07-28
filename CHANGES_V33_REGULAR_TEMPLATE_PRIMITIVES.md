# V33 — Regular Template Primitive Fit

V33 fixes the large flanges, collapsed joints, skewed frusta, and locally deformed primitives seen in V32.

## Root cause fixed

The procedural primitive candidates were regular when generated, but V24–V32 could later modify them by:

- inserting a reconstructed PartField interface polygon into the primitive;
- replacing a local triangle with a zipper-strip adapter;
- forcing a large source seam into the fitted part;
- using constrained-surface simplification for the main body;
- choosing arbitrary side faces as attachment faces.

These operations could create the wide white collar below a bottle cap, slanted faces, spikes, and collapsed animal parts.

## New default

```bash
--primitive-template-mode regular
```

Regular mode guarantees:

1. `box`, `prism`, `frustum`, `cone`, and `ellipsoid` candidates keep their canonical topology.
2. Prism, frustum, and cone axes remain coaxial; source centroid drift cannot shear the template.
3. No frozen interface polygon is cut into a regular primitive.
4. No constrained-surface vertex clustering is used in regular mode.
5. Contact restoration uses an existing canonical attachment face or a small bounded connector.
6. Connector area is capped relative to the smaller attachment face.
7. A rigid-motion-invariant edge-length signature verifies that contact processing did not deform the source primitive.
8. `convex` is a fallback only in regular mode unless explicitly requested alone.

Legacy behaviour is still available:

```bash
--primitive-template-mode adaptive
```

Adaptive mode retains constrained surfaces and exact frozen-interface deformation for cases that explicitly require it.

## Canonical attachment rules

- `box`: any planar face;
- `prism`, `frustum`: axial end caps only;
- `cone`: base cap only;
- `ellipsoid`: a small polar face band in the requested direction;
- `convex`: all faces, fallback only.

## Bounded connector

```bash
--primitive-regular-connector-max-face-area-ratio 0.08
```

The connector end area cannot exceed 8% of the smaller selected attachment face by default. A long or noisy source seam can reduce connector size, but cannot enlarge it into a visible flange.

## Spatial contact inference

Hunyuan meshes often contain tiny gaps between parts that visually touch. V33 adds:

```bash
--primitive-contact-proximity-ratio 0.015
--primitive-contact-proximity-min-points 8
--primitive-contact-proximity-min-coverage 0.01
```

This infers contact from nearby source surfaces even when there are no shared mesh edges, useful for bottle caps, limbs, ears, and tails.

## Recommended milk-bottle settings

```bash
--fit-mode primitive \
--primitive-template-mode regular \
--primitive-part-mode auto \
--primitive-types prism,frustum,ellipsoid,cone \
--primitive-contact-mode auto \
--primitive-contact-proximity-ratio 0.025 \
--primitive-contact-proximity-min-points 8 \
--primitive-contact-proximity-min-coverage 0.01 \
--primitive-contact-weak-threshold 0.12 \
--primitive-contact-strong-threshold 0.42 \
--primitive-contact-medium-mode connector \
--primitive-connector-radius-ratio 0.018 \
--primitive-regular-connector-max-face-area-ratio 0.05
```

## Validation

V33 passes 46 automated tests, including:

- canonical axial-cap attachment selection;
- no constrained-surface use in regular mode;
- no frozen-interface cutting in regular mode;
- contact-stage shape signature preservation;
- bounded connector area;
- spatial recovery of an unwelded bottle-cap contact;
- all previous primitive, AABB, export, texture, and interface tests.
