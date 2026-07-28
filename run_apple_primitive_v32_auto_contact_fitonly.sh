#!/usr/bin/env bash
set -euo pipefail

: "${SOURCE_GLB:?Set SOURCE_GLB to the textured Hunyuan3D GLB}"
: "${NORMALIZED_MESH:?Set NORMALIZED_MESH to the matching PartField normalized PLY}"
: "${LABELS:?Set LABELS to the matching PartField labels NPY}"

OUTPUT_DIR="${OUTPUT_DIR:-apple_primitive_v32_auto_contact}"
CLUSTERS="${CLUSTERS:-3}"

python -m partfield_mc.cli \
  "$SOURCE_GLB" \
  -o "$OUTPUT_DIR" \
  --postprocess-only \
  --normalized-mesh "$NORMALIZED_MESH" \
  --labels "$LABELS" \
  --clusters "$CLUSTERS" \
  --category generic \
  --up-axis y \
  --forward-axis auto \
  --fit-mode primitive \
  --primitive-part-mode auto \
  --obj-mode surface \
  --primitive-types ellipsoid,convex,frustum,cone,prism \
  --primitive-target-faces 0 \
  --primitive-max-faces 72 \
  --primitive-max-sides 24 \
  --primitive-fit-samples 3500 \
  --primitive-complexity-weight 0.012 \
  --primitive-contact-mode auto \
  --primitive-contact-weak-threshold 0.28 \
  --primitive-contact-strong-threshold 0.62 \
  --primitive-contact-min-edge-count 8 \
  --primitive-contact-medium-mode connector \
  --primitive-contact-overlap-ratio 0 \
  --primitive-interface-max-sides 12 \
  --primitive-interface-min-width-ratio 0.004 \
  --primitive-interface-plane-tolerance-ratio 0.000001 \
  --primitive-surface-main-body-min-area-ratio 0.35 \
  --primitive-surface-boundary-rings 0 \
  --primitive-surface-search-steps 24 \
  --primitive-surface-min-reduction-ratio 0.50 \
  --primitive-surface-hard-max-faces 512 \
  --primitive-validation-policy repair \
  --face-resolution 96 \
  --surface-samples 500000 \
  --palette-size 0 \
  --texture-filter bilinear \
  --uv-wrap clamp \
  --padding 2
