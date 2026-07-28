#!/usr/bin/env bash
set -euo pipefail

: "${SOURCE_GLB:?Set SOURCE_GLB to the textured Hunyuan GLB}"
: "${NORMALIZED_MESH:?Set NORMALIZED_MESH to PartField input_..._0.ply}"
: "${LABELS:?Set LABELS to PartField ..._0_NN.npy}"

OUTPUT_DIR="${OUTPUT_DIR:-puppy_primitive_v33_regular}"
CLUSTERS="${CLUSTERS:-8}"

python -m partfield_mc.cli \
  "$SOURCE_GLB" \
  -o "$OUTPUT_DIR" \
  --postprocess-only \
  --normalized-mesh "$NORMALIZED_MESH" \
  --labels "$LABELS" \
  --clusters "$CLUSTERS" \
  --category animal \
  --up-axis y \
  --forward-axis auto \
  --fit-mode primitive \
  --obj-mode surface \
  --primitive-template-mode regular \
  --primitive-part-mode closed \
  --primitive-types box,prism,frustum,cone,ellipsoid \
  --primitive-target-faces 0 \
  --primitive-max-faces 64 \
  --primitive-max-sides 24 \
  --primitive-fit-samples 3000 \
  --primitive-complexity-weight 0.020 \
  --primitive-regularity-weight 0.10 \
  --primitive-contact-mode auto \
  --primitive-contact-proximity-ratio 0.015 \
  --primitive-contact-proximity-min-points 6 \
  --primitive-contact-proximity-min-coverage 0.008 \
  --primitive-contact-weak-threshold 0.18 \
  --primitive-contact-strong-threshold 0.50 \
  --primitive-contact-min-edge-count 6 \
  --primitive-contact-medium-mode connector \
  --primitive-connector-sides 4 \
  --primitive-connector-radius-ratio 0.018 \
  --primitive-connector-inset-ratio 0.35 \
  --primitive-regular-connector-max-face-area-ratio 0.06 \
  --primitive-validation-policy repair \
  --face-resolution 64 \
  --surface-samples 500000 \
  --palette-size 0 \
  --texture-filter bilinear \
  --uv-wrap clamp \
  --padding 2
