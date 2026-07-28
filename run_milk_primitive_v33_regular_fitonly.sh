#!/usr/bin/env bash
set -euo pipefail

: "${SOURCE_GLB:?Set SOURCE_GLB to the textured Hunyuan GLB}"
: "${NORMALIZED_MESH:?Set NORMALIZED_MESH to PartField input_..._0.ply}"
: "${LABELS:?Set LABELS to PartField ..._0_NN.npy}"

OUTPUT_DIR="${OUTPUT_DIR:-milk_primitive_v33_regular}"
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
  --obj-mode surface \
  --primitive-template-mode regular \
  --primitive-part-mode auto \
  --primitive-types prism,frustum,ellipsoid,cone \
  --primitive-target-faces 0 \
  --primitive-max-faces 64 \
  --primitive-max-sides 24 \
  --primitive-fit-samples 4000 \
  --primitive-complexity-weight 0.015 \
  --primitive-regularity-weight 0.10 \
  --primitive-contact-mode auto \
  --primitive-contact-proximity-ratio 0.025 \
  --primitive-contact-proximity-min-points 8 \
  --primitive-contact-proximity-min-coverage 0.01 \
  --primitive-contact-weak-threshold 0.12 \
  --primitive-contact-strong-threshold 0.42 \
  --primitive-contact-min-edge-count 6 \
  --primitive-contact-medium-mode connector \
  --primitive-connector-sides 8 \
  --primitive-connector-radius-ratio 0.018 \
  --primitive-connector-inset-ratio 0.35 \
  --primitive-connector-min-length-ratio 0.002 \
  --primitive-regular-connector-max-face-area-ratio 0.05 \
  --primitive-validation-policy repair \
  --face-resolution 128 \
  --surface-samples 750000 \
  --palette-size 0 \
  --texture-filter bilinear \
  --uv-wrap clamp \
  --padding 3
