#!/usr/bin/env bash
set -euo pipefail

: "${SOURCE_GLB:?Set SOURCE_GLB to the textured Hunyuan3D GLB}"
: "${NORMALIZED_MESH:?Set NORMALIZED_MESH to the matching PartField normalized PLY}"
: "${LABELS:?Set LABELS to the matching PartField labels NPY}"

OUTPUT_DIR="${OUTPUT_DIR:-fox_primitive_v28_closed}"
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
  --primitive-part-mode closed \
  --obj-mode surface \
  --primitive-types auto \
  --primitive-target-faces 0 \
  --primitive-max-faces 48 \
  --primitive-max-sides 24 \
  --primitive-fit-samples 2500 \
  --primitive-contact-mode fixed \
  --primitive-interface-max-sides 8 \
  --primitive-interface-min-width-ratio 0.006 \
  --primitive-interface-plane-tolerance-ratio 0.000001 \
  --face-resolution 64 \
  --surface-samples 500000 \
  --palette-size 0 \
  --texture-filter bilinear \
  --uv-wrap clamp \
  --padding 2
