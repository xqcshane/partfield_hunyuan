#!/usr/bin/env bash
set -euo pipefail

PARTFIELD_REPO="${PARTFIELD_REPO:-/mnt/e/yp/PartField}"
INPUT_MODEL="${1:-fox_result/hunyuan3d_textured_multiview.glb}"
OUTPUT_DIR="${2:-fox_primitive_paper_result}"
CLUSTERS="${CLUSTERS:-8}"

python -m partfield_mc.cli \
  "$INPUT_MODEL" \
  --partfield-repo "$PARTFIELD_REPO" \
  --checkpoint "$PARTFIELD_REPO/model/model_objaverse.ckpt" \
  --simplify-faces 5000 \
  --n-point-per-face 50 \
  --n-sample-each 1000 \
  --clusters "$CLUSTERS" \
  --category animal \
  --fit-mode primitive \
  --obj-mode surface \
  --primitive-types auto \
  --primitive-target-faces 0 \
  --primitive-max-faces 48 \
  --primitive-max-sides 24 \
  --primitive-fit-samples 2500 \
  --primitive-contact-mode fixed \
  --primitive-interface-max-sides 8 \
  --primitive-interface-min-width-ratio 0.006 \
  --primitive-contact-overlap-ratio 0 \
  --face-resolution 64 \
  --surface-samples 500000 \
  -o "$OUTPUT_DIR"
