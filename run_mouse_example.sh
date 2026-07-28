#!/usr/bin/env bash
set -euo pipefail

python run_partfield_mc.py "glb/mouse.glb" \
  -o mouse_partfield_mc \
  --partfield-repo /mnt/e/yp/PartField \
  --checkpoint /mnt/e/yp/PartField/model/model_objaverse.ckpt \
  --clusters 9 \
  --category animal \
  --fit-mode obb \
  --face-resolution 64 \
  --surface-samples 500000 \
  --palette-size 0
