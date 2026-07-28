#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$PWD/PartField}"
ENV_NAME="${PARTFIELD_ENV:-partfield}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda was not found." >&2
  exit 1
fi

if [ ! -d "$ROOT/.git" ]; then
  git clone https://github.com/nv-tlabs/PartField.git "$ROOT"
fi

# Use the versions documented by the official repository.
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda create -n "$ENV_NAME" python=3.10 -y
fi
conda activate "$ENV_NAME"
conda install nvidia/label/cuda-12.4.0::cuda -y
python -m pip install psutil
python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install lightning==2.2 h5py yacs trimesh scikit-image loguru boto3
python -m pip install mesh2sdf tetgen pymeshlab plyfile einops libigl polyscope potpourri3d simple_parsing arrgh open3d
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
python -m pip install vtk scipy pillow huggingface_hub

if command -v sudo >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y libx11-6 libgl1 libxrender1
fi

mkdir -p "$ROOT/model"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/download_checkpoint.py" --output "$ROOT/model/model_objaverse.ckpt"

echo
echo "Installed PartField at: $ROOT"
echo "Conda environment: $ENV_NAME"
echo "Activate with: conda activate $ENV_NAME"
