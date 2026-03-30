#!/bin/bash
set -e

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

echo "=== UniTok Test Setup ==="

# 1. Download external models (LPIPS-VGG, DINOv2, Inception for FID)
echo "[1/2] Downloading external models from HuggingFace ..."
mkdir -p external
python -c "from huggingface_hub import snapshot_download; snapshot_download('FoundationVision/unitok_external', local_dir='external', local_dir_use_symlinks=False)"
echo "  Saved to external/"
ls -lh external/

# 2. Download small DataComp-1B subset (~1GB)
echo ""
echo "[2/2] Downloading small DataComp-1B subset ..."
pip install img2dataset datasets pandas pyarrow -q
python scripts/download_datacomp_small.py \
    --output_dir data/datacomp_small \
    --num_pairs 20000

echo ""
echo "=== Setup Complete ==="
echo "External models:  external/"
echo "Training data:    data/datacomp_small/shards/"
echo ""
echo "Run:  bash scripts/test_train.sh"
