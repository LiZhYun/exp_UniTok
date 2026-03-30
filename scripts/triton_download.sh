#!/bin/bash
#SBATCH --job-name=datacomp_dl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err

# ── Environment ────────────────────────────────────────────────────────────────
module load mamba
module load triton-dev/2025.1-gcc
module load gcc/13.3.0

export HF_HOME=$WRKDIR/.huggingface_cache
export PIP_CACHE_DIR=$WRKDIR/.pip_cache
export CONDA_ENVS_PATH=$WRKDIR/.conda_envs

source activate unitok   # <-- change to your conda env name

# ── Config ─────────────────────────────────────────────────────────────────────
# ~50% of metadata URLs succeed as actual downloads.
# 2,000,000 metadata pairs → ~1,000,000 images → ~50GB at ~50KB/image avg.
# Reduce to 1,000,000 for ~25GB, or increase to 4,000,000 for ~100GB.
NUM_PAIRS=2000000

OUTPUT_DIR="$WRKDIR/data/datacomp"

# ── Setup ──────────────────────────────────────────────────────────────────────
pip install img2dataset datasets pandas pyarrow --quiet

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$OUTPUT_DIR" logs

# ── Download ───────────────────────────────────────────────────────────────────
echo "Downloading ${NUM_PAIRS} metadata pairs -> images at ${OUTPUT_DIR}/"
python "$PROJ_DIR/scripts/download_datacomp_small.py" \
    --output_dir "$OUTPUT_DIR" \
    --num_pairs "$NUM_PAIRS" \
    --processes_count 16 \
    --thread_count 64

echo ""
echo "Final size:"
du -sh "$OUTPUT_DIR/shards/"
