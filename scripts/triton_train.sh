#!/bin/bash
#SBATCH --job-name=unitok_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=0-23:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --output=logs/unitok_%j.out
#SBATCH --error=logs/unitok_%j.err

# ── Environment ────────────────────────────────────────────────────────────────
module load mamba
module load triton-dev/2025.1-gcc
module load gcc/13.3.0
module load cuda/12.6.2

export HF_HOME=$WRKDIR/.huggingface_cache
export PIP_CACHE_DIR=$WRKDIR/.pip_cache
export CONDA_PKGS_DIRS=$WRKDIR/.conda_pkgs
export CONDA_ENVS_PATH=$WRKDIR/.conda_envs
export TORCH_EXTENSIONS_DIR=$WRKDIR/torch_extensions
export WANDB_DIR=$WRKDIR/wandb
export WANDB_CACHE_DIR=$WRKDIR/wandb_cache

source activate unitok   # <-- change to your conda env name

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

export PYTHONPATH="${PROJ_DIR}:$PYTHONPATH"

# Directory containing webdataset shards (*.tar) on scratch
SHARD_DIR="${WRKDIR}/data/datacomp/shards"   # <-- adjust to your shard path
OUTPUT_DIR="${WRKDIR}/unitok_output"

# ── Data config (adjust these) ─────────────────────────────────────────────────
SAMPLES_PER_SHARD=1000   # typical for DataComp/CC12M ~100MB shards; check with: tar -t shard.tar | wc -l
EPOCHS=1                 # increase for longer training (2-3 for ablations)

# ── Auto-detect shards ─────────────────────────────────────────────────────────
NUM_SHARDS=$(ls "${SHARD_DIR}"/*.tar 2>/dev/null | wc -l)
if [ "$NUM_SHARDS" -eq 0 ]; then
    echo "Error: no tar shards found in ${SHARD_DIR}/"
    exit 1
fi
LAST=$((NUM_SHARDS - 1))
SHARD_PATTERN="${SHARD_DIR}/{00000..$(printf '%05d' $LAST)}.tar"
TRAIN_SAMPLES=$((NUM_SHARDS * SAMPLES_PER_SHARD))
echo "Found ${NUM_SHARDS} shards, ~${TRAIN_SAMPLES} samples (${SAMPLES_PER_SHARD}/shard), ${EPOCHS} epoch(s) -> ${SHARD_PATTERN}"

mkdir -p logs

# ── Common training args ────────────────────────────────────────────────────────
COMMON="
    --local_bs 32
    --vae_local_bs 32
    --vocab_size 32768
    --num_codebooks 8
    --model vitamin_base
    --train_data ${SHARD_PATTERN}
    --train_num_samples ${TRAIN_SAMPLES}
    --workers 10
    --report_wandb True
    --grad_ckpt True
    --bf16 True
    --vis_img_dir ${OUTPUT_DIR}/vis_imgs/
    --eval_per_epoch 1
    --epoch ${EPOCHS}
"

LAUNCH="torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0
    --master_addr=localhost --master_port=29500"

# ── Run 1: Baseline ─────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Baseline (standard VQ)"
echo "=========================================="
$LAUNCH main.py $COMMON \
    --output_dir ${OUTPUT_DIR}/baseline \
    --exp_name baseline \
    --use_percent_gate False \
    "$@"

# ── Run 2: Percent-Gated ────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Percent-Gated Selective Quantization"
echo "=========================================="
$LAUNCH main.py $COMMON \
    --output_dir ${OUTPUT_DIR}/percent_gated \
    --exp_name percent_gated \
    --use_percent_gate True \
    --percent_warmup_frac 0.7 \
    --percent_schedule cosine \
    "$@"

echo ""
echo "=== Both runs complete ==="
echo "Baseline:      ${OUTPUT_DIR}/baseline/"
echo "Percent-gated: ${OUTPUT_DIR}/percent_gated/"
