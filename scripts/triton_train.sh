#!/bin/bash
#SBATCH --job-name=unitok_train
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=0-23:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --output=logs/unitok_%A_%a.out
#SBATCH --error=logs/unitok_%A_%a.err

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
export WANDB_CACHE_DIR=$WRKDIR/wandb_cache

mkdir -p "$WANDB_CACHE_DIR"

source activate unitok   # <-- change to your conda env name

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJ_DIR="/scratch/work/liz23/exp_UniTok"
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

# ── Ensure vis_imgs directory has sample images ───────────────────────────────
VIS_DIR="${OUTPUT_DIR}/vis_imgs"
if [ ! -d "$VIS_DIR" ] || [ -z "$(ls -A "$VIS_DIR" 2>/dev/null)" ]; then
    mkdir -p "$VIS_DIR"
    echo "Extracting sample images for visualization ..."
    python -c "
import tarfile, os, sys
shard = '${SHARD_DIR}/00000.tar'
out = '${VIS_DIR}'
with tarfile.open(shard) as t:
    jpgs = [m for m in t.getmembers() if m.name.endswith('.jpg')][:8]
    for m in jpgs:
        m.name = os.path.basename(m.name)
        t.extract(m, out)
print(f'Extracted {len(jpgs)} images to {out}')
"
fi

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

# Use unique master_port per array task to avoid collisions
MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID))
LAUNCH="torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0
    --master_addr=localhost --master_port=${MASTER_PORT}"

# ── Select experiment by array task ID ─────────────────────────────────────────
case $SLURM_ARRAY_TASK_ID in
    0)
        echo "=========================================="
        echo "  Baseline (standard VQ)"
        echo "=========================================="
        export WANDB_DIR=${OUTPUT_DIR}/baseline/wandb
        mkdir -p "$WANDB_DIR"
        $LAUNCH main.py $COMMON \
            --output_dir ${OUTPUT_DIR}/baseline \
            --exp_name baseline \
            --use_percent_gate False \
            "$@"
        ;;
    1)
        echo "=========================================="
        echo "  Percent-Gated Selective Quantization"
        echo "=========================================="
        export WANDB_DIR=${OUTPUT_DIR}/percent_gated/wandb
        mkdir -p "$WANDB_DIR"
        $LAUNCH main.py $COMMON \
            --output_dir ${OUTPUT_DIR}/percent_gated \
            --exp_name percent_gated \
            --use_percent_gate True \
            --percent_warmup_frac 0.7 \
            --percent_schedule cosine \
            "$@"
        ;;
esac
