#!/bin/bash
set -e

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

SHARD_DIR="data/datacomp_small/shards"

# auto-detect shard count
NUM_SHARDS=$(ls "${SHARD_DIR}"/*.tar 2>/dev/null | wc -l)
if [ "$NUM_SHARDS" -eq 0 ]; then
    echo "Error: no tar shards found in ${SHARD_DIR}/"
    echo "Run:  bash scripts/setup_test.sh  first."
    exit 1
fi
LAST=$((NUM_SHARDS - 1))
SHARD_PATTERN="${SHARD_DIR}/{00000..$(printf '%05d' $LAST)}.tar"
echo "Found ${NUM_SHARDS} shards -> ${SHARD_PATTERN}"

# estimate total available samples (assume ~1000 per shard)
TRAIN_SAMPLES=$((NUM_SHARDS * 1000))

# common args (single GPU, small batch for 12GB VRAM)
COMMON="
    --local_bs 4
    --vae_local_bs 4
    --vocab_size 32768
    --num_codebooks 8
    --model vitamin_base
    --train_data ${SHARD_PATTERN}
    --train_num_samples ${TRAIN_SAMPLES}
    --workers 2
    --report_wandb False
    --grad_ckpt True
    --bf16 True
    --vis_img_dir assets/vis_imgs/
    --eval_per_epoch 1
    --epoch 1
"

LAUNCH="torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0
    --master_addr=localhost --master_port=29500"

# === Run 1: Baseline ===
echo ""
echo "=========================================="
echo "  Baseline (standard VQ)"
echo "=========================================="
$LAUNCH main.py $COMMON \
    --output_dir local_output/baseline \
    --exp_name baseline \
    --use_percent_gate False \
    "$@"

# === Run 2: Percent-Gated ===
echo ""
echo "=========================================="
echo "  Percent-Gated Selective Quantization"
echo "=========================================="
$LAUNCH main.py $COMMON \
    --output_dir local_output/percent_gated \
    --exp_name percent_gated \
    --use_percent_gate True \
    --percent_warmup_frac 0.7 \
    --percent_schedule cosine \
    "$@"

echo ""
echo "=== Both runs complete ==="
echo "Baseline:      local_output/baseline/"
echo "Percent-gated: local_output/percent_gated/"
