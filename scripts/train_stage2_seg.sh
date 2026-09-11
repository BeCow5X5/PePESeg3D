#!/bin/bash
# Stage 2 — PePE Contrastive Learning (paper Sec. 4.2), 10k iterations on the
# frozen Stage-1 geometry.
#
# Ablations (paper Table 7) are selected through EXTRA_ARGS:
#   (none)                        full model
#   --ablate_perception_loss      drop the perception contrastive loss L_p (Sec. 4.2.2)
#   --ablate_consistency_loss     drop the view-consistent centroid loss L_c (Sec. 4.2.3)
#
# Usage: bash scripts/train_stage2_seg.sh <dataset_dir> <scene> [<scene> ...]
set -e
source "$(dirname "$0")/common.sh"
DATASET="$1"; shift
EXTRA_ARGS="${EXTRA_ARGS:---eval}"
SCRIPT=train_pepe_contrastive.py

train_one() {
  local scene="$1" gpu="$2"
  local out="${OUTPUT_ROOT}/${DATASET}_pepe/${scene}_pepe"
  [ -d "$out" ] || { echo "skip ${scene}: Stage-1 model ${out} not found"; return 0; }
  echo "launch ${scene} (stage 2) on GPU ${gpu}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python "$SCRIPT" \
      -m "$out" \
      --iterations "$SEG_ITERS" \
      --num_sampled_rays "$SEG_RAYS" \
      $EXTRA_ARGS
  ) > "${out}/train_seg_log.txt" 2>&1 &
}

run_batched train_one "$@"
echo "stage 2 complete"
