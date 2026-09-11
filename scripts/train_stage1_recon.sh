#!/bin/bash
# Stage 1 — PePE Reconstruction (paper Sec. 4.1): Gaussian-refinement-guided
# initialization + monocular-depth constrained learning, 30k iterations.
#
# Usage: bash scripts/train_stage1_recon.sh <dataset_dir> <scene> [<scene> ...]
#   DEPTH_FROM / DECOMPOSE_FROM default to the values used for each benchmark
#   in the paper (see README "Reproducing the paper" for the exact settings).
set -e
source "$(dirname "$0")/common.sh"
DATASET="$1"; shift

DEPTH_FROM="${DEPTH_FROM:-0}"
DECOMPOSE_FROM="${DECOMPOSE_FROM:-4000}"
EXTRA_ARGS="${EXTRA_ARGS:---eval}"

train_one() {
  local scene="$1" gpu="$2"
  local src="${DATA_ROOT}/${DATASET}/${scene}"
  local out="${OUTPUT_ROOT}/${DATASET}_pepe/${scene}_pepe"
  [ -d "$src" ] || { echo "skip ${scene}: ${src} not found"; return 0; }
  mkdir -p "$out"
  echo "launch ${scene} (stage 1) on GPU ${gpu} -> ${out}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python train_pepe_reconstruction.py \
      -s "$src" \
      --output_folder "$out" \
      --depth_from "$DEPTH_FROM" \
      --decompose_from "$DECOMPOSE_FROM" \
      $EXTRA_ARGS
  ) > "${out}/train_recon_log.txt" 2>&1 &
}

run_batched train_one "$@"
echo "stage 1 complete"
