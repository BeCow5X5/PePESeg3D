#!/bin/bash
# Stage 0 — perception priors.
#   1. SAM ViT-H "segment everything" masks           -> <scene>/sam_masks
#   2. single dense map (used by PePE Reconstruction) -> <scene>/single_dense_maps
#   3. physical mask scales, Eq. (2)                  -> <scene>/mask_scales
# Monocular depth (<scene>/depth) is produced separately with Depth-Anything-V2; see README.
#
# Usage: bash scripts/preprocess.sh <dataset_dir> <scene> [<scene> ...]
set -e
source "$(dirname "$0")/common.sh"
DATASET="$1"; shift
for scene in "$@"; do
  echo "=== preprocessing ${DATASET}/${scene} ==="
  python extract_masks_and_scales.py \
    --source_path "${DATA_ROOT}/${DATASET}/${scene}" \
    --model_path  "${OUTPUT_ROOT}/${DATASET}_pepe/${scene}_pepe" \
    --sam_checkpoint_path "${SAM_CKPT}"
done
