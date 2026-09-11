#!/bin/bash
# LERF-Mask evaluation (paper Table 2, coarse) and LERF-Mask-Fine (Table 3).
# Usage: [GRANULARITY=coarse|fine] bash scripts/eval_lerf_mask.sh [<scene> ...]
set -e
source "$(dirname "$0")/common.sh"
GRANULARITY="${GRANULARITY:-coarse}"
if [ "$GRANULARITY" = "fine" ]; then
  SCRIPT=eval_lerf_mask_fine.py; DATASET=lerf_mask_fine
else
  SCRIPT=eval_lerf_mask.py;      DATASET=lerf_mask
fi
SCENES=("$@"); [ ${#SCENES[@]} -eq 0 ] && SCENES=(figurines ramen teatime)
mkdir -p logs/eval
for scene in "${SCENES[@]}"; do
  echo "=== eval ${scene} (${GRANULARITY}) ==="
  # NOTE: do not pass --iteration; the Stage-1 (30k) and Stage-2 (10k) checkpoints
  # are resolved separately and a single value cannot address both.
  python "$SCRIPT" \
    -m "${OUTPUT_ROOT}/lerf_mask_pepe/${scene}_pepe" \
    -s "${DATA_ROOT}/${DATASET}/${scene}" \
    --train_split --skip_train --use_hdbscan \
    --gt_folder_path "${DATA_ROOT}/${DATASET}/${scene}/test_mask" \
    2>&1 | tee "logs/eval/${scene}_${GRANULARITY}.log" | grep -vE "^Reading camera" | tail -20
done
