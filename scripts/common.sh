# Shared configuration for all PePESeg3D pipeline scripts.
# Override any of these from the environment, e.g.  DATA_ROOT=/mnt/data bash scripts/train_spin_nerf.sh
DATA_ROOT="${DATA_ROOT:-../data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output}"
GPUS="${GPUS:-0}"                 # space-separated list, e.g. GPUS="0 1 2 3"
SAM_CKPT="${SAM_CKPT:-./third_party/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth}"
export SAM_CKPT

read -r -a GPU_ARR <<< "$GPUS"

# Stage 2 iteration/sampling budget (paper: 10k iterations, 1000 sampled pixels)
SEG_ITERS="${SEG_ITERS:-10000}"
SEG_RAYS="${SEG_RAYS:-1000}"

run_batched() {
  # run_batched <fn> <item>...   — dispatches items round-robin over $GPU_ARR,
  # waiting whenever every GPU is busy.
  local fn="$1"; shift
  local count=0
  for item in "$@"; do
    "$fn" "$item" "${GPU_ARR[$((count % ${#GPU_ARR[@]}))]}"
    count=$((count + 1))
    if [ $((count % ${#GPU_ARR[@]})) -eq 0 ]; then wait; fi
  done
  wait
}
