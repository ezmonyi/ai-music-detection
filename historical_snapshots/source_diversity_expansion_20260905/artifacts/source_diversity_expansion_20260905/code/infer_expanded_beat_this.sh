#!/usr/bin/env bash
set -euo pipefail

if (( $# != 5 )); then
  echo "usage: infer_expanded_beat_this.sh PREPARED_DIR OUTPUT_ROOT RUNTIME_ROOT GPU SHARD_INDEX" >&2
  exit 2
fi
prepared=$1; output_root=$2; runtime_root=$3; gpu=$4; shard_index=$5
beat_this=${runtime_root}/venv/bin/beat_this
checkpoint=${runtime_root}/checkpoints/hub/checkpoints/beat_this-final0.ckpt
shard=$(printf '%s/inference_shard_%02d.txt' "$prepared" "$shard_index")
[[ -x "$beat_this" && -f "$checkpoint" && -s "$shard" ]] || { echo "missing runtime/checkpoint/shard" >&2; exit 2; }
read -r used util < <(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
if (( used > 1024 || util > 5 )); then
  echo "refusing busy gpu=$gpu used_mib=$used util_pct=$util" >&2; exit 3
fi
mkdir -p "$output_root/beats" "$output_root/logs"
mapfile -t audio_files < "$shard"
log=$(printf '%s/logs/beat_this_gpu%s_shard_%02d.log' "$output_root" "$gpu" "$shard_index")
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="$gpu" \
  "$beat_this" "${audio_files[@]}" -o "$output_root/beats" --model "$checkpoint" \
  --no-dbn --gpu 0 --float16 --skip-existing 2>&1 | tee "$log"
