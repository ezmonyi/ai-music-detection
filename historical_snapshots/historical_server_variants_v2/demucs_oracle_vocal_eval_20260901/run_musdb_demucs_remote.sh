#!/usr/bin/env bash
set -euo pipefail

experiment_dir=/mnt/nfs-code/users/yi/demucs_oracle_vocal_eval_20260901
previous_dir=/home/yi/code/ai_music_stem_pilot_20260813

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="$previous_dir/cache/huggingface"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$experiment_dir/runtime:$previous_dir/vendor"

mkdir -p "$experiment_dir/output"
python3 -m demucs.separate \
  --name htdemucs \
  --two-stems vocals \
  --device cuda \
  --shifts 0 \
  --overlap 0.25 \
  --segment 7 \
  --jobs 1 \
  --out "$experiment_dir/output" \
  "$experiment_dir"/input/*.wav
