#!/usr/bin/env bash
set -euo pipefail

if (( $# < 4 )); then
  echo "usage: infer_expanded_allinone.sh PREPARED_DIR OUTPUT_ROOT RUNTIME_ROOT GPU [GPU ...]" >&2
  exit 2
fi
prepared=$1
output_root=$2
runtime_root=$3
shift 3
gpus=("$@")
infer=${runtime_root}/venv/bin/all-in-one-infer
[[ -x "$infer" ]] || { echo "missing executable: $infer" >&2; exit 2; }

mkdir -p "$output_root/structure" "$output_root/demix" "$output_root/spec" "$output_root/logs"
for gpu in "${gpus[@]}"; do
  read -r used util < <(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
  if (( used > 1024 || util > 5 )); then
    echo "refusing busy gpu=$gpu used_mib=$used util_pct=$util" >&2
    exit 3
  fi
done

pids=()
for index in "${!gpus[@]}"; do
  gpu=${gpus[$index]}
  shard=$(printf '%s/inference_shard_%02d.txt' "$prepared" "$index")
  [[ -s "$shard" ]] || { echo "missing/empty shard: $shard" >&2; exit 2; }
  mapfile -t audio_files < "$shard"
  log=$(printf '%s/logs/allinone_gpu%s_shard_%02d.log' "$output_root" "$gpu" "$index")
  (
    set -o pipefail
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="$gpu" \
      "$infer" "${audio_files[@]}" -o "$output_root/structure" -m harmonix-all -d cuda -k \
      --demix-dir "$output_root/demix" --spec-dir "$output_root/spec" 2>&1 | tee "$log"
  ) &
  pids+=("$!")
  echo "started gpu=$gpu pid=${pids[-1]} shard=$index files=${#audio_files[@]}"
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "completed gpu=${gpus[$index]} status=0"
  else
    status=$?; echo "completed gpu=${gpus[$index]} status=$status" >&2; failed=1
  fi
done
exit "$failed"
