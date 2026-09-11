#!/usr/bin/env bash
set -uo pipefail

heuristic_root=${1:?usage: run_allinone_shards.sh HEURISTIC_ROOT BENCHMARK_ROOT}
benchmark_root=${2:?usage: run_allinone_shards.sh HEURISTIC_ROOT BENCHMARK_ROOT}
infer=${benchmark_root}/venv/bin/all-in-one-infer

mkdir -p \
  "${heuristic_root}/inference/structure" \
  "${heuristic_root}/inference/demix" \
  "${heuristic_root}/inference/spec" \
  "${heuristic_root}/logs"

pids=()
for gpu in $(seq 0 7); do
  shard=$(printf '%s/inference_shards/shard_%02d.txt' "${heuristic_root}" "${gpu}")
  log=$(printf '%s/logs/allinone_shard_%02d.log' "${heuristic_root}" "${gpu}")
  mapfile -t audio_files < "${shard}"
  (
    set -o pipefail
    CUDA_VISIBLE_DEVICES=${gpu} "${infer}" "${audio_files[@]}" \
      -o "${heuristic_root}/inference/structure" \
      -m harmonix-all \
      -d cuda \
      -k \
      --demix-dir "${heuristic_root}/inference/demix" \
      --spec-dir "${heuristic_root}/inference/spec" \
      2>&1 | tee "${log}"
  ) &
  pids+=("$!")
  printf 'started gpu=%d pid=%d shard=%s files=%d\n' \
    "${gpu}" "${pids[-1]}" "${shard}" "${#audio_files[@]}"
done

failed=0
for gpu in $(seq 0 7); do
  if wait "${pids[${gpu}]}"; then
    printf 'completed gpu=%d status=0\n' "${gpu}"
  else
    status=$?
    printf 'completed gpu=%d status=%d\n' "${gpu}" "${status}"
    failed=1
  fi
done

exit "${failed}"
