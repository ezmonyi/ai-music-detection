#!/usr/bin/env bash
# Final coordinated resume. This never starts implicitly: bulk NFS extraction,
# view generation, and the final rehash must be scheduled with inference.
set -uo pipefail

CODE_ROOT=/mnt/nfs-code/users/yi/source_diversity_expansion_20260905
DATA_ROOT=/mnt/nfs-data/users/yi/source_diversity_expansion_20260905
FFMPEG_ROOT=/mnt/nfs-code/users/yi/jujitsu-caption-benchmark/.ffmpeg-env/bin
LOG="$CODE_ROOT/logs/all_supervisor.log"

if [[ "${1:-}" != "--coordinated-start" ]]; then
  echo "Refusing implicit start. Coordinate bulk NFS, then pass --coordinated-start." >&2
  exit 64
fi

exec 9>"$CODE_ROOT/logs/all_supervisor.lock"
if ! flock -n 9; then
  echo "$(date -Is) another coordinated runner holds the lock" >> "$LOG"
  exit 65
fi
if pgrep -af "$CODE_ROOT/code/materialize_all_selected.py" >> "$LOG"; then
  echo "$(date -Is) refusing duplicate canonical materializer" >> "$LOG"
  exit 66
fi
if pgrep -af 'materialize_prefetch_(direct|parquet).py' >> "$LOG"; then
  echo "$(date -Is) acquisition prefetch still active" >> "$LOG"
  exit 67
fi

for archive in mp3.zip medleydb.tar.gz moisesdb.tar.gz genres.tar.gz; do
  if [[ ! -s "$DATA_ROOT/cache/archives/$archive" ]]; then
    echo "$(date -Is) required archive not ready: $archive" >> "$LOG"
    exit 68
  fi
done
parquet_count=$(find "$DATA_ROOT/cache/parquet" -maxdepth 1 -name '*.parquet' | wc -l)
if [[ "$parquet_count" -ne 100 ]]; then
  echo "$(date -Is) expected 100 Parquet shards, found $parquet_count" >> "$LOG"
  exit 69
fi

export PYTHONPATH="$DATA_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
for pass in 1 2 3 4; do
  echo "$(date -Is) coordinated materialization pass $pass starting" >> "$LOG"
  python3 "$CODE_ROOT/code/materialize_all_selected.py" \
    --manifest "$CODE_ROOT/manifests/v2/frozen_item_manifest.csv" \
    --output-root "$DATA_ROOT" \
    --ffmpeg "$FFMPEG_ROOT/ffmpeg" --ffprobe "$FFMPEG_ROOT/ffprobe" \
    --workers 10 --checkpoint-every 10 \
    >> "$CODE_ROOT/logs/all_pass_${pass}.log" 2>&1
  rc=$?
  echo "$(date -Is) materialization pass $pass rc=$rc" >> "$LOG"
  if [[ "$rc" -eq 0 ]]; then
    python3 "$CODE_ROOT/code/materialize_validate.py" \
      --frozen "$CODE_ROOT/manifests/v2/frozen_item_manifest.csv" \
      --materialized "$DATA_ROOT/materialization_manifest.jsonl" \
      --output "$DATA_ROOT/materialization_validation.json" --rehash \
      >> "$CODE_ROOT/logs/final_validation.log" 2>&1
    validation_rc=$?
    echo "$(date -Is) full rehash validation rc=$validation_rc" >> "$LOG"
    if [[ "$validation_rc" -eq 0 ]]; then
      echo "$(date -Is) all frozen items materialized and verified" >> "$LOG"
      exit 0
    fi
  fi
  sleep 15
done

echo "$(date -Is) unresolved items remain after four isolated retry passes" >> "$LOG"
exit 2
