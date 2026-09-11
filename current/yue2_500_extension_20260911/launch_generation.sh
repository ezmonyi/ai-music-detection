#!/bin/sh
# Server-only staged launcher. Never modifies the existing experiment runtime.
set -eu
TASK_ROOT=/mnt/nfs-code/users/yi/yue2_500_extension_20260911
DATA_ROOT=/mnt/nfs-data/users/yi/yue2_500_extension_20260911
cd "$TASK_ROOT"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export TMPDIR="$DATA_ROOT/tmp"
export HF_HUB_CACHE="$DATA_ROOT/hf_cache"
mkdir -p "$DATA_ROOT/raw" logs
# Wait for the single installation already launched by the parent, bounded.
i=0
while ps -p 2622229 -o args= | grep -q 'pip --python'; do
    i=$((i + 1))
    if [ "$i" -gt 120 ]; then
        echo 'Installation wait exceeded 30 minutes; no generation started.'
        exit 1
    fi
    sleep 15
done
venv/bin/python -m pip check
venv/bin/python -m pip freeze > logs/runtime_freeze.txt
test "$(git -C YuE rev-parse HEAD)" = 92a73cc7652fcc1f937855e4b765e0a0edd7ff2e
git -C YuE diff --exit-code
sha256sum run_yue2_frozen.py prompts/prompt_manifest.jsonl > logs/input_hashes.txt
venv/bin/python -c 'import torch; from yue2 import YuE2Pipeline; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); assert "5090" in torch.cuda.get_device_name(0)'
echo 'Starting first frozen prompt compatibility smoke.'
venv/bin/python run_yue2_frozen.py --manifest prompts/prompt_manifest.jsonl --manifest-sha256 bb0941b91d41d10a978cb9e4b077399e89fea92526ee469e0e6743b09efb917b --output "$DATA_ROOT/raw" --start 0 --stop 1
echo 'Smoke completed. Starting fixed 500-prompt generation, reusing the first output.'
venv/bin/python run_yue2_frozen.py --manifest prompts/prompt_manifest.jsonl --manifest-sha256 bb0941b91d41d10a978cb9e4b077399e89fea92526ee469e0e6743b09efb917b --output "$DATA_ROOT/raw"
echo 'Generation loop completed; downstream analysis and eligibility audit are still required.'
