#!/bin/sh
set -eu
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
TASK_ROOT=/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907
DATA_ROOT=/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907
TASK_PY=/mnt/nfs-code/users/yi/dynamics_rhythm_external_benchmark_20260903/venv/bin/python
i=0
while ps -p 2884419 -o args= | grep -q prepare_native30_evaluation_inputs_bc_v1.py; do
    i=$((i+1))
    test "$i" -lt 240 || exit 2
    sleep 15
done
test -f "$DATA_ROOT/native30_bc_evaluation_package_v1_20260911/COMMIT.json"
PACKAGE_SHA=$(sha256sum "$DATA_ROOT/native30_bc_evaluation_package_v1_20260911/COMMIT.json" | cut -d ' ' -f 1)
echo "Verified package exists, starting deep preflight: $PACKAGE_SHA"
"$TASK_PY" "$TASK_ROOT/code/evaluate_native30_bc_v1.py" --package "$DATA_ROOT/native30_bc_evaluation_package_v1_20260911" --package-commit-sha256 "$PACKAGE_SHA" --output "$DATA_ROOT/native30_bc_evaluation_v1_20260911" --mode preflight > "$TASK_ROOT/preregistration/native30_bc_evaluation_preflight_r2_20260911.json"
echo 'Deep preflight passed; actual parent review and fit freeze remain required.'
