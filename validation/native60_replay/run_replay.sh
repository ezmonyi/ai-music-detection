#!/usr/bin/env bash
# Run from the code repository root. Requires its installed Python environment.
# Supply an OAuth access token through MUSIC_DETECTOR_DRIVE_TOKEN; never put it
# in this file or in a logged command. The token must read the four pinned IDs.
# Both destinations must be new. Existing files are never overwritten.
#
# Usage:
#   bash validation/native60_replay/run_replay.sh /path/to/new-inputs /path/to/new-report
#
# This verifies persisted AI-only predictions and regenerates sensitivity tables.
# It does not train models, extract features, or establish balanced accuracy.
set -euo pipefail
NATIVE60_INPUT_DIR="${1:?Pass a new temporary input directory}"
NATIVE60_REPORT_DIR="${2:?Pass a new report directory}"
: "${MUSIC_DETECTOR_DRIVE_TOKEN:?Set a read-capable OAuth token in the environment}"
PYTHONPATH=src .venv/bin/python validation/native60_replay/fetch_drive_replay.py \
  --destination "$NATIVE60_INPUT_DIR"
.venv/bin/python current/yue2_500_extension_20260911/report_native60_transfer_v1.py \
  --input-root "$NATIVE60_INPUT_DIR" --output "$NATIVE60_REPORT_DIR"
echo 'Compare generated product hashes against results/native60_transfer_report_v1/COMMIT.json.'
