#!/usr/bin/env python3
"""Capture reproducibility/provenance metadata without changing experiment state."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def command(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def optional_command(*args: str) -> str:
    try:
        return command(*args)
    except Exception as exc:
        return f"ERROR: {exc!r}"


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = args.experiment_root.resolve()
    files = {
        "prompt_manifest": root / "prompts/frozen_500/prompt_manifest.jsonl",
        "prompt_manifest_metadata": root / "prompts/frozen_500/prompt_manifest_metadata.json",
        "selected_source_rows": root / "prompts/frozen_500/selected_source_rows.jsonl",
        "heartmula_dependencies": root / "logs/heartmula_pip_freeze.txt",
        "acestep_dependencies": root / "logs/acestep_pip_freeze.txt",
        "demucs_runtime_dependencies": root / "logs/demucs_runtime_pip_freeze.txt",
        "heartmula_weight_hashes": root / "logs/heartmula_weight_sha256.txt",
        "acestep_weight_hashes_before_handler_sync": root
        / "logs/acestep_weight_sha256.txt",
        "acestep_weight_hashes_final": root
        / "logs/acestep_weight_sha256_final.txt",
        "demucs_bias": root / "analysis/demucs_frequency_bias.npz",
        "muse_train_cn": root / "prompts/train_cn.jsonl",
        "muse_train_en": root / "prompts/train_en.jsonl",
        "protocol": root / "EXPERIMENT_PROTOCOL.md",
        "research_record": root / "MODEL_DATASET_RESEARCH.md",
        "run_log": root / "RUN_LOG.md",
        "reproduce_commands": root / "REPRODUCE_COMMANDS.md",
        "thesis_results_cn": root / "THESIS_RESULTS_CN.md",
        "failed_torchcodec_standardization": root
        / "state/acestep_standardization_failed_torchcodec.jsonl",
        "successful_acestep_standardization": root / "state/acestep_standardization.jsonl",
        "acestep_standardization_summary": root
        / "reports/acestep_standardization_summary.json",
        "successful_heartmula_standardization": root / "state/heartmula_standardization.jsonl",
        "acestep_generation_validation": root / "reports/acestep_generation_validation.json",
        "heartmula_generation_validation": root / "reports/heartmula_generation_validation.json",
        "heartmula_strict_duration_failure": root
        / "reports/heartmula_generation_validation_raw_strict_failed.json",
        "heartmula_standardization_summary": root
        / "reports/heartmula_standardization_summary.json",
        "heartmula_reconciled_generation_state": root
        / "state/heartmula_generation_reconciled.jsonl",
        "demucs_acestep_state": root / "state/demucs_acestep.jsonl",
        "demucs_heartmula_state": root / "state/demucs_heartmula.jsonl",
        "acestep_analysis_report": root / "analysis/human_vs_acestep/REPORT.md",
        "heartmula_analysis_report": root / "analysis/human_vs_heartmula/REPORT.md",
        "paired_analysis_report": root / "analysis/paired_generators/REPORT.md",
        "generator_comparison_report": root / "analysis/comparison/REPORT.md",
        "cross_generator_report": root / "analysis/cross_generator/REPORT.md",
        "cross_generator_vocal_active_report": root
        / "analysis/cross_generator_vocal_active/REPORT.md",
        "heartmula_duration_robust_report": root
        / "analysis/heartmula_duration_robust/REPORT.md",
    }
    code_roots = {
        "heartlib": root / "code/heartlib",
        "acestep": root / "code/ACE-Step-1.5",
        "heartmula_benchmark": root / "code/HeartMuLa-Benchmark",
    }
    metadata = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_root": str(root),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "nvidia_smi": optional_command(
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ),
            "filesystem": optional_command("df", "-h", str(root)),
        },
        "code_revisions": {
            name: optional_command("git", "-C", str(path), "rev-parse", "HEAD")
            for name, path in code_roots.items()
        },
        "artifact_sha256": {name: sha256_file(path) for name, path in files.items()},
        "experiment_code_sha256": {
            path.name: sha256_file(path)
            for path in sorted((root / "code/experiment").glob("*"))
            if path.is_file()
        },
        "frozen_model_revisions": {
            "HeartMuLa/HeartMuLaGen": "9906b2bcd4598772a32cad4aec0760170fe0d177",
            "HeartMuLa/HeartMuLa-oss-3B-happy-new-year": "41f6fc68490e11dc43fdabaa6b5767946408c903",
            "HeartMuLa/HeartCodec-oss-20260123": "f889dab0532cfa4bf459f2a3367eb6d346b8eeda",
            "ACE-Step/Ace-Step1.5": "19671f406d603126926c1b7e2adc169acbcade22",
            "bolshyC/Muse": "b1bf3bf906daab3a896e14f6dea58cc295848452",
        },
        "frozen_external_inputs": {
            "suno_human_baseline_root": "/mnt/nfs-code/users/yi/demucs_bias_corrected_1000_20260901",
            "suno_human_cohort": "500 Human + 500 Suno; development 400+400; locked test 100+100",
            "demucs_bias_sha256": "bcacfecac5ce69927dfed2a51ec21cc346f613a7874c519e419d22d35a97de5e",
        },
        "inference": {
            "duration_seconds": 30,
            "heartmula": {
                "topk": 50,
                "temperature": 1.0,
                "cfg_scale": 1.5,
                "mula_dtype": "bfloat16",
                "codec_dtype": "float32",
                "peak_normalization": False,
            },
            "acestep": {
                "variant": "acestep-v15-turbo",
                "inference_steps": 8,
                "thinking": False,
                "cot_rewrites": False,
                "peak_normalization": False,
                "dcw": {
                    "mode": "double",
                    "low_scaler": 0.05,
                    "high_scaler": 0.02,
                    "wavelet": "haar",
                },
            },
            "standardization": "SoundFile + scipy.signal.resample_poly; stereo, 44100 Hz, exact 30 s, PCM_S16 FLAC; no loudness normalization or EQ",
            "heartmula_workers": {
                "initial": "three disjoint workers: 0-166 on GPUs 0,1; 167-333 on 4,5; 334-499 on 6,7",
                "rebalanced_after_158_valid_outputs": "49-134 on 0,1; 135-277 on 2,3; 278-414 on 4,5; 415-499 on 6,7; existing valid outputs skipped",
                "interruption_policy": "partials removed and identical item seed retried; historical error events preserved",
            },
            "demucs": "Demucs 4.0.1 and torch/torchaudio 2.7.1+cu128; htdemucs, two-stems=vocals, float32, shifts=0, overlap=0.25, segment=7; GPU 3",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
