#!/usr/bin/env python3
"""Rank fixed 30 s candidate clips by singing-vocal evidence with AudioSet AST."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import AutoFeatureExtractor, ASTForAudioClassification


VOCAL_LABELS = (27, 28, 32, 33, 34, 35, 36, 37, 254)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def windows(path: Path) -> list[np.ndarray]:
    waveform, sr = torchaudio.load(path)
    waveform = waveform.mean(0)
    if sr != 16_000:
        waveform = torchaudio.functional.resample(waveform, sr, 16_000)
    target = 160_000
    if waveform.numel() < target * 3:
        waveform = torch.nn.functional.pad(waveform, (0, target * 3 - waveform.numel()))
    return [waveform[offset * target:(offset + 1) * target].numpy() for offset in range(3)]


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_dir.glob("*.flac"))
    if not paths:
        raise SystemExit(f"no FLAC files in {args.input_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = AutoFeatureExtractor.from_pretrained(args.model_dir, local_files_only=True)
    model = ASTForAudioClassification.from_pretrained(args.model_dir, local_files_only=True).to(device).eval()
    rows = []
    pending_audio: list[np.ndarray] = []
    pending_meta: list[tuple[str, int]] = []
    scores: dict[str, list[float]] = {path.stem: [] for path in paths}

    def flush() -> None:
        if not pending_audio:
            return
        inputs = extractor(pending_audio, sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**{key: value.to(device) for key, value in inputs.items()}).logits
            probabilities = torch.sigmoid(logits[:, list(VOCAL_LABELS)]).amax(dim=1).cpu().numpy()
        for (track, _), score in zip(pending_meta, probabilities):
            scores[track].append(float(score))
        pending_audio.clear()
        pending_meta.clear()

    for index, path in enumerate(paths, 1):
        for window_number, audio in enumerate(windows(path)):
            pending_audio.append(audio)
            pending_meta.append((path.stem, window_number))
            if len(pending_audio) >= args.batch_size:
                flush()
        if index == 1 or index % 25 == 0 or index == len(paths):
            print(f"screen {index}/{len(paths)}", flush=True)
    flush()

    for path in paths:
        values = scores[path.stem]
        rows.append({
            "track": path.stem,
            "vocal_score": float(np.mean(values)),
            "vocal_score_max": float(np.max(values)),
            "positive_windows": int(sum(value >= 0.5 for value in values)),
            "window_scores": ";".join(f"{value:.8f}" for value in values),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
