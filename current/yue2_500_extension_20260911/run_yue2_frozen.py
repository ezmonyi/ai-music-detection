"""Single-candidate YuE2 generation; preserves every attempt, no audio selection."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

MODEL_REV = "1a96eca688d6ae5d7f0feb88573fec89920fcd19"
VAE_REV = "95535e72a97bc0f09b8ada125d26b4009428c0e8"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def requests(path):
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    if len(rows) != 500 or len({r["id"] for r in rows}) != 500:
        raise ValueError("Expected exactly 500 distinct frozen prompt IDs")
    if collections.Counter(r["language"] for r in rows) != {"zh": 250, "en": 250}:
        raise ValueError("Language counts differ from frozen manifest")
    if collections.Counter(r["split"] for r in rows) != {"development": 400, "locked_test": 100}:
        raise ValueError("Split counts differ from frozen manifest")
    for row in rows:
        if Path(row["id"]).name != row["id"] or row["id"] in (".", ".."):
            raise ValueError("Unsafe ID")
        if not row["prompt_common"].strip() or not row["lyrics_30s"].strip():
            raise ValueError("Missing style or lyrics")
    return rows


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=500)
    p.add_argument("--validate-only", action="store_true")
    a = p.parse_args()
    if digest(a.manifest) != a.manifest_sha256:
        raise ValueError("Manifest SHA mismatch")
    rows = requests(a.manifest)
    if not 0 <= a.start < a.stop <= 500:
        raise ValueError("Invalid nonempty index range")
    if a.validate_only:
        print(json.dumps({"rows": len(rows), "range": [a.start, a.stop], "status": "validated"}))
        return
    import soundfile as sf
    import numpy as np
    import torch
    from yue2 import YuE2Pipeline
    if not torch.cuda.is_available() or "5090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This protocol requires an RTX 5090 CUDA device")
    a.output.mkdir(parents=True, exist_ok=True)
    with YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", vae="m-a-p/YuE2-Vae",
                                    revision=MODEL_REV, vae_revision=VAE_REV,
                                    device="cuda") as pipe:
        for index in range(a.start, a.stop):
            row = rows[index]
            root = a.output / row["id"]
            root.mkdir(exist_ok=True)
            receipt = root / "ACCEPTED.json"
            if receipt.exists():
                prior = json.loads(receipt.read_text())
                if prior["manifest_sha256"] != a.manifest_sha256:
                    raise ValueError("Existing result belongs to a different manifest")
                if digest(root / prior["audio"]) != prior["audio_sha256"]:
                    raise ValueError("Existing accepted audio hash mismatch")
                continue
            attempt = root / ("attempt_" + str(time.time_ns()))
            attempt.mkdir()
            request = dict(style=row["prompt_common"], lyrics=row["lyrics_30s"],
                           seed=row["seed"], cot="full")
            meta = dict(index=index, id=row["id"], split=row["split"], request=request,
                        manifest_sha256=a.manifest_sha256, model_revision=MODEL_REV,
                        vae_revision=VAE_REV, gpu=torch.cuda.get_device_name(0))
            save(attempt / "input.json", meta)
            started = time.monotonic()
            try:
                song = pipe(**request)
                song.save_artifacts(attempt / "generation")
                audio = attempt / "generation/audio.flac"
                wave, sr = sf.read(audio, always_2d=True)
                if sr != 48000 or wave.shape[1] != 2 or not len(wave) or not np.isfinite(wave).all():
                    raise ValueError("Invalid native audio shape/rate/values")
                meta.update(audio=str(audio.relative_to(root)), audio_sha256=digest(audio),
                            frames=len(wave), sample_rate=sr, seconds=len(wave)/sr,
                            truncated=song.truncated, elapsed=time.monotonic()-started,
                            status="generated_flagged" if any(song.truncated.values()) else "generated")
                save(attempt / "receipt.json", meta)
                # Accepted means persisted native generation, not scientific eligibility.
                save(receipt, meta)
                print(json.dumps({"index": index, "id": row["id"], "status": meta["status"]}), flush=True)
            except Exception:
                meta.update(status="failed", error=traceback.format_exc(), elapsed=time.monotonic()-started)
                save(attempt / "failure.json", meta)
                print(json.dumps({"index": index, "id": row["id"], "status": "failed"}), flush=True)
                raise  # fail closed; resume only after reviewing the technical failure


if __name__ == "__main__":
    main()
