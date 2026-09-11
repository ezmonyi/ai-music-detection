#!/usr/bin/env python3
"""Static scientific heatmaps of every frozen Q pilot note, not AI/human scores."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root, out = Path(args.result_dir), Path(args.output_dir)
    audit = json.loads(Path(args.audit).read_text())
    commit_hash = sha(root / "COMMIT.json")
    committed = json.loads((root / "COMMIT.json").read_text())["products"]
    if audit.get("passed") is not True or audit.get("result_commit_sha256") != commit_hash:
        raise ValueError("requires numerical audit bound to this exact publication")
    if sha(root / "freeze.json") != committed["freeze.json"]["sha256"]:
        raise ValueError("frozen roster bytes changed")
    freeze = json.loads((root / "freeze.json").read_text())
    rows = freeze["selection"]["selected_notes"]
    if len(rows) != 54 or out.exists():
        raise ValueError("expected 54 fixed rows and a new output directory")
    conditions = ("baseline", "am8_depth06", "am48_depth06", "chirp16to64_depth06")
    titles = ("Baseline", "Added 8 Hz AM", "Added 48 Hz AM", "Added chirped AM")
    bands = ("250-1,000 Hz carrier", "1,000-3,000 Hz carrier", "3,000-7,000 Hz carrier")
    values = np.full((3, 4, 54, 252), np.nan)
    input_hashes = {}
    for j, row in enumerate(rows):
        for c, condition in enumerate(conditions):
            path = root / "candidate_json" / f"{row['id']}__{condition}.json"
            input_hashes[str(path.relative_to(root))] = sha(path)
            if input_hashes[str(path.relative_to(root))] != committed[str(path.relative_to(root))]["sha256"]:
                raise ValueError("measurement bytes differ from audited publication")
            data = json.loads(path.read_text())["measurement"]
            frequency = np.asarray(data["frequency_hz"])
            keep = (frequency >= 2) & (frequency < 128)
            for b in range(3):
                window = data["bands"][b]["windows"][0]
                if window["status"] == "ok":
                    power = np.asarray(window["modulation_power"])[keep]
                    values[b, c, j] = power / np.sum(power)
    out.mkdir(parents=True, exist_ok=False)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#b8bec5")
    fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharex=True, sharey=True, layout="constrained")
    for b in range(3):
        for c in range(4):
            display = 10 * np.log10(np.maximum(values[b, c], 1e-6))
            handle = axes[b, c].imshow(display, aspect="auto", interpolation="nearest",
                extent=(1.75, 127.75, 54.5, .5), cmap=cmap, vmin=-60, vmax=0)
            axes[b, c].set_xticks([8, 48, 80, 120])
            axes[b, c].set_yticks([1, 14, 27, 40, 54])
            if b == 0:
                axes[b, c].set_title(titles[c], fontsize=11)
            if c == 0:
                axes[b, c].set_ylabel(bands[b] + "\nFrozen note index")
            if b == 2:
                axes[b, c].set_xlabel("Envelope modulation frequency (Hz)")
    fig.colorbar(handle, ax=axes, label="Normalized analysis-bin power (dB); display floor -60 dB", shrink=.85)
    fig.suptitle("Q development pilot: all 54 notes in unchanged frozen order\n"
                 "One 2-second window per note; gray = ineligible; not AI/human classification", fontsize=14)
    fig.savefig(out / "modulation_control_heatmaps.png", dpi=180)
    fig.savefig(out / "modulation_control_heatmaps.svg")
    plt.close(fig)
    difference = values[:, 2] - values[:, 1]
    bound = float(np.nanmax(np.abs(difference)))
    cmap2 = plt.get_cmap("RdBu_r").copy()
    cmap2.set_bad("#b8bec5")
    fig, axes = plt.subplots(1, 3, figsize=(13, 6), sharey=True, layout="constrained")
    for b, axis in enumerate(axes):
        handle = axis.imshow(difference[b], aspect="auto", interpolation="nearest",
            extent=(1.75, 127.75, 54.5, .5), cmap=cmap2, vmin=-bound, vmax=bound)
        axis.set_title(bands[b])
        axis.set_xlabel("Envelope modulation frequency (Hz)")
        axis.set_xticks([8, 48, 80, 120])
        axis.set_yticks([1, 14, 27, 40, 54])
    axes[0].set_ylabel("Frozen note index")
    fig.colorbar(handle, ax=axes, label="48 Hz AM minus 8 Hz AM: normalized bin-power difference")
    fig.suptitle("Matched intervention difference, all notes retained\n"
                 "Color scale shared across bands; gray = either member ineligible", fontsize=13)
    fig.savefig(out / "modulation_matched_difference.png", dpi=180)
    fig.savefig(out / "modulation_matched_difference.svg")
    plt.close(fig)
    receipt = {"result_commit_sha256": commit_hash, "audit_sha256": sha(args.audit),
               "plot_code_sha256": sha(__file__), "notes": [{"index": j+1, "id": r["id"],
                    "instrument": r["metadata"]["instrument_str"]} for j, r in enumerate(rows)],
               "conditions": conditions, "input_hashes": input_hashes,
               "display_only_log_floor_db": -60, "difference_symmetric_bound": bound,
               "numeric_features_changed": False, "matplotlib": matplotlib.__version__}
    (out / "plot_receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    products = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in out.iterdir() if p.is_file()}
    (out / "COMMIT.json").write_text(json.dumps({"status":"committed", "products":products}, indent=2)+"\n")
    print(json.dumps({"output_dir": str(out), "products": len(products)}))


if __name__ == "__main__":
    main()
