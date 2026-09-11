#!/usr/bin/env python3
"""Probe official SALAMI Internet Archive mappings and freeze an external subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import ssl
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SEED = "20260903"
USER_AGENT = "Mozilla/5.0 SALAMI-research-audit/1.0"


def selection_key(song_id: str) -> str:
    return hashlib.sha256((SEED + song_id).encode("utf-8")).hexdigest()


def canonical_url(url: str) -> str:
    if url.startswith("http://www.archive.org/"):
        return "https://archive.org/" + url[len("http://www.archive.org/") :]
    if url.startswith("http://archive.org/"):
        return "https://archive.org/" + url[len("http://archive.org/") :]
    return url


def probe(row: dict[str, str], timeout: float) -> dict[str, object]:
    url = canonical_url(row["URL"])
    result: dict[str, object] = {
        "song_id": row["SONG_ID"],
        "url": url,
        "reachable": False,
        "http_status": "",
        "content_type": "",
        "content_length": "",
        "final_url": "",
        "probe_method": "",
        "error": "",
    }
    headers = {"User-Agent": USER_AGENT}
    context = ssl.create_default_context()
    for method in ("HEAD", "GET"):
        method_headers = dict(headers)
        if method == "GET":
            method_headers["Range"] = "bytes=0-0"
        request = urllib.request.Request(url, headers=method_headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
                status = int(response.status)
                content_type = response.headers.get("Content-Type", "")
                result.update(
                    {
                        "reachable": status in (200, 206),
                        "http_status": status,
                        "content_type": content_type,
                        "content_length": response.headers.get("Content-Length", ""),
                        "final_url": response.geturl(),
                        "probe_method": method,
                    }
                )
                if result["reachable"]:
                    return result
        except urllib.error.HTTPError as error:
            result["http_status"] = error.code
            result["error"] = f"HTTPError: {error.reason}"
        except Exception as error:  # availability audit must preserve exact failure
            result["error"] = f"{type(error).__name__}: {error}"
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping-csv", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-tracks", type=int, default=50)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    with args.mapping_csv.open(newline="", encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle))

    futures = {}
    audit_by_id: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for row in source_rows:
            futures[executor.submit(probe, row, args.timeout)] = row["SONG_ID"]
        for index, future in enumerate(as_completed(futures), start=1):
            song_id = futures[future]
            audit_by_id[song_id] = future.result()
            print(
                f"[{index:03d}/{len(futures):03d}] {song_id} "
                f"reachable={audit_by_id[song_id]['reachable']}",
                flush=True,
            )

    audit_rows: list[dict[str, object]] = []
    eligible: list[dict[str, object]] = []
    for row in source_rows:
        song_id = row["SONG_ID"]
        upper_files = sorted((args.annotation_root / song_id / "parsed").glob("textfile*_uppercase.txt"))
        audit = audit_by_id[song_id]
        combined = {
            "song_id": song_id,
            "title": row["TITLE"],
            "artist": row["ARTIST"],
            "album": row["ALBUM"],
            "song_duration_seconds": row["SONG_DURATION"],
            "source_filename": row["FILE_NAME"],
            "n_annotators": len(upper_files),
            **audit,
        }
        audit_rows.append(combined)
        if bool(audit["reachable"]) and upper_files and float(row["SONG_DURATION"]) >= 60.0:
            eligible.append(combined)

    eligible.sort(key=lambda row: selection_key(str(row["song_id"])))
    selected = eligible[: args.n_tracks]
    # The protocol specifies "up to 50" because the 2012 Internet Archive
    # derivatives are availability-dependent. Preserve every reachable result
    # and do not replace missing public audio from an unofficial mirror.
    if len(selected) < 30:
        raise RuntimeError(f"Only {len(selected)} reachable annotated tracks; minimum is 30")
    selected_rows = []
    for rank, row in enumerate(selected, start=1):
        selected_rows.append(
            {
                "selection_rank": rank,
                "selection_key": selection_key(str(row["song_id"])),
                **row,
                "status": "frozen_selected",
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "salami_url_audit.csv"
    selected_path = args.output_dir / "salami_selected.csv"
    write_csv(audit_path, audit_rows)
    write_csv(selected_path, selected_rows)
    metadata = {
        "dataset": "SALAMI v2.0 Internet Archive subset",
        "seed": int(SEED),
        "mapping_rows": len(source_rows),
        "reachable_rows": sum(bool(row["reachable"]) for row in audit_rows),
        "eligible_rows": len(eligible),
        "selected_tracks": len(selected_rows),
        "selected_tracks_with_two_or_more_annotators": sum(
            int(row["n_annotators"]) >= 2 for row in selected_rows
        ),
        "selection_rule": "ascending SHA256('20260903' || SALAMI_ID) after reachability and annotation checks",
        "mapping_csv_sha256": sha256_file(args.mapping_csv),
        "audit_manifest_sha256": sha256_file(audit_path),
        "selected_manifest_sha256": sha256_file(selected_path),
    }
    (args.output_dir / "salami_manifest_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
