"""Fetch pinned Native60 report inputs, streaming Drive CSV into historical gzip.

Credentials are supplied only through MUSIC_DETECTOR_DRIVE_TOKEN. No audio or
models are fetched. The report program remains unchanged and checks original
input pins before replay. The CSV is never stored uncompressed on disk.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import urllib.parse
import urllib.request
import zlib

from music_detector.drive import DriveRangeReader

CSV = {
    "id": "1rIkYezgWs5jvrtnyIaFnnooBmmQkyPvm",
    "bytes": 126790798,
    "sha256": "afc31519bdd15da38df142a3de4b18a7fae732250910adcb854e6bff221d3bd8",
}
GZIP = {
    "bytes": 13431894,
    "sha256": "ef8243b8aba510a5d9c44999a1820fbe7be0ad38caa65fbf855bcfddef6bd161",
}
COMPANIONS = [
    ("1JbwT2Kulgr3bbAswwUVE-sd1ZZmPkKMh", "native60_transfer_scores_v1/COMMIT.json", 834,
     "34782ccfa467d5304ec2a4af1000d390eb467f004bb591ddce1474d06a068e64"),
    ("1mxQy0HOk83dQD-7GMCeJLaPEIBEzOgVg", "native60_transfer_scores_v1/per_model_summary.json", 7010049,
     "19f331f15796520d7ed4fc49d83ce9b1ba47972f98dccd17e1a813b7e9e8f3b6"),
    ("1kZaaZGC2iusBCCmC5w8M_ttlsy3zRIIK", "native60_feature_package_v1/metadata.json", 45266,
     "d1e219e9a91370b7e470bac5c0deaaa70759b15ed49316f08b3ef5a4055771f1"),
]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(destination: Path, *, opener=None):
    if destination.exists():
        raise FileExistsError("Input directory must be new; no files are overwritten")
    token = lambda: os.environ["MUSIC_DETECTOR_DRIVE_TOKEN"]

    def reader_for(file_id, expected_size, expected_sha):
        reader = DriveRangeReader(file_id, expected_size, token=token,
                                  max_bytes=expected_size, timeout=60, opener=opener)
        fields = "id,name,mimeType,size,md5Checksum,sha256Checksum,trashed,parents"
        request = urllib.request.Request(
            "https://www.googleapis.com/drive/v3/files/" + file_id + "?" +
            urllib.parse.urlencode({"fields": fields}),
            headers={"Authorization": "Bearer " + token()})
        with reader.opener.open(request, timeout=40) as response:
            metadata = json.load(response)
        assert not metadata["trashed"] and int(metadata["size"]) == expected_size
        if metadata.get("sha256Checksum"):
            assert metadata["sha256Checksum"] == expected_sha
        return reader, metadata

    reader, metadata = reader_for(CSV["id"], CSV["bytes"], CSV["sha256"])
    destination.mkdir(parents=True, exist_ok=False)
    output = destination / "native60_transfer_scores_v1/predictions.csv.gz"
    output.parent.mkdir()
    csv_hash = hashlib.sha256()
    with output.open("xb") as binary:
        # Filename and mtime reproduce score_native60_transfer_v1.py's header.
        with gzip.GzipFile(filename="predictions.csv", fileobj=binary,
                           mode="wb", mtime=0, compresslevel=9) as compressed:
            for offset in range(0, CSV["bytes"], 4 * 1024 * 1024):
                data = reader.read_at(offset, min(4 * 1024 * 1024, CSV["bytes"] - offset))
                csv_hash.update(data)
                compressed.write(data)
            # Historical io.TextIOWrapper closes with a sync flush before gzip.
            compressed.flush()
    assert csv_hash.hexdigest() == CSV["sha256"]
    gz_hash = digest(output)
    decoded_hash, decoded_bytes = hashlib.sha256(), 0
    with gzip.open(output, "rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            decoded_hash.update(block)
            decoded_bytes += len(block)
    assert decoded_hash.hexdigest() == CSV["sha256"] and decoded_bytes == CSV["bytes"]
    receipt = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": metadata,
        "csv_downloaded_bytes": reader.bytes_read,
        "csv_sha256": csv_hash.hexdigest(),
        "csv_saved_uncompressed": False,
        "gzip": {"bytes": output.stat().st_size, "sha256": gz_hash},
        "gzip_decoded": {"bytes": decoded_bytes, "sha256": decoded_hash.hexdigest()},
        "historical_gzip": GZIP,
        "gzip_exact_match": gz_hash == GZIP["sha256"] and output.stat().st_size == GZIP["bytes"],
        "compression": {"gzip_mtime": 0, "gzip_filename": "predictions.csv", "level": 9,
                        "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION},
        "companions": [],
    }
    for file_id, relative, size, pin in COMPANIONS:
        source, item_metadata = reader_for(file_id, size, pin)
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            for offset in range(0, size, 4 * 1024 * 1024):
                stream.write(source.read_at(offset, min(4 * 1024 * 1024, size - offset)))
        assert digest(path) == pin
        receipt["companions"].append({"metadata": item_metadata, "relative_path": relative,
                                      "bytes": size, "sha256": pin})
    with (destination / "DRIVE_DOWNLOAD_ACCEPTANCE.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2))
    if not receipt["gzip_exact_match"]:
        raise ValueError("CSV verified, but gzip does not match original pinned bytes; report not run")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    main(args.destination)
