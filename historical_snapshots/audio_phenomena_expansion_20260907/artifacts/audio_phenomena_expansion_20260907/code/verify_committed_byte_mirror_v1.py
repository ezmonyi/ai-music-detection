"""Verify a complete byte mirror against an explicit trusted COMMIT hash."""
import argparse
import hashlib
import json
from pathlib import Path
import socket


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--commit-sha256", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    root = Path(args.root).resolve(strict=True)
    output = Path(args.output).resolve()
    if output.exists() or output.is_relative_to(root):
        raise ValueError("new receipt outside the mirror required")
    commit = root / "COMMIT.json"
    if sha(commit) != args.commit_sha256:
        raise ValueError("untrusted commit bytes")
    manifest = json.loads(commit.read_text())
    products = manifest["products"]
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("symlink in mirror")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual != set(products) | {"COMMIT.json"}:
        raise ValueError("mirror inventory mismatch")
    total = 0
    for name, record in products.items():
        path = root / name
        if not path.resolve(strict=True).is_relative_to(root):
            raise ValueError("path escape")
        if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError("mirror byte mismatch: " + name)
        total += record["bytes"]
    if sha(commit) != args.commit_sha256:
        raise ValueError("commit changed during verification")
    result = {"passed": True, "scope": "complete_byte_and_inventory_mirror_not_scientific_reanalysis",
        "host": socket.gethostname(), "mirror_root": str(root), "products": len(products),
        "product_bytes": total, "commit_sha256": args.commit_sha256,
        "code_sha256": sha(Path(__file__)), "embedded_provenance_paths_rewritten": False}
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
