"""Bounded, authenticated reads of individual members in uncompressed Drive TARs.

Set MUSIC_DETECTOR_DRIVE_TOKEN to a read-capable OAuth access token, or supply
a token callback to DriveRangeReader. Tokens and signed URLs are never logged.
The caller must obtain current file metadata and pass the observed archive size.
No archive is extracted wholesale; only regular-file members are supported.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
from typing import Callable, Iterable
import urllib.error
import urllib.parse
import urllib.request


class RangeReadError(RuntimeError):
    """A remote response violated the bounded-read contract."""


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise RangeReadError("Refused a non-HTTPS redirect")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if urllib.parse.urlsplit(req.full_url).netloc != urllib.parse.urlsplit(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


def _safe_member_name(name: str) -> str:
    path = PurePosixPath(name)
    if not name or "\x00" in name or path.is_absolute() or ".." in path.parts:
        raise RangeReadError("Unsafe TAR member name")
    return name


@dataclass(frozen=True)
class TarMember:
    name: str
    offset: int
    size: int


class DriveRangeReader:
    def __init__(self, file_id: str, size: int, *, token: Callable[[], str] | None = None,
                 max_bytes: int = 64 * 1024 * 1024, timeout: float = 40,
                 opener=None):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", file_id) or size <= 0 or max_bytes <= 0:
            raise ValueError("Invalid Drive ID, archive size, or byte budget")
        self.file_id, self.size = file_id, size
        self.max_bytes, self.bytes_read, self.timeout = max_bytes, 0, timeout
        self.token = token or (lambda: os.environ["MUSIC_DETECTOR_DRIVE_TOKEN"])
        self.opener = opener or urllib.request.build_opener(_SafeRedirect())

    def read_at(self, offset: int, length: int) -> bytes:
        if offset < 0 or length < 0 or offset + length > self.size:
            raise RangeReadError("Read exceeds archive bounds")
        if not length:
            return b""
        if length > 8 * 1024 * 1024:
            raise RangeReadError("One request may not exceed 8 MiB")
        if self.bytes_read + length > self.max_bytes:
            raise RangeReadError("Download byte budget exhausted")
        end = offset + length - 1
        request = urllib.request.Request(
            f"https://www.googleapis.com/drive/v3/files/{self.file_id}?alt=media",
            headers={"Authorization": "Bearer " + self.token(),
                     "Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity"})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                if response.status != 206:
                    raise RangeReadError("Server ignored Range; full download refused")
                expected = f"bytes {offset}-{end}/{self.size}"
                if response.headers.get("Content-Range") != expected:
                    raise RangeReadError("Unexpected Content-Range or changed archive size")
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise RangeReadError("Encoded byte-range response refused")
                data = response.read(length + 1)
                self.bytes_read += len(data)
                if len(data) != length:
                    raise RangeReadError("Truncated or oversized byte-range response")
                return data
        except urllib.error.HTTPError as exc:
            raise RangeReadError(f"Drive HTTP {exc.code}") from None
        except urllib.error.URLError:
            raise RangeReadError("Drive connection failed") from None


def _pax_records(payload: bytes) -> dict[str, str]:
    records, pos = {}, 0
    while pos < len(payload):
        space = payload.find(b" ", pos)
        if space == -1:
            raise RangeReadError("Malformed PAX record")
        try:
            length = int(payload[pos:space])
        except ValueError:
            raise RangeReadError("Malformed PAX length") from None
        if length <= space - pos + 1 or pos + length > len(payload):
            raise RangeReadError("PAX record exceeds header payload")
        record = payload[space + 1:pos + length]
        if not record.endswith(b"\n") or b"=" not in record:
            raise RangeReadError("Malformed PAX key/value")
        key, value = record[:-1].split(b"=", 1)
        records[key.decode("utf-8")] = value.decode("utf-8")
        pos += length
    return records


def find_tar_members(reader: DriveRangeReader, names: Iterable[str], *,
                     max_headers: int = 10000) -> dict[str, TarMember]:
    """Skip member payloads by their lengths; read only TAR metadata headers.

    PAX and GNU long-name records are supported. A byte budget and header limit
    bound discovery even for an untrusted or very large archive.
    """
    pending = {_safe_member_name(name) for name in names}
    found: dict[str, TarMember] = {}
    offset, local_pax, global_pax, long_name = 0, {}, {}, None
    for _ in range(max_headers):
        if not pending:
            return found
        if offset + 512 > reader.size:
            break
        raw = reader.read_at(offset, 512)
        if raw == b"\0" * 512:
            break
        try:
            info = tarfile.TarInfo.frombuf(raw, "utf-8", "surrogateescape")
        except tarfile.HeaderError:
            raise RangeReadError("Invalid TAR header; compressed archives unsupported") from None
        data_offset = offset + 512
        size = info.size
        if info.type in (tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME):
            if size > 1024 * 1024 or size < 0:
                raise RangeReadError("Oversized TAR extension header")
            payload = reader.read_at(data_offset, size)
            if info.type == tarfile.GNUTYPE_LONGNAME:
                long_name = payload.rstrip(b"\0").decode("utf-8")
            elif info.type == tarfile.XGLTYPE:
                global_pax.update(_pax_records(payload))
            else:
                local_pax.update(_pax_records(payload))
        else:
            pax = {**global_pax, **local_pax}
            name = pax.get("path", long_name or info.name)
            size = int(pax.get("size", size))
            if name in pending:
                _safe_member_name(name)
                if not info.isreg() or any(key.startswith("GNU.sparse") for key in pax):
                    raise RangeReadError("Selected TAR member is not a regular non-sparse file")
                found[name] = TarMember(name, data_offset, size)
                pending.remove(name)
            local_pax, long_name = {}, None
        if size < 0 or data_offset + size > reader.size:
            raise RangeReadError("TAR member exceeds archive bounds")
        offset = data_offset + (size + 511) // 512 * 512
    if pending:
        raise RangeReadError(f"Members not found within scan limits ({len(pending)} missing)")
    return found


def download_member(reader: DriveRangeReader, member: TarMember, destination: Path,
                    expected_sha256: str) -> dict:
    """Read a known member offset and commit a new file only after SHA-256 passes."""
    _safe_member_name(member.name)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("A canonical member SHA-256 is required")
    if member.size < 0 or member.offset < 0 or member.offset + member.size > reader.size:
        raise RangeReadError("Invalid member bounds")
    if reader.bytes_read + member.size > reader.max_bytes:
        raise RangeReadError("Member exceeds remaining download budget")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    # Exclusive creation prevents clobbering an existing user file.
    with destination.open("xb") as stream:
        try:
            for relative in range(0, member.size, 4 * 1024 * 1024):
                block = reader.read_at(member.offset + relative, min(4 * 1024 * 1024, member.size - relative))
                digest.update(block)
                stream.write(block)
            if digest.hexdigest() != expected_sha256:
                raise RangeReadError("Member SHA-256 mismatch")
        except BaseException:
            stream.close()
            destination.unlink()
            raise
    return {"drive_file_id": reader.file_id, "archive_size": reader.size,
            "member": asdict(member), "sha256": digest.hexdigest(),
            "downloaded_bytes": reader.bytes_read, "destination": str(destination)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file_id")
    parser.add_argument("archive_size", type=int)
    parser.add_argument("member")
    parser.add_argument("--offset", type=int, help="Verified TAR payload offset from an index")
    parser.add_argument("--member-size", type=int)
    parser.add_argument("--sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-bytes", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args(argv)
    reader = DriveRangeReader(args.file_id, args.archive_size, max_bytes=args.max_bytes)
    if args.offset is None:
        member = find_tar_members(reader, [args.member])[args.member]
    else:
        if args.member_size is None:
            parser.error("--offset requires --member-size")
        member = TarMember(args.member, args.offset, args.member_size)
    if args.output:
        if not args.sha256:
            parser.error("--output requires --sha256")
        result = download_member(reader, member, args.output, args.sha256)
    else:
        result = {**asdict(member), "downloaded_bytes": reader.bytes_read}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
