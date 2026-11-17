"""Read the directory of a remote ZIP archive without downloading it.

Used to audit the 13.5 GiB AVDD-TCMI archive over HTTP range requests: the ZIP
central directory holds every member name, size and offset, so file counts,
formats and label files can be verified before a byte of research data is
stored locally. Standard library only (zlib) plus the `requests` session the
caller already has.

Only reading is supported; nothing is written and no full download happens.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

EOCD_SIG = b"PK\x05\x06"
EOCD64_LOCATOR_SIG = b"PK\x06\x07"
EOCD64_SIG = b"PK\x06\x06"
CENTRAL_SIG = b"PK\x01\x02"
LOCAL_SIG = b"PK\x03\x04"

TAIL_BYTES = 1 << 18  # ZIP64 EOCD + locator + central-directory tail always live in the last 256 KiB


@dataclass
class Entry:
    name: str
    compress_type: int
    crc: int
    compressed_size: int
    uncompressed_size: int
    header_offset: int

    @property
    def is_dir(self) -> bool:
        return self.name.endswith("/")


class RemoteZip:
    def __init__(self, url: str, session, timeout: int = 180):
        self.url = url
        self.session = session
        self.timeout = timeout
        r = session.head(url, allow_redirects=True, timeout=timeout)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])
        if "bytes" not in (r.headers.get("Accept-Ranges") or "").lower():
            raise RuntimeError("server does not support HTTP range requests")
        self._entries: list[Entry] | None = None

    def _get(self, start: int, end: int) -> bytes:
        if end < start:
            return b""
        r = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=self.timeout)
        if r.status_code not in (200, 206):
            raise RuntimeError(f"range request failed: HTTP {r.status_code}")
        return r.content

    def _central_directory(self) -> bytes:
        tail = self._get(max(0, self.size - TAIL_BYTES), self.size - 1)
        i = tail.rfind(EOCD_SIG)
        if i < 0:
            raise RuntimeError("end-of-central-directory record not found")
        cd_size, cd_offset = struct.unpack("<II", tail[i + 12:i + 20])

        # ZIP64 (the archive is > 4 GiB, so the 32-bit fields are saturated)
        loc = tail.rfind(EOCD64_LOCATOR_SIG)
        if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
            if loc < 0:
                raise RuntimeError("ZIP64 locator not found")
            eocd64_off, = struct.unpack("<Q", tail[loc + 8:loc + 16])
            rec = self._get(eocd64_off, eocd64_off + 55)
            if not rec.startswith(EOCD64_SIG):
                raise RuntimeError("bad ZIP64 end-of-central-directory record")
            cd_size, cd_offset = struct.unpack("<QQ", rec[40:56])
        return self._get(cd_offset, cd_offset + cd_size - 1)

    def entries(self) -> list[Entry]:
        if self._entries is not None:
            return self._entries
        cd = self._central_directory()
        out: list[Entry] = []
        k = 0
        while k + 46 <= len(cd) and cd[k:k + 4] == CENTRAL_SIG:
            comp_type, = struct.unpack("<H", cd[k + 10:k + 12])
            crc, = struct.unpack("<I", cd[k + 16:k + 20])
            comp_size, uncomp_size = struct.unpack("<II", cd[k + 20:k + 28])
            fn_len, extra_len, comment_len = struct.unpack("<HHH", cd[k + 28:k + 34])
            offset, = struct.unpack("<I", cd[k + 42:k + 46])
            name = cd[k + 46:k + 46 + fn_len].decode("utf-8", "replace")
            extra = cd[k + 46 + fn_len:k + 46 + fn_len + extra_len]

            if 0xFFFFFFFF in (comp_size, uncomp_size, offset):
                comp_size, uncomp_size, offset = _apply_zip64_extra(
                    extra, comp_size, uncomp_size, offset)

            out.append(Entry(name, comp_type, crc, comp_size, uncomp_size, offset))
            k += 46 + fn_len + extra_len + comment_len
        self._entries = out
        return out

    def read_prefix(self, entry: Entry, max_bytes: int = 262144) -> bytes:
        """Decompressed prefix of one member (enough for a header line or a small CSV)."""
        head = self._get(entry.header_offset, entry.header_offset + 29)
        if not head.startswith(LOCAL_SIG):
            raise RuntimeError(f"bad local header for {entry.name}")
        fn_len, extra_len = struct.unpack("<HH", head[26:30])
        start = entry.header_offset + 30 + fn_len + extra_len
        take = min(entry.compressed_size, max_bytes)
        blob = self._get(start, start + take - 1)
        if entry.compress_type == 0:
            return blob[:max_bytes]
        if entry.compress_type == 8:
            d = zlib.decompressobj(-15)
            return d.decompress(blob, max_bytes)
        raise RuntimeError(f"unsupported compression type {entry.compress_type} for {entry.name}")


def _apply_zip64_extra(extra: bytes, comp_size: int, uncomp_size: int, offset: int):
    k = 0
    while k + 4 <= len(extra):
        tag, size = struct.unpack("<HH", extra[k:k + 4])
        body = extra[k + 4:k + 4 + size]
        if tag == 0x0001:
            values = [struct.unpack("<Q", body[i:i + 8])[0] for i in range(0, len(body) - 7, 8)]
            it = iter(values)
            if uncomp_size == 0xFFFFFFFF:
                uncomp_size = next(it)
            if comp_size == 0xFFFFFFFF:
                comp_size = next(it)
            if offset == 0xFFFFFFFF:
                offset = next(it)
            return comp_size, uncomp_size, offset
        k += 4 + size
    raise RuntimeError("ZIP64 extra field missing while offsets are saturated")
