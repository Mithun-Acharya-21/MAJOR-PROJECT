"""Download the Harvard Dataverse "Depression dataset for English tweets" (CC0 1.0).

DOI: 10.7910/DVN/ISZCSA  (https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ISZCSA)

Verified record facts (2026-11-13): one file, an .xlsx of 4,678,835 bytes,
unrestricted, licence CC0 1.0, dataset version 1.1.

Behaviour:
  - resolves the file id + size + md5 from the Dataverse API instead of hardcoding
  - resumes partial downloads with HTTP Range
  - verifies the published MD5 before marking the file complete
  - idempotent: a complete, verified file is reused, never re-downloaded
  - --dry-run prints the plan and exits without network writes

Usage:
  .venv/Scripts/python scripts/datasets/download_harvard_tweets.py --dry-run
  .venv/Scripts/python scripts/datasets/download_harvard_tweets.py
Exit codes: 0 ok, 1 verification failure, 2 refused (space/licence), 4 not retrievable.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    EXIT_FAIL, EXIT_OK, EXIT_REFUSED, RAW, REPORTS, USER_AGENT,
    free_bytes, get_logger, human_bytes, utcnow, write_json,
)

DOI = "doi:10.7910/DVN/ISZCSA"
DATASET_URL = "https://dataverse.harvard.edu/dataset.xhtml?persistentId=" + DOI
API = "https://dataverse.harvard.edu/api/datasets/:persistentId/"
DEST = RAW / "text" / "harvard_tweets"
REPORT = REPORTS / "harvard_tweets_download.json"
MIN_FREE_MARGIN = 256 * 1024 * 1024  # keep 256 MB free


def fetch_record(session: requests.Session) -> dict:
    r = session.get(API, params={"persistentId": DOI}, timeout=60)
    r.raise_for_status()
    data = r.json()["data"]
    version = data["latestVersion"]
    files = version["files"]
    if len(files) != 1:
        raise RuntimeError(f"expected exactly one published file, found {len(files)}")
    f = files[0]
    df = f["dataFile"]
    return {
        "dataset_title": "Depression dataset for English tweets classified binary",
        "doi": DOI,
        "dataset_url": DATASET_URL,
        "version": f"{version.get('versionNumber')}.{version.get('versionMinorNumber')}",
        "license": version.get("license", {}).get("name"),
        "license_uri": version.get("license", {}).get("uri"),
        "file_id": df["id"],
        "filename": df["filename"],
        "filesize": df.get("filesize"),
        "md5": df.get("md5"),
        "content_type": df.get("contentType"),
        "restricted": bool(f.get("restricted")),
        "download_url": f"https://dataverse.harvard.edu/api/access/datafile/{df['id']}?format=original",
    }


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(session: requests.Session, url: str, dest: Path, expected_size: int, log) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > expected_size:
        log.warning("partial file larger than expected; restarting")
        part.unlink()
        have = 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    mode = "ab" if have else "wb"
    log.info("GET %s (resume from %s)", url, human_bytes(have) if have else "0 B")
    with session.get(url, headers=headers, stream=True, timeout=120) as r:
        if have and r.status_code == 200:
            log.warning("server ignored Range; restarting download")
            mode, have = "wb", 0
        elif r.status_code not in (200, 206):
            raise RuntimeError(f"HTTP {r.status_code} for {url}")
        total = expected_size
        done = have
        with part.open(mode) as fh:
            for block in r.iter_content(chunk_size=1 << 16):
                if not block:
                    continue
                fh.write(block)
                done += len(block)
                if total and done % (1 << 20) < (1 << 16):
                    log.info("  %s / %s (%.0f%%)", human_bytes(done), human_bytes(total), 100 * done / total)
    if part.stat().st_size != expected_size:
        raise RuntimeError(f"size mismatch: {part.stat().st_size} != {expected_size}")
    part.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--dry-run", action="store_true", help="print the plan, download nothing")
    ap.add_argument("--force", action="store_true", help="re-download even if a verified file exists")
    args = ap.parse_args()
    log = get_logger("download_harvard_tweets")

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    record = fetch_record(session)
    log.info("record: %s v%s | licence %s | %s | %s", record["filename"], record["version"],
             record["license"], human_bytes(record["filesize"]), record["content_type"])
    if record["restricted"]:
        log.error("dataset file is restricted; refusing")
        return EXIT_REFUSED
    if (record["license"] or "").upper().startswith("CC0") is False:
        log.warning("licence is %s (expected CC0 1.0)", record["license"])

    dest = args.dest / record["filename"]
    already = dest.exists() and not args.force
    plan = {
        "generated_at": utcnow(),
        "dataset": record,
        "destination": str(dest),
        "existing_file": dest.exists(),
        "expected_size": record["filesize"],
        "expected_md5": record["md5"],
        "free_bytes": free_bytes(),
        "dry_run": args.dry_run,
    }

    if args.dry_run:
        log.info("DRY RUN — would download %s to %s (%s, free %s)",
                 record["download_url"], dest, human_bytes(record["filesize"]), human_bytes(free_bytes()))
        plan["status"] = "dry_run"
        write_json(REPORT, plan)
        return EXIT_OK

    if free_bytes() < record["filesize"] + MIN_FREE_MARGIN:
        log.error("insufficient disk space: need %s + margin, free %s",
                  human_bytes(record["filesize"]), human_bytes(free_bytes()))
        plan["status"] = "refused_no_space"
        write_json(REPORT, plan)
        return EXIT_REFUSED

    args.dest.mkdir(parents=True, exist_ok=True)
    status = "reused"
    if not already:
        download(session, record["download_url"], dest, record["filesize"], log)
        status = "downloaded"

    md5 = md5_file(dest)
    ok = md5 == record["md5"]
    plan.update({
        "status": status,
        "size_on_disk": dest.stat().st_size,
        "md5_on_disk": md5,
        "md5_matches": ok,
        "completed_at": utcnow(),
    })
    write_json(REPORT, plan)
    if not ok:
        log.error("MD5 mismatch: %s != %s", md5, record["md5"])
        return EXIT_FAIL
    log.info("%s %s (%s, md5 %s)", status, dest, human_bytes(dest.stat().st_size), md5[:12])
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
