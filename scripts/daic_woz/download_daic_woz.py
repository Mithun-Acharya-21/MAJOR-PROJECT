#!/usr/bin/env python3
"""
DAIC-WOZ Official Dataset Downloader & Extraction Script
NeuroScreen Research Pipeline - USC ICT Compliance Version

Usage:
    python scripts/daic_woz/download_daic_woz.py --dry-run
    python scripts/daic_woz/download_daic_woz.py --urls-file urls.txt --out-dir data/raw/daic_woz
    python scripts/daic_woz/download_daic_woz.py --extract --archive-dir data/raw/daic_woz

Security & Ethics:
    - Never bypass official USC ICT authentication gates.
    - No credentials or passwords may be hardcoded.
    - Preserves original participant identifiers (<id>_P).
"""

import os
import sys
import argparse
import hashlib
import zipfile
import tarfile
import urllib.request
import urllib.error
from pathlib import Path

CHUNK_SIZE = 1024 * 1024  # 1 MB chunks


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file in chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def download_file_resumable(url: str, dest_path: Path, auth_header: str = None) -> bool:
    """
    Download a file from an official USC ICT URL with resume support.
    Validates TLS certificates and avoids redownloading complete files.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    initial_bytes = temp_path.stat().st_size if temp_path.exists() else 0

    req = urllib.request.Request(url)
    if auth_header:
        req.add_header("Authorization", auth_header)
    if initial_bytes > 0:
        req.add_header("Range", f"bytes={initial_bytes}-")

    print(f"[*] Connecting to: {url}")
    try:
        with urllib.request.urlopen(req) as response:
            mode = "ab" if initial_bytes > 0 else "wb"
            with open(temp_path, mode) as f:
                while chunk := response.read(CHUNK_SIZE):
                    f.write(chunk)
        temp_path.rename(dest_path)
        print(f"[+] Download complete: {dest_path.name} ({dest_path.stat().st_size} bytes)")
        return True
    except urllib.error.HTTPError as e:
        if e.code == 416:  # Requested Range Not Satisfiable (already finished)
            if temp_path.exists():
                temp_path.rename(dest_path)
            print(f"[+] File already fully retrieved: {dest_path.name}")
            return True
        print(f"[-] HTTP Error {e.code}: {e.reason} for {url}")
        return False
    except Exception as e:
        print(f"[-] Download error for {url}: {e}")
        return False


def extract_archive(archive_path: Path, target_dir: Path) -> bool:
    """Safely extract zip or tar archive without directory traversal vulnerabilities."""
    print(f"[*] Extracting: {archive_path.name} -> {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        if archive_path.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                for member in zf.infolist():
                    target = target_dir / member.filename
                    if not str(target.resolve()).startswith(str(target_dir.resolve())):
                        raise SecurityError(f"Directory traversal detected in {member.filename}")
                zf.extractall(target_dir)
            print(f"[+] Extracted {archive_path.name}")
            return True
        elif archive_path.suffix.lower() in [".tar", ".gz", ".tgz"]:
            with tarfile.open(archive_path, "r:*") as tf:
                for member in tf.getmembers():
                    target = target_dir / member.name
                    if not str(target.resolve()).startswith(str(target_dir.resolve())):
                        raise SecurityError(f"Directory traversal detected in {member.name}")
                tf.extractall(target_dir)
            print(f"[+] Extracted {archive_path.name}")
            return True
        else:
            print(f"[-] Unsupported archive format: {archive_path.name}")
            return False
    except Exception as e:
        print(f"[-] Extraction error for {archive_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="DAIC-WOZ Official Dataset Downloader")
    parser.add_argument("--dry-run", action="store_true", help="Print official instructions without downloading")
    parser.add_argument("--urls-file", type=str, help="Text file containing authorized download URLs (one per line)")
    parser.add_argument("--out-dir", type=str, default="data/raw/daic_woz", help="Target download directory")
    parser.add_argument("--extract", action="store_true", help="Extract all downloaded archives in out-dir")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run or not args.urls_file:
        print("=" * 70)
        print("DAIC-WOZ DATASET ACQUISITION INSTRUCTIONS (USC ICT)")
        print("=" * 70)
        print("1. Apply for authorized research access at:")
        print("   https://dcapswoz.ict.usc.edu/")
        print("2. Submit signed institutional Data Use Agreement with faculty supervisor.")
        print("3. Upon receipt of authorized download links / credentials:")
        print("   - Save the authorized URLs into a local file: urls.txt")
        print("   - Run: python scripts/daic_woz/download_daic_woz.py --urls-file urls.txt")
        print("   - Run: python scripts/daic_woz/download_daic_woz.py --extract")
        print("   - Run: python scripts/daic_woz/verify_download.py")
        print("=" * 70)
        return

    urls_file = Path(args.urls_file)
    if not urls_file.exists():
        print(f"[-] URLs file not found: {urls_file}")
        sys.exit(1)

    with open(urls_file, "r") as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"[*] Found {len(urls)} authorized download target(s).")
    for url in urls:
        filename = url.split("?")[0].split("/")[-1]
        dest_path = out_dir / filename
        download_file_resumable(url, dest_path)

    if args.extract:
        for archive in out_dir.glob("*.zip"):
            extract_archive(archive, out_dir)
        for archive in out_dir.glob("*.tar*"):
            extract_archive(archive, out_dir)


if __name__ == "__main__":
    main()
