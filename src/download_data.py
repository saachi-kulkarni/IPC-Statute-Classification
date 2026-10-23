"""Download the ILSI splits + statute texts from Zenodo record 6053791 and verify MD5s.

Only train/dev/test.jsonl and secs.jsonl are fetched (~510 MB); the 2.3 GB graph
embeddings and citation files are skipped.

Usage: python src/download_data.py [--out_dir data/raw]
"""
import argparse
import hashlib
from pathlib import Path

import requests

RECORD = "https://zenodo.org/api/records/6053791"
WANTED = ["train.jsonl", "dev.jsonl", "test.jsonl", "secs.jsonl"]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/raw")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {f["key"]: f for f in requests.get(RECORD, timeout=60).json()["files"]}
    for name in WANTED:
        meta = files[name]
        expected = meta["checksum"].split(":", 1)[1]
        dest = out / name
        if dest.exists() and md5(dest) == expected:
            print(f"{name}: present, checksum OK")
            continue
        print(f"{name}: downloading {meta['size'] / 1e6:.1f} MB ...", flush=True)
        with requests.get(meta["links"]["self"], stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        got = md5(dest)
        if got != expected:
            raise SystemExit(f"{name}: MD5 mismatch ({got} != {expected})")
        print(f"{name}: checksum OK")


if __name__ == "__main__":
    main()
