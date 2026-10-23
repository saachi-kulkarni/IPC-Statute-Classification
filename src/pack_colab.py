"""Bundle everything the Colab notebook needs into colab_upload/ and colab_upload.zip.

Usage: python src/pack_colab.py
"""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from common import PROC, ROOT

FILES = ["train.parquet", "dev.parquet", "test.parquet", "statutes.parquet",
         "label_list.json", "label_buckets.json"]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(ROOT / "colab_upload"))
    ap.add_argument("--zip_path", default=str(ROOT / "colab_upload.zip"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        shutil.copy2(PROC / f, out / f)
    shutil.copy2(ROOT / "src" / "metrics.py", out / "metrics.py")
    manifest = {f.name: md5(f) for f in sorted(out.iterdir()) if f.name != "MANIFEST.json"}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))

    with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):
            z.write(f, arcname=f"colab_upload/{f.name}")
    size = Path(args.zip_path).stat().st_size / 1e6
    print(f"wrote {args.zip_path} ({size:.1f} MB) with: {sorted(manifest)}")


if __name__ == "__main__":
    main()
