"""Convert raw ILSI jsonl into parquet files + a fixed label list.

Outputs (in --out_dir):
  train/dev/test.parquet  columns: id (str), text (str), labels (list[str], e.g. ["302", "34"])
  statutes.parquet        columns: section_id (str), text (str)
  label_list.json         the fixed label order used everywhere (natural numeric sort)

Usage: python src/prepare_data.py --raw_dir data/raw --out_dir data/processed
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

SPLITS = ["train", "dev", "test"]
LABEL_RE = re.compile(r"^Section (\S+) in The Indian Penal Code$")


def short_label(raw_label):
    """'Section 120B in The Indian Penal Code' -> '120B'."""
    m = LABEL_RE.match(raw_label)
    if not m:
        raise ValueError(f"Unexpected label format: {raw_label!r}")
    return m.group(1)


def section_sort_key(sec):
    """Natural order: 34 < 107 < 120 < 120B < 302."""
    m = re.match(r"^(\d+)([A-Za-z]*)$", sec)
    return (int(m.group(1)), m.group(2)) if m else (10**9, sec)


def join_text(text):
    return " ".join(s.strip() for s in text) if isinstance(text, list) else str(text)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--out_dir", default="data/processed")
    args = ap.parse_args()
    raw, out = Path(args.raw_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_labels, ids_by_split = set(), {}
    for split in SPLITS:
        recs = read_jsonl(raw / f"{split}.jsonl")
        df = pd.DataFrame({
            "id": [str(r["id"]) for r in recs],
            "text": [join_text(r["text"]) for r in recs],
            "labels": [sorted({short_label(l) for l in r["labels"]}, key=section_sort_key)
                       for r in recs],
        })
        del recs
        assert df["id"].is_unique, f"duplicate ids in {split}"
        df.to_parquet(out / f"{split}.parquet", index=False)
        all_labels.update(l for ls in df["labels"] for l in ls)
        ids_by_split[split] = set(df["id"])
        print(f"{split}: {len(df)} docs -> {out / f'{split}.parquet'}")
        del df

    for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        overlap = len(ids_by_split[a] & ids_by_split[b])
        print(f"id overlap {a}/{b}: {overlap}")

    secs = read_jsonl(raw / "secs.jsonl")
    statutes = pd.DataFrame({
        "section_id": [short_label(s["id"]) for s in secs],
        "text": [join_text(s["text"]) for s in secs],
    })
    statutes = statutes.sort_values("section_id", key=lambda c: c.map(section_sort_key))
    statutes.to_parquet(out / "statutes.parquet", index=False)
    print(f"statutes: {len(statutes)} -> {out / 'statutes.parquet'}")

    label_list = sorted(all_labels, key=section_sort_key)
    missing = set(label_list) - set(statutes["section_id"])
    assert not missing, f"labels without statute text: {missing}"
    with open(out / "label_list.json", "w") as f:
        json.dump(label_list, f, indent=1)
    print(f"label_list.json: {len(label_list)} labels; first 10: {label_list[:10]}")


if __name__ == "__main__":
    main()
