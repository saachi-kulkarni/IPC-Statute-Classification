"""Inspect the raw ILSI jsonl files: field names, split sizes, label stats, lengths.

Usage: python src/inspect_data.py --raw_dir data/raw
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

SPLITS = ["train", "dev", "test"]
# Explicit numeric section mentions left in the text (potential label leakage).
SECTION_MENTION = re.compile(r"\bsec(?:tion|s?\.)?s?\s*\d+[a-z]?\b", re.IGNORECASE)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def section_number(label):
    """'Section 120B in The Indian Penal Code' -> '120B'."""
    return label.split()[1]


def gold_leak(text, labels):
    """How many gold sections are named as 'section <N>' in the document text."""
    found = {m.group(1).upper() for m in
             re.finditer(r"sec(?:tion|s?\.)?s?\s*(\d+[a-z]?)\b", text, re.IGNORECASE)}
    return sum(section_number(l).upper() in found for l in labels)


def describe(values, name):
    a = np.asarray(values)
    print(f"  {name}: mean={a.mean():.1f} median={np.median(a):.0f} "
          f"p90={np.percentile(a, 90):.0f} p95={np.percentile(a, 95):.0f} "
          f"min={a.min()} max={a.max()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    args = ap.parse_args()
    raw = Path(args.raw_dir)

    label_sets = {}
    for split in SPLITS:
        keys = Counter()
        n_docs, n_labels, words, sents = 0, [], [], []
        labels = Counter()
        text_types = Counter()
        n_mention, n_gold_named, n_gold = 0, 0, 0
        for rec in read_jsonl(raw / f"{split}.jsonl"):
            keys.update(rec.keys())
            text_types[type(rec["text"]).__name__] += 1
            text = " ".join(rec["text"]) if isinstance(rec["text"], list) else rec["text"]
            n_docs += 1
            n_labels.append(len(rec["labels"]))
            labels.update(rec["labels"])
            words.append(len(text.split()))
            sents.append(len(rec["text"]) if isinstance(rec["text"], list) else 1)
            if SECTION_MENTION.search(text):
                n_mention += 1
            n_gold_named += gold_leak(text, rec["labels"])
            n_gold += len(rec["labels"])
        label_sets[split] = labels
        print(f"\n=== {split} ===")
        print(f"  record keys (count of records containing key): {dict(keys)}")
        print(f"  'text' field type: {dict(text_types)}")
        print(f"  documents: {n_docs}")
        print(f"  unique labels: {len(labels)}")
        describe(n_labels, "labels/doc")
        describe(words, "words/doc")
        describe(sents, "sentences/doc")
        print(f"  docs with an explicit 'section <number>' mention in text: "
              f"{n_mention} ({100 * n_mention / n_docs:.1f}%)")
        print(f"  gold labels whose number appears as 'section <N>' in the text: "
              f"{n_gold_named}/{n_gold} ({100 * n_gold_named / n_gold:.1f}%)  [leakage check]")

    secs = list(read_jsonl(raw / "secs.jsonl"))
    print("\n=== secs.jsonl ===")
    print(f"  record keys: {sorted(secs[0].keys())}; records: {len(secs)}")
    sec_ids = {s["id"] for s in secs}
    describe([len(" ".join(s["text"]).split()) for s in secs], "words/statute")

    train, dev, test = (label_sets[s] for s in SPLITS)
    all_labels = set(train) | set(dev) | set(test)
    print("\n=== label coverage ===")
    print(f"  union of labels across splits: {len(all_labels)}")
    print(f"  labels in dev but not train: {sorted(set(dev) - set(train))}")
    print(f"  labels in test but not train: {sorted(set(test) - set(train))}")
    print(f"  labels without a statute text in secs.jsonl: {sorted(all_labels - sec_ids)}")
    print(f"  statutes in secs.jsonl that are never a label: {len(sec_ids - all_labels)}")
    print(f"  example label strings: {sorted(all_labels)[:3]}")
    print("  top 10 train labels:")
    for lab, c in train.most_common(10):
        print(f"    {c:6d}  {lab}")


if __name__ == "__main__":
    main()
