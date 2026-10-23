"""Reusable EDA helpers used by notebooks/01_eda.ipynb."""
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BERT_WORD_LIMIT = 400  # ~512 word-piece tokens for English legal text (rough rule of thumb)


def load_split(proc_dir, split, with_text=True):
    cols = ["id", "text", "labels"] if with_text else ["id", "labels"]
    df = pd.read_parquet(Path(proc_dir) / f"{split}.parquet", columns=cols)
    df["labels"] = df["labels"].map(list)
    return df


def load_label_list(proc_dir):
    with open(Path(proc_dir) / "label_list.json") as f:
        return json.load(f)


def label_counts(df, label_list):
    c = Counter(l for ls in df["labels"] for l in ls)
    return pd.Series({l: c.get(l, 0) for l in label_list}, name="count")


def plot_label_frequency(counts, out_path, top_k=15):
    s = counts.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#1f77b4" if i < top_k else "#9ecae1" for i in range(len(s))]
    ax.bar(range(len(s)), s.values, color=colors)
    ax.set_yscale("log")
    ax.set_xticks(range(top_k))
    ax.set_xticklabels([f"§{l}" for l in s.index[:top_k]], rotation=60, fontsize=8)
    ax.set_xlabel(f"IPC section (ranked by train frequency; top {top_k} named)")
    ax.set_ylabel("# train documents (log)")
    ax.set_title("Label frequency (train)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return fig


def plot_labels_per_doc(dfs, out_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(1, 22) - 0.5
    for name, df in dfs.items():
        n = df["labels"].map(len)
        ax.hist(n.clip(upper=20), bins=bins, alpha=0.5, density=True,
                label=f"{name} (mean {n.mean():.2f})")
    ax.set_xlabel("# labels per document (20 = 20+)")
    ax.set_ylabel("fraction of documents")
    ax.set_title("Labels per document")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return fig


def plot_doc_lengths(dfs, out_path, limit=BERT_WORD_LIMIT):
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.logspace(0, np.log10(200_000), 60)
    for name, df in dfs.items():
        n = df["text"].str.split().str.len()
        ax.hist(n, bins=bins, alpha=0.5, label=name, weights=np.full(len(n), 1 / len(n)))
    ax.axvline(limit, color="red", ls="--", label=f"~{limit} words ≈ 512 BERT tokens")
    ax.set_xscale("log")
    ax.set_xlabel("document length (words, log)")
    ax.set_ylabel("fraction of documents per bin")
    ax.set_title("Document lengths")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return fig


def cooccurrence_pairs(df, top_k=20):
    c = Counter()
    for ls in df["labels"]:
        c.update(combinations(sorted(set(ls)), 2))
    single = Counter(l for ls in df["labels"] for l in ls)
    rows = []
    for (a, b), n in c.most_common(top_k):
        rows.append({"a": a, "b": b, "pair": f"§{a} & §{b}", "count": n,
                     "P(b|a)": round(n / single[a], 3), "P(a|b)": round(n / single[b], 3),
                     "jaccard": round(n / (single[a] + single[b] - n), 3)})
    return pd.DataFrame(rows)


def plot_cooccurrence(pairs, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(pairs["pair"][::-1], pairs["count"][::-1], color="#3182bd")
    ax.set_xlabel("# train documents with both sections")
    ax.set_title("Top label co-occurrence pairs (train)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return fig


def tertile_buckets(counts):
    """Split labels into head/mid/tail by train-frequency tertiles (stable tie-break by label order)."""
    s = counts.sort_values(ascending=False, kind="stable")
    head, mid, tail = (list(map(str, b)) for b in np.array_split(np.array(s.index), 3))
    return {
        "head": head, "mid": mid, "tail": tail,
        "ranges": {name: [int(counts[b].min()), int(counts[b].max())]
                   for name, b in [("head", head), ("mid", mid), ("tail", tail)]},
        "train_counts": {k: int(v) for k, v in counts.items()},
    }


def plot_buckets(counts, buckets, out_path):
    fig, ax = plt.subplots(figsize=(14, 4))
    s = counts.sort_values(ascending=False, kind="stable")
    colors = {"head": "#08519c", "mid": "#6baed6", "tail": "#c6dbef"}
    lab2b = {l: b for b in colors for l in buckets[b]}
    ax.bar(range(len(s)), s.values, color=[colors[lab2b[l]] for l in s.index])
    ax.set_yscale("log")
    for b, col in colors.items():
        lo, hi = buckets["ranges"][b]
        ax.bar([0], [0], color=col, label=f"{b}: {len(buckets[b])} labels, {lo}–{hi} docs")
    ax.legend()
    ax.set_xlabel("label rank")
    ax.set_ylabel("# train docs (log)")
    ax.set_title("Head / mid / tail label buckets (train-frequency tertiles)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return fig
