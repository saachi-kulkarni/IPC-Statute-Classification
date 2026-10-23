"""Shared multi-label evaluation used by every model (baselines, Colab BERT, analysis).

Conventions
-----------
* Columns of every y / prob matrix follow data/processed/label_list.json.
* Thresholds are tuned on DEV only, then applied to TEST.
* Standard per-model outputs (see `save_outputs`):
    dev_probs.npy, test_probs.npy   float32 [n_docs, n_labels]
    thresholds.json                 {"global": t, "per_label": [t_1..t_L], "label_list": [...]}
    metrics.json                    {"<strategy>": {"dev": {...}, "test": {...}}, ...}
  where strategies are "t0.5", "global", "per_label".

This file is self-contained (numpy + sklearn only) so it can be copied to Colab.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score

GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)
MIN_DEV_POSITIVES = 5


def binarize(labels_lists, label_list):
    """List of label lists -> multi-hot uint8 array [n, len(label_list)]. Unknown labels raise."""
    index = {l: i for i, l in enumerate(label_list)}
    y = np.zeros((len(labels_lists), len(label_list)), dtype=np.uint8)
    for r, labels in enumerate(labels_lists):
        for l in labels:
            y[r, index[str(l)]] = 1
    return y


def apply_thresholds(prob, thresholds):
    """thresholds: scalar or array of shape [n_labels]."""
    return (np.asarray(prob) >= np.asarray(thresholds, dtype=float)).astype(np.uint8)


def ensure_at_least_one(pred, prob):
    """Rows with no predicted label get their top-1 label. Returns a new array."""
    pred = np.array(pred, dtype=np.uint8, copy=True)
    empty = pred.sum(axis=1) == 0
    if empty.any():
        pred[np.where(empty)[0], np.asarray(prob)[empty].argmax(axis=1)] = 1
    return pred


def _f1_from_counts(tp, fp, fn):
    tp, fp, fn = (np.asarray(x, dtype=float) for x in (tp, fp, fn))
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def evaluate(y_true, y_prob, thresholds=0.5, at_least_one=True):
    """Full metric set for one threshold choice.

    Returns a dict with micro/macro P/R/F1, samples F1, Hamming loss,
    per-label F1 and support (lists in label_list order).
    """
    y_true = np.asarray(y_true).astype(np.uint8)
    pred = apply_thresholds(y_prob, thresholds)
    if at_least_one:
        pred = ensure_at_least_one(pred, y_prob)
    kw = dict(zero_division=0)
    return {
        "micro_f1": float(f1_score(y_true, pred, average="micro", **kw)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", **kw)),
        "micro_precision": float(precision_score(y_true, pred, average="micro", **kw)),
        "micro_recall": float(recall_score(y_true, pred, average="micro", **kw)),
        "macro_precision": float(precision_score(y_true, pred, average="macro", **kw)),
        "macro_recall": float(recall_score(y_true, pred, average="macro", **kw)),
        "samples_f1": float(f1_score(y_true, pred, average="samples", **kw)),
        "hamming_loss": float(hamming_loss(y_true, pred)),
        "avg_pred_labels": float(pred.sum(axis=1).mean()),
        "per_label_f1": f1_score(y_true, pred, average=None, **kw).astype(float).tolist(),
        "support": y_true.sum(axis=0).astype(int).tolist(),
    }


def tune_global_threshold(y_true_dev, prob_dev, grid=GRID, at_least_one=True):
    """Single threshold in [0.05, 0.95] (step 0.01) maximising dev micro-F1."""
    y = np.asarray(y_true_dev).astype(bool)
    prob = np.asarray(prob_dev)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        pred = prob >= t
        if at_least_one:
            pred = ensure_at_least_one(pred, prob).astype(bool)
        tp = np.logical_and(pred, y).sum()
        fp = np.logical_and(pred, ~y).sum()
        fn = np.logical_and(~pred, y).sum()
        f1 = float(_f1_from_counts(tp, fp, fn))
        if f1 > best_f1 + 1e-12:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


def tune_per_label_thresholds(y_true_dev, prob_dev, grid=GRID, global_t=None,
                              min_positives=MIN_DEV_POSITIVES):
    """Per-label threshold maximising that label's dev F1.

    Labels with fewer than `min_positives` dev positives fall back to the global
    threshold (tuned here if not given). Returns np.array [n_labels].
    """
    y = np.asarray(y_true_dev).astype(bool)
    prob = np.asarray(prob_dev)
    if global_t is None:
        global_t, _ = tune_global_threshold(y, prob, grid)
    n_labels = y.shape[1]
    thresholds = np.full(n_labels, global_t, dtype=float)
    pos = y.sum(axis=0)
    for j in range(n_labels):
        if pos[j] < min_positives:
            continue
        pj = prob[:, j][:, None] >= grid[None, :]          # [n, |grid|]
        yj = y[:, j][:, None]
        tp = (pj & yj).sum(0)
        fp = (pj & ~yj).sum(0)
        fn = (~pj & yj).sum(0)
        f1 = _f1_from_counts(tp, fp, fn)
        thresholds[j] = float(grid[int(np.argmax(f1))])     # first max -> lowest threshold on ties
    return thresholds


def bucket_report(per_label_f1, label_buckets, label_list):
    """Mean F1 per head/mid/tail bucket. label_buckets: dict with 'head','mid','tail' lists."""
    f1 = dict(zip(label_list, per_label_f1))
    return {b: float(np.mean([f1[l] for l in label_buckets[b]]))
            for b in ("head", "mid", "tail")}


def full_report(y_dev, prob_dev, y_test, prob_test, label_list, label_buckets=None):
    """Tune on dev, then evaluate dev+test under the t=0.5, global and per-label strategies."""
    t_glob, _ = tune_global_threshold(y_dev, prob_dev)
    t_lab = tune_per_label_thresholds(y_dev, prob_dev, global_t=t_glob)
    strategies = {"t0.5": 0.5, "global": t_glob, "per_label": t_lab}
    metrics = {}
    for name, t in strategies.items():
        metrics[name] = {}
        for split, y, p in [("dev", y_dev, prob_dev), ("test", y_test, prob_test)]:
            m = evaluate(y, p, t)
            if label_buckets is not None:
                m["buckets"] = bucket_report(m["per_label_f1"], label_buckets, label_list)
            metrics[name][split] = m
    thresholds = {"global": t_glob, "per_label": [float(x) for x in t_lab],
                  "label_list": list(label_list)}
    return metrics, thresholds


def save_outputs(out_dir, prob_dev, prob_test, thresholds, metrics, extra=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "dev_probs.npy", np.asarray(prob_dev, dtype=np.float32))
    np.save(out / "test_probs.npy", np.asarray(prob_test, dtype=np.float32))
    with open(out / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=1)
    payload = dict(metrics)
    if extra:
        payload["_meta"] = extra
    with open(out / "metrics.json", "w") as f:
        json.dump(payload, f, indent=1)


def summary_rows(model_name, metrics):
    """Flatten metrics.json into table rows (test split) for printing/comparison."""
    rows = []
    for strat, d in metrics.items():
        if strat.startswith("_"):
            continue
        m = d["test"]
        row = {"model": model_name, "thresholds": strat,
               "micro_f1": m["micro_f1"], "macro_f1": m["macro_f1"],
               "micro_p": m["micro_precision"], "micro_r": m["micro_recall"],
               "samples_f1": m["samples_f1"]}
        for b, v in m.get("buckets", {}).items():
            row[f"{b}_f1"] = v
        rows.append(row)
    return rows
