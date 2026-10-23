"""Analysis helpers for notebooks/02_analysis.ipynb (Phase 5).

All comparisons use each model's dev-selected threshold strategy (best_models.json /
the strategy with the highest dev micro-F1), applied unchanged to test.
"""
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import metrics as M
from common import ROOT, cited_sections, first_citation_word, load_buckets, load_label_list, load_split

OUT = ROOT / "outputs" / "analysis"


# ----------------------------------------------------------------------------- loading
class Model:
    """Probabilities + dev-chosen thresholds + binary test predictions of one model."""

    def __init__(self, name, directory, strategy, label_list):
        d = ROOT / directory
        self.name, self.strategy = name, strategy
        self.prob_test = np.load(d / "test_probs.npy")
        thr = json.loads((d / "thresholds.json").read_text())
        assert thr["label_list"] == label_list
        self.thr = {"t0.5": 0.5, "global": thr["global"],
                    "per_label": np.array(thr["per_label"])}[strategy]
        self.pred = M.ensure_at_least_one(M.apply_thresholds(self.prob_test, self.thr), self.prob_test)


def best_strategy_per_model(results):
    """{model: strategy with highest dev micro-F1}."""
    idx = results.groupby("model")["dev_micro_f1"].idxmax()
    return results.loc[idx].set_index("model")["thresholds"].to_dict()


def load_context():
    label_list = load_label_list()
    buckets = load_buckets()
    results = pd.read_csv(OUT / "results.csv")
    best = json.loads((OUT / "best_models.json").read_text())
    test = load_split("test")
    y_test = M.binarize(test["labels"], label_list)
    train_counts = pd.Series(buckets["train_counts"])[label_list]
    return dict(label_list=label_list, buckets=buckets, results=results, best=best,
                test=test, y_test=y_test, train_counts=train_counts)


def load_model(ctx, key):
    b = ctx["best"][key]
    return Model(b["model"], b["dir"], b["thresholds"], ctx["label_list"])


# ----------------------------------------------------------------------------- plots
MODEL_ORDER = ["keyword", "regex_mention", "tfidf_lr", "tfidf_svm", "tfidf_char",
               "bert-base-uncased_head_512", "inlegalbert_head_512", "inlegalbert_head_tail_512"]


def dev_selected_rows(results):
    strat = best_strategy_per_model(results)
    rows = pd.concat([results[(results.model == m) & (results.thresholds == s)] for m, s in strat.items()])
    present = [m for m in MODEL_ORDER if m in set(rows.model)]
    present += sorted(set(rows.model) - set(present))
    return rows.set_index("model").loc[present].reset_index()


def plot_micro_macro(results, out_path):
    rows = dev_selected_rows(results)
    labels = [f"{m}\n({s})" + ("\n[diagnostic]" if m == "regex_mention" else "")
              for m, s in zip(rows.model, rows.thresholds)]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 0.2, rows.test_micro_f1, 0.4, label="micro-F1", color="#2c7fb8")
    ax.bar(x + 0.2, rows.test_macro_f1, 0.4, label="macro-F1", color="#7fcdbb")
    for i, (a, b) in enumerate(zip(rows.test_micro_f1, rows.test_macro_f1)):
        ax.text(i - 0.2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.2, b + 0.01, f"{b:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("test F1")
    ax.set_ylim(0, 1)
    ax.set_title("Test micro vs macro F1 (each model at its dev-selected threshold strategy)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return rows


def per_label_f1(y, pred):
    y, pred = y.astype(bool), pred.astype(bool)
    tp = (y & pred).sum(0)
    fp = (~y & pred).sum(0)
    fn = (y & ~pred).sum(0)
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)


def plot_f1_vs_freq(ctx, models, out_path):
    freq = ctx["train_counts"].to_numpy()
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.5), sharey=True)
    out = {}
    for ax, m in zip(np.atleast_1d(axes), models):
        f1 = per_label_f1(ctx["y_test"], m.pred)
        out[m.name] = f1
        ax.scatter(np.log10(freq), f1, s=14, alpha=0.7)
        r = np.corrcoef(np.log10(freq), f1)[0, 1]
        ax.set_title(f"{m.name} ({m.strategy})  Pearson r={r:.2f}", fontsize=10)
        ax.set_xlabel("log10(train documents with label)")
        for i in np.argsort(f1)[:5]:
            ax.annotate(f"§{ctx['label_list'][i]}", (np.log10(freq[i]), f1[i]), fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel("per-label test F1")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return pd.DataFrame(out, index=ctx["label_list"])


def plot_buckets(results, out_path):
    rows = dev_selected_rows(results).set_index("model")[["test_head_f1", "test_mid_f1", "test_tail_f1"]]
    rows.columns = ["head", "mid", "tail"]
    ax = rows.plot.bar(figsize=(11, 4.5), color=["#08519c", "#6baed6", "#c6dbef"], edgecolor="k")
    ax.set_ylabel("mean per-label test F1")
    ax.set_title("F1 by label-frequency bucket (train tertiles)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    return rows


def improvement_table(f1_df, base, bert, statutes, train_counts, k=15):
    d = pd.DataFrame({"train_docs": train_counts, f"F1 {base}": f1_df[base], f"F1 {bert}": f1_df[bert]})
    d["delta (BERT - base)"] = d[f"F1 {bert}"] - d[f"F1 {base}"]
    d["statute (first 70 chars)"] = [statutes.get(s, "")[:70] for s in d.index]
    d = d.sort_values("delta (BERT - base)", ascending=False)
    return d.head(k), d.tail(k).iloc[::-1]


# ----------------------------------------------------------------------------- confusions
def confusion_pairs(y, pred, label_list):
    """Row per (a, b): #docs where a is a FALSE POSITIVE while b is gold; rate = count / #docs with b gold."""
    y, pred = y.astype(np.int32), pred.astype(np.int32)
    C = (pred * (1 - y)).T @ y                  # C[a, b]
    np.fill_diagonal(C, 0)
    support = y.sum(0)
    a_idx, b_idx = np.nonzero(C)
    df = pd.DataFrame({"pred_FP": np.array(label_list)[a_idx], "gold": np.array(label_list)[b_idx],
                       "count": C[a_idx, b_idx], "rate_given_gold": C[a_idx, b_idx] / np.maximum(support[b_idx], 1)})
    return df.sort_values("count", ascending=False).reset_index(drop=True)


FOCUS_GROUPS = [("302", "304"), ("323", "324", "325"), ("420", "406"), ("147", "148", "149"), ("34", "149")]


def focus_confusions(conf_by_model, label_list):
    rows = []
    for group in FOCUS_GROUPS:
        for a, b in [(a, b) for a in group for b in group if a != b]:
            if a not in label_list or b not in label_list:
                continue
            r = {"pred_FP": a, "gold": b}
            for name, conf in conf_by_model.items():
                hit = conf[(conf.pred_FP == a) & (conf.gold == b)]
                r[f"{name} count"] = int(hit["count"].iloc[0]) if len(hit) else 0
                r[f"{name} rate"] = round(float(hit["rate_given_gold"].iloc[0]), 3) if len(hit) else 0.0
            rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- error analysis
def doc_f1(y, pred):
    y, pred = y.astype(bool), pred.astype(bool)
    tp = (y & pred).sum(1)
    return 2 * tp / np.maximum(y.sum(1) + pred.sum(1), 1)


def mentioned(text):
    return cited_sections(text)


def first_mention_word(text, gold):
    """Word position of the first citation naming a gold section (None if never named)."""
    return first_citation_word(text, set(gold))


def error_cases(ctx, a, b, k=8, win=0.75, lose=0.25, seed=42):
    """k test docs where model `a` does well (doc-F1 >= win) and `b` does badly (doc-F1 <= lose)."""
    y = ctx["y_test"].astype(bool)
    fa, fb = doc_f1(y, a.pred), doc_f1(y, b.pred)
    idx = np.where((fa >= win) & (fb <= lose))[0]
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(idx, size=min(k, len(idx)), replace=False)) if len(idx) else idx
    L = np.array(ctx["label_list"])
    rows = []
    for i in pick:
        text = ctx["test"]["text"].iloc[i]
        gold = [str(g) for g in L[y[i]]]
        rows.append({
            "row": int(i), "id": ctx["test"]["id"].iloc[i], "words": len(text.split()),
            "gold": gold,
            f"pred {a.name}": [str(x) for x in L[a.pred[i].astype(bool)]],
            f"pred {b.name}": [str(x) for x in L[b.pred[i].astype(bool)]],
            f"docF1 {a.name}": round(float(fa[i]), 2), f"docF1 {b.name}": round(float(fb[i]), 2),
            "gold named in text": sorted(set(gold) & mentioned(text)),
            "first gold mention at word": first_mention_word(text, set(gold)),
            "facts (first 500 chars)": text[:500],
        })
    return pd.DataFrame(rows), len(idx)


def mention_position_stats(ctx, word_limit=400):
    """Where in each test document a gold section is first named, vs the ~400-word BERT window."""
    y = ctx["y_test"].astype(bool)
    L = np.array(ctx["label_list"])
    s = pd.Series([first_mention_word(t, set(L[y[i]])) for i, t in enumerate(ctx["test"]["text"])],
                  dtype="float")
    return {"docs": len(s),
            "a gold section is named somewhere": round(float(s.notna().mean()), 3),
            "first named within first ~400 words": round(float((s < word_limit).mean()), 3),
            "first named only after ~400 words": round(float((s >= word_limit).mean()), 3)}, s


# ----------------------------------------------------------------------------- bootstrap
def _counts(y, pred):
    y, pred = y.astype(bool), pred.astype(bool)
    return ((y & pred).astype(np.float32), (~y & pred).astype(np.float32), (y & ~pred).astype(np.float32))


def _scores(w, tp, fp, fn):
    TP, FP, FN = w @ tp, w @ fp, w @ fn                      # per-label sums under resampling weights
    micro = 2 * TP.sum() / max(2 * TP.sum() + FP.sum() + FN.sum(), 1e-12)
    den = 2 * TP + FP + FN
    macro = np.mean(np.where(den > 0, 2 * TP / np.maximum(den, 1e-12), 0.0))
    return micro, macro


def paired_bootstrap(y, pred_a, pred_b, n_boot=1000, seed=42):
    """Document-level paired bootstrap. 95% CIs for a, b and (b - a), plus a two-sided p-value."""
    n = len(y)
    ca, cb = _counts(y, pred_a), _counts(y, pred_b)
    ones = np.ones(n, np.float32)
    point_a, point_b = _scores(ones, *ca), _scores(ones, *cb)
    rng = np.random.default_rng(seed)
    A, B = np.zeros((n_boot, 2)), np.zeros((n_boot, 2))
    for r in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float32)
        A[r], B[r] = _scores(w, *ca), _scores(w, *cb)
    D = B - A
    out = {}
    for j, metric in enumerate(["micro_f1", "macro_f1"]):
        p = 2 * min((D[:, j] <= 0).mean(), (D[:, j] >= 0).mean())
        out[metric] = {"a": float(point_a[j]), "a_CI95": np.percentile(A[:, j], [2.5, 97.5]).round(4).tolist(),
                       "b": float(point_b[j]), "b_CI95": np.percentile(B[:, j], [2.5, 97.5]).round(4).tolist(),
                       "diff_b_minus_a": float(point_b[j] - point_a[j]),
                       "diff_CI95": np.percentile(D[:, j], [2.5, 97.5]).round(4).tolist(),
                       "p_two_sided": f"< {1 / n_boot:.3g}" if p == 0 else round(float(p), 4)}
    return out


def f1_by_mention_group(ctx, models, positions, word_limit=400):
    """Micro-F1 of each model on docs grouped by where a gold section is first named in the text."""
    groups = {"named in first ~400 words": positions < word_limit,
              "named only after ~400 words": positions >= word_limit,
              "never named": positions.isna()}
    y = ctx["y_test"].astype(bool)
    rows = []
    for g, mask in groups.items():
        mask = mask.to_numpy()
        r = {"group": g, "docs": int(mask.sum())}
        for m in models:
            p = m.pred[mask].astype(bool)
            tp, fp, fn = (y[mask] & p).sum(), (~y[mask] & p).sum(), (y[mask] & ~p).sum()
            r[m.name] = round(2 * tp / max(2 * tp + fp + fn, 1), 4)
        rows.append(r)
    return pd.DataFrame(rows)
