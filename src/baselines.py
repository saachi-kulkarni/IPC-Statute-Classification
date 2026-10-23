"""Non-neural baselines for ILSI statute identification.

Models (select with --models, comma-separated):
  keyword        keyword overlap with distinctive statute-text terms (no training docs used)
  regex_mention  DIAGNOSTIC: predict §N when the text literally cites "section N"/"N IPC"
  tfidf_lr       word 1-2gram TF-IDF + one-vs-rest LogisticRegression (C tuned on dev)
  tfidf_svm      same features + LinearSVC, Platt-scaled on dev (C tuned on dev)
  tfidf_char     char_wb 3-5gram TF-IDF + LR (optional; expensive)

Every model writes outputs/baselines/<name>/{dev_probs.npy,test_probs.npy,thresholds.json,metrics.json}.
Hyperparameters and thresholds are chosen on dev only; test is only scored.

Usage: python src/baselines.py --models keyword,regex_mention,tfidf_lr,tfidf_svm --n_jobs 4
"""
import argparse
import json
import re
import time
from pathlib import Path

import joblib
from joblib import parallel_config
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

import metrics as M
from common import (PROC, ROOT, SEED, cited_sections, load_buckets, load_label_list, load_split,
                    load_statutes, set_seed)

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- data
class Data:
    def __init__(self, proc_dir):
        self.label_list = load_label_list(proc_dir)
        self.buckets = load_buckets(proc_dir)
        self.statutes = load_statutes(proc_dir)
        self.df = {s: load_split(s, proc_dir) for s in ["train", "dev", "test"]}
        self.y = {s: M.binarize(d["labels"], self.label_list) for s, d in self.df.items()}
        self.text = {s: d["text"].tolist() for s, d in self.df.items()}
        log("loaded data: " + ", ".join(f"{s}={len(d)}" for s, d in self.df.items()))


def finish(name, data, prob_dev, prob_test, out_root, extra=None):
    metrics, thr = M.full_report(data.y["dev"], prob_dev, data.y["test"], prob_test,
                                 data.label_list, data.buckets)
    M.save_outputs(Path(out_root) / name, prob_dev, prob_test, thr, metrics, extra)
    for strat in ["t0.5", "global", "per_label"]:
        d, t = metrics[strat]["dev"], metrics[strat]["test"]
        log(f"{name:14s} {strat:9s} dev µF1={d['micro_f1']:.4f} MF1={d['macro_f1']:.4f} | "
            f"test µF1={t['micro_f1']:.4f} MF1={t['macro_f1']:.4f}")
    return metrics


def dev_score(y_dev, prob_dev):
    """Model-selection criterion: dev micro-F1 at the dev-tuned global threshold."""
    t, f1 = M.tune_global_threshold(y_dev, prob_dev)
    macro = M.evaluate(y_dev, prob_dev, t)["macro_f1"]
    return f1, macro, t


# ----------------------------------------------------------------------------- keyword
TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z]+\b"   # alphabetic tokens only (drop section numbers)


def keyword_model(data, out_root, k_grid=(5, 10, 15, 25), **_):
    st = data.statutes.set_index("section_id").loc[data.label_list, "text"].tolist()
    vec = TfidfVectorizer(stop_words="english", token_pattern=TOKEN_PATTERN, lowercase=True,
                          sublinear_tf=True)
    W_full = vec.fit_transform(st).toarray()                       # [L, V]
    vocab = np.array(vec.get_feature_names_out())
    counter = CountVectorizer(vocabulary=vocab, token_pattern=TOKEN_PATTERN, lowercase=True,
                              binary=True, dtype=np.float32)
    B = {s: counter.transform(data.text[s]) for s in ["dev", "test"]}  # [n, V] binary

    def scores(split, k):
        W = np.zeros_like(W_full)
        for j in range(W_full.shape[0]):
            top = np.argsort(-W_full[j])[:k]
            W[j, top] = W_full[j, top]
        W = W / np.maximum(W.sum(1, keepdims=True), 1e-12)           # rows sum to 1
        return np.asarray(B[split] @ W.T.astype(np.float32))          # weighted coverage in [0,1]

    results = {}
    for k in k_grid:
        results[k] = dev_score(data.y["dev"], scores("dev", k))
        log(f"keyword k={k}: dev µF1={results[k][0]:.4f} MF1={results[k][1]:.4f} t={results[k][2]}")
    best_k = max(results, key=lambda k: results[k][0])
    prob_dev, prob_test = scores("dev", best_k), scores("test", best_k)
    keywords = {}
    for j, sec in enumerate(data.label_list):
        top = np.argsort(-W_full[j])[:best_k]
        keywords[sec] = [(vocab[i], round(float(W_full[j, i]), 4)) for i in top if W_full[j, i] > 0]
    out = Path(out_root) / "keyword"
    out.mkdir(parents=True, exist_ok=True)
    (out / "keywords.json").write_text(json.dumps(keywords, indent=1))
    return finish("keyword", data, prob_dev, prob_test, out_root,
                  {"best_k": best_k, "k_grid_dev": {str(k): v[:2] for k, v in results.items()}})


# ----------------------------------------------------------------------------- regex (diagnostic)
def mentioned_sections(text):
    return cited_sections(text)


def regex_mention_model(data, out_root, **_):
    idx = {l: i for i, l in enumerate(data.label_list)}
    prior = data.y["train"].mean(0).astype(np.float32)

    def scores(split):
        P = np.tile(prior * 1e-3, (len(data.text[split]), 1))   # tiny prior -> top-1 fallback = 34
        for r, t in enumerate(data.text[split]):
            for s in mentioned_sections(t):
                if s in idx:
                    P[r, idx[s]] = 1.0
        return P

    return finish("regex_mention", data, scores("dev"), scores("test"), out_root,
                  {"note": "diagnostic: exploits explicit section citations left in the text"})


# ----------------------------------------------------------------------------- tf-idf features
def tfidf_features(data, kind, cache_dir):
    cache = Path(cache_dir) / f"tfidf_{kind}"
    files = {s: cache / f"X_{s}.npz" for s in ["train", "dev", "test"]}
    vec_path = cache / "vectorizer.joblib"
    if all(f.exists() for f in files.values()) and vec_path.exists():
        log(f"loading cached {kind} TF-IDF from {cache}")
        return {s: sp.load_npz(f) for s, f in files.items()}, joblib.load(vec_path)
    if kind == "word":
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=100_000, sublinear_tf=True,
                              min_df=3, max_df=0.9, dtype=np.float32)
    else:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=100_000,
                              sublinear_tf=True, min_df=3, max_df=0.9, dtype=np.float32)
    log(f"fitting {kind} TF-IDF on train")
    X = {"train": vec.fit_transform(data.text["train"])}
    vec.stop_words_ = None   # can hold millions of pruned n-grams; not needed for transform
    for s in ["dev", "test"]:
        X[s] = vec.transform(data.text[s])
    cache.mkdir(parents=True, exist_ok=True)
    for s, f in files.items():
        sp.save_npz(f, X[s].tocsr())
    joblib.dump(vec, vec_path)
    log(f"{kind} TF-IDF: train {X['train'].shape}, nnz/doc {X['train'].nnz / X['train'].shape[0]:.0f}")
    return X, vec


def fit_ovr(estimator, X, Y, n_jobs):
    """One-vs-rest fit with a *threading* pool.

    liblinear releases the GIL while training, so threads parallelise well and share X.
    The default process pool memory-maps X read-only, which liblinear rejects
    ("WRITEBACKIFCOPY base is read-only"); disabling memmapping instead re-sends X to
    every one of the 100 per-label tasks, which is far too slow and memory-hungry.
    X should already be float64 so liblinear does not copy it once per label.
    """
    with parallel_config(backend="threading"):
        return OneVsRestClassifier(estimator, n_jobs=n_jobs).fit(X, Y)


def fit_ovr_lr(X, Y, C, n_jobs):
    lr = LogisticRegression(C=C, class_weight="balanced", solver="liblinear", random_state=SEED)
    return fit_ovr(lr, X, Y, n_jobs)


def lr_model(data, out_root, name, X, vec, c_grid, n_jobs, save_model=False, **_):
    X_train = X["train"].astype(np.float64)
    best = None
    grid_log = {}
    for C in c_grid:
        t = time.time()
        clf = fit_ovr_lr(X_train, data.y["train"], C, n_jobs)
        prob_dev = clf.predict_proba(X["dev"])
        f1, macro, thr = dev_score(data.y["dev"], prob_dev)
        grid_log[str(C)] = [f1, macro]
        log(f"{name} C={C}: dev µF1={f1:.4f} MF1={macro:.4f} (t={thr}) [{time.time() - t:.0f}s]")
        if best is None or f1 > best[0]:
            best = (f1, C, clf, prob_dev)
        else:
            del clf
    _, C, clf, prob_dev = best
    prob_test = clf.predict_proba(X["test"])
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    if name == "tfidf_lr":
        feats = np.array(vec.get_feature_names_out())
        top = {}
        for j, sec in enumerate(data.label_list):
            coef = clf.estimators_[j].coef_.ravel()
            order = np.argsort(-coef)[:15]
            top[sec] = [(feats[i], round(float(coef[i]), 4)) for i in order]
        (out / "top_features.json").write_text(json.dumps(top, indent=1))
    if save_model:
        joblib.dump({"vectorizer": vec, "clf": clf, "label_list": data.label_list},
                    out / "model.joblib", compress=3)
    return finish(name, data, prob_dev, prob_test, out_root,
                  {"best_C": C, "C_grid_dev": grid_log})


def platt_fit(scores_dev, y_dev):
    """Per-label 1-D logistic calibration fitted on DEV decision scores."""
    ab = np.zeros((scores_dev.shape[1], 2))
    for j in range(scores_dev.shape[1]):
        lr = LogisticRegression(C=1e4, solver="lbfgs", max_iter=1000)
        lr.fit(scores_dev[:, [j]], y_dev[:, j])
        ab[j] = lr.coef_.ravel()[0], lr.intercept_[0]
    return ab


def platt_apply(scores, ab):
    return (1.0 / (1.0 + np.exp(-(scores * ab[:, 0] + ab[:, 1])))).astype(np.float32)


def svm_model(data, out_root, X, svm_c_grid, n_jobs, **_):
    X_train = X["train"].astype(np.float64)
    best = None
    grid_log = {}
    for C in svm_c_grid:
        t = time.time()
        clf = fit_ovr(LinearSVC(C=C, class_weight="balanced", random_state=SEED),
                      X_train, data.y["train"], n_jobs)
        s_dev = clf.decision_function(X["dev"])
        ab = platt_fit(s_dev, data.y["dev"])
        prob_dev = platt_apply(s_dev, ab)
        f1, macro, thr = dev_score(data.y["dev"], prob_dev)
        grid_log[str(C)] = [f1, macro]
        log(f"tfidf_svm C={C}: dev µF1={f1:.4f} MF1={macro:.4f} (t={thr}) [{time.time() - t:.0f}s]")
        if best is None or f1 > best[0]:
            best = (f1, C, clf, ab, prob_dev)
    _, C, clf, ab, prob_dev = best
    prob_test = platt_apply(clf.decision_function(X["test"]), ab)
    return finish("tfidf_svm", data, prob_dev, prob_test, out_root,
                  {"best_C": C, "C_grid_dev": grid_log,
                   "calibration": "per-label Platt sigmoid fitted on dev decision scores"})


# ----------------------------------------------------------------------------- main
def print_table(out_root, names):
    rows = []
    for n in names:
        p = Path(out_root) / n / "metrics.json"
        if p.exists():
            rows += M.summary_rows(n, json.loads(p.read_text()))
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(Path(out_root) / "results_baselines.csv", index=False)
        with pd.option_context("display.width", 200, "display.precision", 4):
            print("\nTEST results (thresholds tuned on dev):")
            print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="keyword,regex_mention,tfidf_lr,tfidf_svm")
    ap.add_argument("--proc_dir", default=str(PROC))
    ap.add_argument("--out_dir", default=str(ROOT / "outputs" / "baselines"))
    ap.add_argument("--cache_dir", default=str(PROC / "cache"))
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--c_grid", default="0.5,1,4,10")
    ap.add_argument("--svm_c_grid", default="0.05,0.1,0.5,1")
    args = ap.parse_args()
    set_seed(SEED)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    data = Data(args.proc_dir)
    c_grid = [float(c) for c in args.c_grid.split(",")]
    svm_c_grid = [float(c) for c in args.svm_c_grid.split(",")]
    X_word = vec_word = None
    for m in models:
        log(f"===== {m} =====")
        if m == "keyword":
            keyword_model(data, args.out_dir)
        elif m == "regex_mention":
            regex_mention_model(data, args.out_dir)
        elif m in ("tfidf_lr", "tfidf_svm"):
            if X_word is None:
                X_word, vec_word = tfidf_features(data, "word", args.cache_dir)
            if m == "tfidf_lr":
                lr_model(data, args.out_dir, "tfidf_lr", X_word, vec_word, c_grid, args.n_jobs,
                         save_model=True)
            else:
                svm_model(data, args.out_dir, X_word, svm_c_grid, args.n_jobs)
        elif m == "tfidf_char":
            X_char, vec_char = tfidf_features(data, "char", args.cache_dir)
            lr_model(data, args.out_dir, "tfidf_char", X_char, vec_char, c_grid, args.n_jobs)
            del X_char
        else:
            raise ValueError(f"unknown model {m}")
    print_table(args.out_dir, ["keyword", "regex_mention", "tfidf_lr", "tfidf_svm", "tfidf_char"])


if __name__ == "__main__":
    main()
