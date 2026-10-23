"""Validate every model's saved outputs and build the master results table.

Sources
  outputs/baselines/<name>/      Mac baselines
  colab_downloads/<run_name>/    Colab BERT runs (smoke-test runs are skipped)

For every model the dev/test probabilities are checked (shape, label order, row order,
range) and ALL metrics are recomputed from probs + saved thresholds with src/metrics.py,
then cross-checked against the model's own metrics.json.

Outputs (outputs/analysis/)
  results.csv / results.md   model x threshold strategy (test metrics + dev micro/macro F1)
  validation.json            per-model validation report
  best_models.json           best baseline and best BERT, selected on DEV micro-F1

Usage: python src/compare.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import metrics as M
from common import PROC, ROOT, load_buckets, load_label_list, load_split

STRATEGIES = ["t0.5", "global", "per_label"]
DIAGNOSTIC = {"regex_mention"}          # reported, but never chosen as "best baseline"


def to_markdown(df, floatfmt="{:.4f}"):
    cells = [[floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
             for row in df.itertuples(index=False)]
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    return "\n".join([head, sep] + ["| " + " | ".join(r) + " |" for r in cells])


def discover(baseline_dir, colab_dir):
    models = []
    for d in sorted(Path(baseline_dir).glob("*/")):
        if (d / "test_probs.npy").exists():
            models.append({"name": d.name, "family": "baseline", "dir": d})
    for d in sorted(Path(colab_dir).glob("*/")):
        if not (d / "test_probs.npy").exists():
            continue
        cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
        if cfg.get("smoke_test") or d.name.endswith("_smoke"):
            continue
        models.append({"name": d.name, "family": "bert", "dir": d, "config": cfg})
    return models


def thresholds_of(thr, strategy):
    return {"t0.5": 0.5, "global": thr["global"], "per_label": np.array(thr["per_label"])}[strategy]


def validate(model, label_list, n, ids):
    d, problems = model["dir"], []
    prob = {s: np.load(d / f"{s}_probs.npy") for s in ["dev", "test"]}
    thr = json.loads((d / "thresholds.json").read_text())
    for s in ["dev", "test"]:
        if prob[s].shape != (n[s], len(label_list)):
            problems.append(f"{s}_probs shape {prob[s].shape} != {(n[s], len(label_list))}")
        if not np.isfinite(prob[s]).all():
            problems.append(f"{s}_probs has NaN/inf")
        if prob[s].min() < -1e-5 or prob[s].max() > 1 + 1e-5:
            problems.append(f"{s}_probs outside [0,1]: {prob[s].min():.3g}..{prob[s].max():.3g}")
    if thr.get("label_list") != label_list:
        problems.append("thresholds.json label_list differs from data/processed/label_list.json")
    if len(thr.get("per_label", [])) != len(label_list):
        problems.append("per_label thresholds length mismatch")
    cfg = model.get("config")
    if cfg is not None:
        if cfg.get("label_list") != label_list:
            problems.append("config.json label_list differs")
        if cfg.get("n_rows", {}).get("test") != n["test"]:
            problems.append(f"config n_rows {cfg.get('n_rows')} != full splits")
        sid = d / "split_ids.json"
        if sid.exists():
            saved = json.loads(sid.read_text())
            for s in ["dev", "test"]:
                if saved.get(s) != ids[s]:
                    problems.append(f"{s} row order (split_ids.json) differs from parquet")
        else:
            problems.append("split_ids.json missing (row order unverified)")
    return prob, thr, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", default=str(ROOT / "outputs" / "baselines"))
    ap.add_argument("--colab_dir", default=str(ROOT / "colab_downloads"))
    ap.add_argument("--out_dir", default=str(ROOT / "outputs" / "analysis"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    label_list, buckets = load_label_list(PROC), load_buckets(PROC)
    dfs = {s: load_split(s, PROC, with_text=False) for s in ["dev", "test"]}
    Y = {s: M.binarize(d["labels"], label_list) for s, d in dfs.items()}
    n = {s: len(d) for s, d in dfs.items()}
    ids = {s: d["id"].tolist() for s, d in dfs.items()}

    rows, report = [], {}
    for model in discover(args.baseline_dir, args.colab_dir):
        prob, thr, problems = validate(model, label_list, n, ids)
        stored = json.loads((model["dir"] / "metrics.json").read_text())
        for strat in STRATEGIES:
            t = thresholds_of(thr, strat)
            m_dev = M.evaluate(Y["dev"], prob["dev"], t)
            m = M.evaluate(Y["test"], prob["test"], t)
            b = M.bucket_report(m["per_label_f1"], buckets, label_list)
            drift = abs(m["micro_f1"] - stored[strat]["test"]["micro_f1"])
            if drift > 1e-6:
                problems.append(f"{strat}: recomputed test micro-F1 differs from metrics.json by {drift:.2e}")
            rows.append({
                "model": model["name"], "family": model["family"], "thresholds": strat,
                "test_micro_f1": m["micro_f1"], "test_macro_f1": m["macro_f1"],
                "test_micro_p": m["micro_precision"], "test_micro_r": m["micro_recall"],
                "test_samples_f1": m["samples_f1"],
                "test_head_f1": b["head"], "test_mid_f1": b["mid"], "test_tail_f1": b["tail"],
                "dev_micro_f1": m_dev["micro_f1"], "dev_macro_f1": m_dev["macro_f1"],
                "avg_pred_labels": m["avg_pred_labels"],
                "diagnostic": model["name"] in DIAGNOSTIC,
            })
        status = "OK" if not problems else "FAIL"
        report[model["name"]] = {"family": model["family"], "status": status, "problems": problems,
                                 "dir": str(model["dir"].relative_to(ROOT))}
        print(f"[{status}] {model['family']:8s} {model['name']}" + (f": {problems}" if problems else ""))

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    show = df.drop(columns=["diagnostic"]).copy()
    show["model"] = [m + " (diagnostic)" if m in DIAGNOSTIC else m for m in show["model"]]
    (out / "results.md").write_text(
        "# Master results\n\nTest-set metrics; hyperparameters and thresholds tuned on dev only. "
        "`regex_mention` is a diagnostic (predicts sections cited verbatim in the text).\n\n"
        + to_markdown(show) + "\n")
    (out / "validation.json").write_text(json.dumps(report, indent=1))

    passed = np.array([report[m]["status"] == "OK" for m in df["model"]])
    ok = df[passed & ~df["diagnostic"].to_numpy()]
    best = {}
    for fam in ["baseline", "bert"]:
        sub = ok[ok["family"] == fam]
        if sub.empty:
            continue
        r = sub.sort_values("dev_micro_f1", ascending=False).iloc[0]
        best[f"best_{fam}"] = {"model": r["model"], "thresholds": r["thresholds"],
                               "dir": report[r["model"]]["dir"],
                               "dev_micro_f1": float(r["dev_micro_f1"]),
                               "test_micro_f1": float(r["test_micro_f1"]),
                               "test_macro_f1": float(r["test_macro_f1"]),
                               "selection": "highest DEV micro-F1 over (model, threshold strategy)"}
    (out / "best_models.json").write_text(json.dumps(best, indent=1))

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\n" + show.round(4).to_string(index=False))
    print("\nbest (selected on dev):", json.dumps(best, indent=1))
    if any(v["status"] != "OK" for v in report.values()):
        raise SystemExit("validation failed for some models (see above)")


if __name__ == "__main__":
    main()
