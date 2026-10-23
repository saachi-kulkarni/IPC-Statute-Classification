"""IPC -> BNS transfer analysis (Phase 6).

1. Deterministic relabelling: map test gold + predictions of the best models to BNS on the
   clean subset (mapping_type == one_to_one AND verified == yes) and recompute F1.
   NOTE: a one-to-one relabelling is a renaming of columns, so F1 on the clean subset is
   identical under IPC and BNS names; the script asserts this. What matters is coverage.
2. Coverage: how many labels / test documents / label occurrences map cleanly vs not.
3. Zero-shot retrieval with the fine-tuned encoder (mean-pooled last layer, cosine):
   a) calibration: facts vs IPC statute texts (always available), recall@5 over 100 IPC sections
   b) BNS: facts vs BNS section texts, recall@5 against mapped gold, clean subset only.
      Needs data/raw/bns_sections.csv (columns: section_id,text); skipped if absent.

Usage: python src/bns_transfer.py [--n_zero_shot 2000] [--skip_zero_shot]
"""
import argparse
import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import metrics as M
from common import PROC, ROOT, SEED, load_label_list, load_split, load_statutes, set_seed

OUT = ROOT / "outputs" / "bns"


def load_mapping():
    m = pd.read_csv(PROC / "ipc_to_bns_mapping.csv", dtype=str).fillna("")
    m["clean"] = (m.mapping_type == "one_to_one") & (m.verified == "yes")
    return m.set_index("ipc_section")


def load_preds(best_entry):
    d = ROOT / best_entry["dir"]
    prob = np.load(d / "test_probs.npy")
    thr = json.loads((d / "thresholds.json").read_text())
    t = {"t0.5": 0.5, "global": thr["global"], "per_label": np.array(thr["per_label"])}[best_entry["thresholds"]]
    return prob, M.ensure_at_least_one(M.apply_thresholds(prob, t), prob)


def f1s(y, p):
    return {"micro_f1": round(float(f1_score(y, p, average="micro", zero_division=0)), 4),
            "macro_f1": round(float(f1_score(y, p, average="macro", zero_division=0)), 4)}


# ----------------------------------------------------------------------------- 1 + 2
def coverage(mapping, label_list, y_test, y_train):
    rows = []
    for j, lab in enumerate(label_list):
        r = mapping.loc[lab]
        rows.append({"ipc": lab, "bns": r.bns_sections, "mapping_type": r.mapping_type, "verified": r.verified,
                     "clean": bool(r.clean), "test_occurrences": int(y_test[:, j].sum()),
                     "train_occurrences": int(y_train[:, j].sum())})
    df = pd.DataFrame(rows)
    clean = df.clean.to_numpy()
    doc_all_clean = y_test[:, ~clean].sum(1) == 0
    doc_any_clean = y_test[:, clean].sum(1) > 0
    summary = {
        "labels_total": len(df),
        "labels_clean (one_to_one & verified)": int(clean.sum()),
        "labels_by_mapping_type": df.mapping_type.value_counts().to_dict(),
        "labels_unverified": df.loc[df.verified == "no", "ipc"].tolist(),
        "labels_not_clean": df.loc[~clean, "ipc"].tolist(),
        "test_docs_total": int(len(y_test)),
        "test_docs_all_gold_clean": int(doc_all_clean.sum()),
        "test_docs_with_some_ambiguous_gold": int((~doc_all_clean).sum()),
        "test_docs_with_no_clean_gold": int((~doc_any_clean).sum()),
        "test_label_mass_clean": round(float(df.loc[clean, "test_occurrences"].sum() / df.test_occurrences.sum()), 4),
        "train_label_mass_clean": round(float(df.loc[clean, "train_occurrences"].sum() / df.train_occurrences.sum()), 4),
        "test_label_mass_by_type": (df.groupby("mapping_type").test_occurrences.sum()
                                    / df.test_occurrences.sum()).round(4).to_dict(),
    }
    return df, summary, clean, doc_all_clean


def remap_eval(name, y_test, pred, clean, doc_all_clean, label_list, mapping):
    ipc_clean = [l for l, c in zip(label_list, clean) if c]
    bns_names = [mapping.loc[l, "bns_sections"] for l in ipc_clean]
    assert len(set(bns_names)) == len(bns_names), "clean mapping must be injective"
    order = np.argsort(bns_names)                       # columns re-ordered under BNS names
    y_ipc, p_ipc = y_test[:, clean], pred[:, clean]
    y_bns, p_bns = y_ipc[:, order], p_ipc[:, order]
    ipc_scores, bns_scores = f1s(y_ipc, p_ipc), f1s(y_bns, p_bns)
    assert ipc_scores == bns_scores, "one-to-one relabelling must not change F1"
    return {
        "model": name,
        "all_100_ipc_labels": f1s(y_test, pred),
        "clean_label_subset (BNS names; identical under IPC names)": bns_scores,
        "clean_subset_on_docs_whose_gold_is_all_clean": f1s(y_bns[doc_all_clean], p_bns[doc_all_clean]),
        "clean_subset_on_docs_with_some_ambiguous_gold": f1s(y_bns[~doc_all_clean], p_bns[~doc_all_clean]),
        "n_clean_labels": int(clean.sum()),
    }


# ----------------------------------------------------------------------------- 3 zero-shot
def main_section(s):
    m = re.match(r"\s*(\d+[A-Z]?)", str(s))
    return m.group(1) if m else str(s)


def embed(texts, model, tok, device, max_len=512, batch=16):
    import torch
    vecs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], truncation=True, max_length=max_len, padding=True,
                      return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            vecs.append(torch.nn.functional.normalize(v, dim=-1).float().cpu().numpy())
    return np.concatenate(vecs)


def recall_at_k(scores, gold_sets, cand_names, k=5):
    topk = np.argsort(-scores, axis=1)[:, :k]
    rec = [len({cand_names[j] for j in row} & g) / len(g) for row, g in zip(topk, gold_sets) if g]
    return float(np.mean(rec)), len(rec)


def zero_shot(args, best_bert, label_list, mapping, test, y_test, clean, doc_all_clean, sup_prob):
    model_dir = ROOT / best_bert["dir"] / "model"
    if not (model_dir / "config.json").exists():
        return {"status": f"skipped: no fine-tuned weights at {model_dir}"}
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).to(device)

    rng = np.random.default_rng(SEED)
    idx_all = np.where(doc_all_clean)[0]
    idx = np.sort(rng.choice(idx_all, size=min(args.n_zero_shot, len(idx_all)), replace=False))
    print(f"zero-shot: embedding {len(idx)} test docs (all-clean gold, seed {SEED}) on {device} ...", flush=True)
    F = embed(test["text"].iloc[idx].tolist(), model, tok, device)
    res = {"status": "ok", "encoder": str(model_dir.relative_to(ROOT)), "pooling": "mean of last layer, cosine",
           "n_docs": int(len(idx)), "doc_selection": f"random sample (seed {SEED}) of test docs whose gold labels are all clean-mapped"}

    # a) calibration: same encoder, IPC statute texts (100 candidates)
    st = load_statutes().set_index("section_id").loc[label_list, "text"].tolist()
    S_ipc = embed(st, model, tok, device)
    gold_ipc = [set(np.array(label_list)[y_test[i].astype(bool)]) for i in idx]
    r_ipc, _ = recall_at_k(F @ S_ipc.T, gold_ipc, label_list)
    r_sup, _ = recall_at_k(sup_prob[idx], gold_ipc, label_list)
    res["ipc_statute_retrieval"] = {"candidates": len(label_list), "recall@5": round(r_ipc, 4),
                                    "random_recall@5": round(5 / len(label_list), 4),
                                    "supervised_classifier_recall@5 (same docs)": round(r_sup, 4)}

    # b) BNS section texts, if provided by the user
    bns_file = ROOT / "data" / "raw" / "bns_sections.csv"
    if not bns_file.exists():
        res["bns_statute_retrieval"] = {"status": f"skipped: provide {bns_file.relative_to(ROOT)} (columns section_id,text)"}
        return res
    bns = pd.read_csv(bns_file, dtype=str).dropna()
    bns["main"] = bns.section_id.map(main_section)
    bns = bns.groupby("main", sort=False)["text"].apply(" ".join).reset_index()
    S_bns = embed(bns.text.tolist(), model, tok, device)
    names = bns.main.tolist()
    ipc2bns = {l: main_section(mapping.loc[l, "bns_sections"]) for l, c in zip(label_list, clean) if c}
    gold_bns = [{ipc2bns[l] for l in g if l in ipc2bns} for g in gold_ipc]
    covered = sum(all(b in set(names) for b in g) for g in gold_bns)
    r_bns, n_bns = recall_at_k(F @ S_bns.T, gold_bns, names)
    res["bns_statute_retrieval"] = {"status": "ok", "candidates": len(names), "recall@5": round(r_bns, 4),
                                    "random_recall@5": round(5 / len(names), 4),
                                    "docs_scored": n_bns, "docs_with_all_gold_in_bns_file": int(covered)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_zero_shot", type=int, default=2000)
    ap.add_argument("--skip_zero_shot", action="store_true")
    args = ap.parse_args()
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    label_list = load_label_list()
    mapping = load_mapping()
    test = load_split("test")
    y_test = M.binarize(test["labels"], label_list)
    y_train = M.binarize(load_split("train", with_text=False)["labels"], label_list)
    best = json.loads((ROOT / "outputs" / "analysis" / "best_models.json").read_text())

    cov_df, cov, clean, doc_all_clean = coverage(mapping, label_list, y_test, y_train)
    cov_df.to_csv(OUT / "label_coverage.csv", index=False)
    print(json.dumps(cov, indent=1))

    remap, bert_prob = {}, None
    for key in ["best_baseline", "best_bert"]:
        if key in best:
            prob, pred = load_preds(best[key])
            remap[key] = remap_eval(best[key]["model"], y_test, pred, clean, doc_all_clean, label_list, mapping)
            if key == "best_bert":
                bert_prob = prob
    (OUT / "remap_metrics.json").write_text(json.dumps({"coverage": cov, "remap": remap}, indent=1))
    print(json.dumps(remap, indent=1))

    if args.skip_zero_shot or bert_prob is None:
        print("zero-shot skipped (flag or no BERT run); existing outputs/bns/zero_shot.json left unchanged")
        return
    zs = zero_shot(args, best["best_bert"], label_list, mapping, test, y_test, clean, doc_all_clean, bert_prob)
    (OUT / "zero_shot.json").write_text(json.dumps(zs, indent=1))
    print(json.dumps(zs, indent=1))


if __name__ == "__main__":
    main()
