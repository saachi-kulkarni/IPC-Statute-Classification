"""Single-document inference for the demo app.

Loads the dev-selected best BERT run (colab_downloads/<run>/model) and the TF-IDF LR baseline,
applies each model's dev-tuned thresholds, and joins predictions with titles and the IPC->BNS
mapping. Only artefacts produced by earlier phases are read; nothing is written.
"""
import json
import re
import threading
from pathlib import Path

import numpy as np
import pandas as pd

import metrics as M
from common import PROC, ROOT, SEED, cited_sections, load_label_list, load_split

ANALYSIS = ROOT / "outputs" / "analysis"


def _thresholds(run_dir, strategy):
    thr = json.loads((Path(run_dir) / "thresholds.json").read_text())
    n = len(thr["label_list"])
    t = {"t0.5": np.full(n, 0.5), "global": np.full(n, thr["global"]),
         "per_label": np.array(thr["per_label"])}[strategy]
    return t, thr["label_list"]


def _test_scores(model_name):
    r = pd.read_csv(ANALYSIS / "results.csv")
    r = r[r.model == model_name].sort_values("dev_micro_f1", ascending=False)
    return (None, None) if r.empty else (float(r.iloc[0].test_micro_f1), float(r.iloc[0].test_macro_f1))


class BertModel:
    kind = "bert"

    def __init__(self, entry, label_list):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.dir = ROOT / entry["dir"]
        self.key = entry["model"]
        self.name = "InLegalBERT (fine-tuned)" if "inlegal" in entry["model"] else entry["model"]
        self.strategy = entry["thresholds"]
        self.thr, ll = _thresholds(self.dir, self.strategy)
        assert ll == label_list, "label order mismatch"
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(self.dir / "model")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.dir / "model").to(self.device).eval()
        self.max_len = json.loads((self.dir / "config.json").read_text()).get("max_len", 512)
        self._lock = threading.Lock()
        self._torch = torch

    def predict(self, text):
        with self._lock, self._torch.no_grad():
            n_tokens = len(self.tok(text)["input_ids"])
            enc = self.tok(text, truncation=True, max_length=self.max_len, return_tensors="pt")
            logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits
            prob = self._torch.sigmoid(logits.float()).cpu().numpy()[0]
        return prob, {"tokens": n_tokens, "truncated": n_tokens > self.max_len, "max_len": self.max_len}


class TfidfModel:
    kind = "tfidf"

    def __init__(self, entry, label_list):
        import joblib
        self.dir = ROOT / entry["dir"]
        self.key, self.name, self.strategy = entry["model"], "TF-IDF + Logistic Regression", entry["thresholds"]
        bundle = joblib.load(self.dir / "model.joblib")
        assert bundle["label_list"] == label_list, "label order mismatch"
        self.vec, self.clf = bundle["vectorizer"], bundle["clf"]
        self.thr, ll = _thresholds(self.dir, self.strategy)
        assert ll == label_list, "label order mismatch"
        self.device = "cpu"

    def predict(self, text):
        return self.clf.predict_proba(self.vec.transform([text]))[0], {"truncated": False}


class Predictor:
    def __init__(self):
        self.label_list = load_label_list()
        best = json.loads((ANALYSIS / "best_models.json").read_text())
        self.models = {}
        bert = best.get("best_bert")
        if bert and (ROOT / bert["dir"] / "model" / "config.json").exists():
            self.models[bert["model"]] = BertModel(bert, self.label_list)
        base = best.get("best_baseline")
        if base and (ROOT / base["dir"] / "model.joblib").exists():
            self.models[base["model"]] = TfidfModel(base, self.label_list)
        if not self.models:
            raise RuntimeError("No model found: run the baselines and/or place a BERT run in colab_downloads/")
        self.default = next(iter(self.models))           # BERT first when present, else TF-IDF
        mp = pd.read_csv(PROC / "ipc_to_bns_mapping.csv", dtype=str).fillna("")
        self.mapping = mp.set_index("ipc_section").to_dict("index")

    def meta(self):
        out = []
        for k, m in self.models.items():
            micro, macro = _test_scores(k)
            out.append({"key": k, "name": m.name, "kind": m.kind, "device": m.device, "thresholds": m.strategy,
                        "test_micro_f1": micro, "test_macro_f1": macro, "default": k == self.default})
        return {"models": out, "n_labels": len(self.label_list), "examples": examples()}

    def predict(self, text, model_key=None, top_k=10):
        m = self.models.get(model_key) or self.models[self.default]
        prob, info = m.predict(text)
        pred = M.ensure_at_least_one(M.apply_thresholds(prob[None, :], m.thr), prob[None, :])[0].astype(bool)
        rows = []
        for rank, j in enumerate(np.argsort(-prob)[:top_k], 1):
            ipc = self.label_list[j]
            mp = self.mapping.get(ipc, {})
            rows.append({"rank": rank, "ipc": ipc, "title": mp.get("ipc_title", ""), "prob": round(float(prob[j]), 4),
                         "threshold": round(float(m.thr[j]), 2), "predicted": bool(pred[j]),
                         "bns": mp.get("bns_sections", ""), "mapping_type": mp.get("mapping_type", ""),
                         "verified": mp.get("verified", ""), "notes": mp.get("notes", "")})
        predicted = [self.label_list[j] for j in np.argsort(-prob) if pred[j]]
        cited = sorted(cited_sections(text) & set(self.label_list),
                       key=lambda x: (int(re.match(r"\d+", x).group(0)), x))
        return {"model": {"key": m.key, "name": m.name, "thresholds": m.strategy}, "top": rows,
                "predicted": predicted, "cited_in_text": cited, "words": len(text.split()), **info}


_EXAMPLES = None


NARRATIVE = re.compile(r"prosecution case|prosecution story|F\.?I\.?R\.?|first information report|complainant", re.I)


def examples(seed=SEED):
    """Three test-set cases (150-400 words, containing a fact narrative: 'prosecution case', 'FIR'
    or 'complainant') whose gold labels include 302, 420 and 498A respectively.
    Picked at random (seed 42) among those; NOT selected for being predicted correctly."""
    global _EXAMPLES
    if _EXAMPLES is None:
        test = load_split("test")
        test = test[test.text.str.split().str.len().between(150, 400)]
        test = test[test.text.str.contains(NARRATIVE)]
        rng = np.random.default_rng(seed)
        out = []
        for anchor, theme in [("302", "Homicide"), ("420", "Cheating"), ("498A", "Cruelty to a married woman")]:
            cand = test[test.labels.map(lambda ls, a=anchor: a in ls)]
            r = cand.iloc[int(rng.integers(len(cand)))]
            out.append({"id": r.id, "theme": theme, "text": r.text, "gold": list(r.labels)})
        _EXAMPLES = out
    return _EXAMPLES
