import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import metrics as M  # noqa: E402

LABELS = ["34", "302", "420"]


def test_binarize_order_and_values():
    y = M.binarize([["302"], ["34", "420"], []], LABELS)
    assert y.tolist() == [[0, 1, 0], [1, 0, 1], [0, 0, 0]]
    assert y.dtype == np.uint8


def test_binarize_unknown_label_raises():
    with pytest.raises(KeyError):
        M.binarize([["999"]], LABELS)


def test_ensure_at_least_one_only_fills_empty_rows():
    pred = np.array([[0, 0, 0], [1, 0, 0]])
    prob = np.array([[0.1, 0.3, 0.2], [0.9, 0.95, 0.1]])
    out = M.ensure_at_least_one(pred, prob)
    assert out.tolist() == [[0, 1, 0], [1, 0, 0]]
    assert pred.tolist() == [[0, 0, 0], [1, 0, 0]]  # input untouched


def test_evaluate_perfect_and_matches_sklearn():
    y = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    m = M.evaluate(y, y.astype(float), 0.5)
    assert m["micro_f1"] == m["macro_f1"] == m["samples_f1"] == 1.0
    assert m["hamming_loss"] == 0.0
    assert m["support"] == [2, 2, 1]

    prob = np.array([[0.8, 0.6, 0.2], [0.1, 0.7, 0.1], [0.4, 0.3, 0.9]])
    m = M.evaluate(y, prob, 0.5, at_least_one=False)
    pred = (prob >= 0.5).astype(int)
    assert m["micro_f1"] == pytest.approx(f1_score(y, pred, average="micro"))
    assert m["macro_f1"] == pytest.approx(f1_score(y, pred, average="macro"))
    assert len(m["per_label_f1"]) == 3


def test_evaluate_per_label_threshold_array():
    y = np.array([[1, 0], [0, 1]])
    prob = np.array([[0.3, 0.1], [0.1, 0.8]])
    assert M.evaluate(y, prob, [0.5, 0.5], at_least_one=False)["micro_f1"] < 1.0
    assert M.evaluate(y, prob, [0.5, 0.5])["micro_f1"] == 1.0  # top-1 fill rescues row 0
    assert M.evaluate(y, prob, [0.25, 0.5])["micro_f1"] == 1.0


def test_tune_global_threshold_finds_separating_value():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=(200, 5))
    y[:, 0] = 1  # every doc has >=1 gold label, as in ILSI (else top-1 fill adds an FP)
    prob = np.where(y == 1, 0.3, 0.1) + rng.uniform(0, 0.05, size=y.shape)
    t, f1 = M.tune_global_threshold(y, prob)
    # negatives lie in [0.10, 0.15), positives in [0.30, 0.35): any t in (max_neg, 0.30] is perfect
    assert prob[y == 0].max() < t <= 0.30 and f1 == pytest.approx(1.0)


def test_tune_per_label_thresholds_and_fallback():
    rng = np.random.default_rng(42)
    n = 300
    y = np.zeros((n, 3), dtype=int)
    y[:, 0] = rng.integers(0, 2, n)
    y[:, 1] = rng.integers(0, 2, n)
    y[:3, 2] = 1                       # only 3 dev positives -> global fallback
    prob = np.zeros((n, 3))
    prob[:, 0] = np.where(y[:, 0], 0.8, 0.6)   # separable between 0.6 and 0.8
    prob[:, 1] = np.where(y[:, 1], 0.2, 0.1)   # separable between 0.1 and 0.2
    prob[:, 2] = rng.uniform(0, 1, n)
    t = M.tune_per_label_thresholds(y, prob, global_t=0.42)
    assert 0.6 < t[0] <= 0.8
    assert 0.1 < t[1] <= 0.2
    assert t[2] == 0.42


def test_bucket_report():
    buckets = {"head": ["34"], "mid": ["302"], "tail": ["420"]}
    r = M.bucket_report([1.0, 0.5, 0.0], buckets, LABELS)
    assert r == {"head": 1.0, "mid": 0.5, "tail": 0.0}


def test_full_report_and_save_outputs(tmp_path):
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=(50, 3))
    prob = np.clip(y * 0.6 + rng.uniform(0, 0.4, size=y.shape), 0, 1)
    buckets = {"head": ["34"], "mid": ["302"], "tail": ["420"]}
    metrics, thr = M.full_report(y, prob, y, prob, LABELS, buckets)
    assert set(metrics) == {"t0.5", "global", "per_label"}
    assert len(thr["per_label"]) == 3 and thr["label_list"] == LABELS
    M.save_outputs(tmp_path, prob, prob, thr, metrics)
    for f in ["dev_probs.npy", "test_probs.npy", "thresholds.json", "metrics.json"]:
        assert (tmp_path / f).exists()
    assert np.load(tmp_path / "dev_probs.npy").dtype == np.float32
    assert "buckets" in json.loads((tmp_path / "metrics.json").read_text())["global"]["test"]
    assert len(M.summary_rows("x", metrics)) == 3
