"""Shared paths, loading helpers and seeding."""
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def load_split(split, proc_dir=PROC, with_text=True):
    cols = ["id", "text", "labels"] if with_text else ["id", "labels"]
    df = pd.read_parquet(Path(proc_dir) / f"{split}.parquet", columns=cols)
    df["labels"] = df["labels"].map(list)
    return df


def load_label_list(proc_dir=PROC):
    with open(Path(proc_dir) / "label_list.json") as f:
        return json.load(f)


def load_buckets(proc_dir=PROC):
    with open(Path(proc_dir) / "label_buckets.json") as f:
        return json.load(f)


def load_statutes(proc_dir=PROC):
    return pd.read_parquet(Path(proc_dir) / "statutes.parquet")


# ---------------------------------------------------------------- citation parsing
# Shared by baselines.regex_mention, analysis (citation position) and the demo app.
# Handles lists ("Sections 147, 148 and 149 IPC", "302/34", "302 read with 34"),
# "u/s 498-A", "Section302" (no space) and "420 IPC" / "420 I.P.C."; a letter suffix is
# only taken when attached or hyphenated ("498A", "120-B"), never from "471 I.P.C.".
_SEC_LIST = re.compile(
    r"(?:\bsec(?:tion)?s?(?![a-z])\.?|\bu/ss?\.?)\s*"
    r"((?:\d+(?:\s*-\s*)?[a-z]?(?![a-z])(?:\s*(?:,|/|&|and|r/w|read\s+with|or)\s*)?)+)",
    re.IGNORECASE)
_NUM_IPC = re.compile(r"\b(\d+(?:\s*-\s*)?[a-z]?)\s+(?:of\s+the\s+)?(?:IPC|I\.\s*P\.\s*C)", re.IGNORECASE)
_NUM = re.compile(r"(\d+)(?:\s*-\s*)?([a-z](?![a-z.])|)", re.IGNORECASE)


def citation_matches(text):
    """[(char_start, {sections})] for every citation span, in text order."""
    out = []
    for m in list(_SEC_LIST.finditer(text)) + list(_NUM_IPC.finditer(text)):
        secs = {n + s.upper() for n, s in _NUM.findall(m.group(1))}
        if secs:
            out.append((m.start(), secs))
    return sorted(out, key=lambda x: x[0])


def cited_sections(text):
    return set().union(*[s for _, s in citation_matches(text)]) if text else set()


def first_citation_word(text, targets):
    """Word index of the first citation span naming any of `targets` (None if never named)."""
    for start, secs in citation_matches(text):
        if secs & targets:
            return len(text[:start].split())
    return None
