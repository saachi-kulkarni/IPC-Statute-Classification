# IPC Statute Identification → BNS

Predict which **Indian Penal Code sections** apply to an Indian criminal judgment (multi-label, 100 sections, ILSI dataset). The project compares keyword, TF-IDF and BERT models, analyses why they succeed or fail, and studies what changes now that the **Bharatiya Nyaya Sanhita (BNS)** replaced the IPC on 1 July 2024. It includes a law-themed web demo.

📄 **Full report: [report/REPORT.pdf](report/REPORT.pdf)** (source: [report/REPORT.md](report/REPORT.md)) · BNS discussion: [report/bns_discussion.md](report/bns_discussion.md)

> Research/course project. **Not legal advice.**

## Results (ILSI test set; thresholds tuned on dev)

| Model | Micro-F1 | Macro-F1 | Tail-label F1 |
|---|---|---|---|
| Keyword overlap with statute text | 0.176 | 0.107 | 0.061 |
| **TF-IDF + logistic regression** | **0.691** | **0.585** | **0.463** |
| bert-base-uncased (first 512 tokens) | 0.497 | 0.417 | 0.293 |
| InLegalBERT (first 512 tokens) | 0.520 | 0.445 | 0.336 |
| InLegalBERT (first 128 + last 382 tokens) | 0.511 | 0.436 | 0.329 |
| *Diagnostic: read "section N" citations from the text* | *0.701* | *0.627* | *0.497* |

Key findings:
- **ILSI judgments usually cite their own answer.** A citation regex alone matches the best model. TF-IDF beats BERT by 17 micro-F1 points (p < 0.001), mainly because it reads the whole document and picks up those citations.
- **On judgments that cite no gold section** (19% of test), every model is weak (micro-F1 0.31–0.35), but they are inferring offences from facts (the regex gets 0.07).
- **InLegalBERT > bert-base**, most on rare sections. Keeping the end of the document did not help.
- **IPC → BNS:**
  - 84 of 100 labels map one-to-one, covering 87% of label occurrences.
  - 4 unverifiable mappings are withheld.
  - New BNS offences (organised crime, snatching, mob lynching…) have no training data.
  - Section numbers now mean different offences (BNS 304 = snatching), which breaks citation-reading models.

## Demo app

```bash
make app            # or: .venv/bin/python app/app.py
# open http://127.0.0.1:7860
```

The app:
- pastes or loads case facts and shows the 10 most likely sections with scores and their dev-tuned thresholds;
- shows each section's BNS counterpart, marked verified, withheld, split, scope-changed or repealed;
- warns when sections are already cited in the text.

It defaults to the fine-tuned InLegalBERT (Apple MPS) and falls back to TF-IDF if no BERT weights are present. It opens in a dark theme, with a light-theme toggle.

## Reproduce

Requirements: macOS/Linux with Python 3.11, about 2 GB free disk; BERT training needs a Colab GPU. Run the steps in order.

| Step | Command | What it does | Time (M1, 8 GB) |
|---|---|---|---|
| 0. Setup | `make setup` | `.venv`, pinned packages (`requirements.txt`), Jupyter kernel `ilsi` | ~5 min |
| 1. Data | `make data` | Download ILSI from Zenodo (MD5-checked) → inspect → `data/processed/*.parquet`, `label_list.json` | ~5 min |
| 1b. EDA | `make eda` | Runs `notebooks/01_eda.ipynb` → `outputs/eda/`, `label_buckets.json` | ~1 min |
| 2. Tests | `make test` | Metric and citation-parser unit tests | seconds |
| 3. Baselines | `make baselines` | Keyword, citation diagnostic, TF-IDF LR (C=1) → `outputs/baselines/` | ~35 min |
| 4a. Colab bundle | `make colab-pack` | Builds `colab_upload.zip` | ~1 min |
| 4b. BERT (Colab) | see [colab/RUN_INSTRUCTIONS.md](colab/RUN_INSTRUCTIONS.md) | 3 runs of `colab/train_bert.ipynb`; unzip results into `colab_downloads/<run>/` | ~15 min/run on A100, ~2 h on T4 |
| 5. Compare | `make compare` | Validates all outputs, builds the master table, runs `notebooks/02_analysis.ipynb` → `outputs/analysis/` | ~2 min |
| 6. BNS | `make bns` | Mapping CSVs, relabelling/coverage, zero-shot retrieval (MPS) → `outputs/bns/` | ~4 min |
| 7. Report | `make report` | Report figures, tables, `REPORT.html` and `REPORT.pdf` | ~1 min |
| 8. App | `make app` | Web demo | — |

Notes:
- The original plan's full grid (C ∈ {0.5, 1, 4, 10} plus LinearSVC) is available with `make baselines MODELS=keyword,regex_mention,tfidf_lr,tfidf_svm C_GRID=0.5,1,4,10`. It takes several hours on an 8 GB laptop; the reported run used C = 1, chosen on dev over C = 0.5.
- `make all` runs data → eda → baselines → compare → bns, assuming the Colab runs are already in `colab_downloads/`.
- The BNS zero-shot experiment against **BNS** statute texts needs `data/raw/bns_sections.csv` (columns `section_id,text`); without it, only the IPC-text calibration runs.

## Project structure

```
src/
  download_data.py   inspect_data.py   prepare_data.py   # Phase 1: data
  common.py          # paths, loaders, seeding, shared citation parser
  eda.py             # EDA plotting helpers
  metrics.py         # shared evaluation + threshold tuning (also shipped to Colab)
  baselines.py       # keyword, citation diagnostic, TF-IDF LR/SVM
  pack_colab.py      # builds colab_upload.zip
  compare.py         # validates all model outputs, master results table
  analysis.py        # Phase 5 analysis helpers (buckets, confusions, bootstrap…)
  bns_mapping.py     # curated IPC→BNS mapping (source-checked) + new BNS offences
  bns_transfer.py    # relabelling, coverage, zero-shot retrieval
  predict.py         # single-document inference for the app
  report_figures.py  build_report.py   # report figures/tables and PDF
notebooks/  01_eda.ipynb  02_analysis.ipynb
colab/      train_bert.ipynb  RUN_INSTRUCTIONS.md
app/        app.py (FastAPI)  static/ (index.html, styles.css, app.js)
tests/      test_metrics.py  test_citations.py
report/     REPORT.pdf  REPORT.md  REPORT.html  bns_discussion.md  figures/  tables/   (make report rebuilds them)
data/processed/  label_list.json  label_buckets.json  ipc_to_bns_mapping.csv  bns_new_offences.csv  (parquet files are git-ignored)
outputs/    eda/  baselines/  analysis/  bns/      (figures, metrics, tables; .npy files git-ignored)
colab_downloads/  <run_name>/ results (+ model/ for the best run)   (git-ignored)
```

## Data and sources

- **ILSI** dataset: Paul, Goyal & Ghosh, *LeSICiN*, AAAI 2022, [Zenodo 6053791](https://zenodo.org/records/6053791).
- **InLegalBERT**: Paul, Mandal, Goyal & Ghosh, ICAIL 2023, [law-ai/InLegalBERT](https://huggingface.co/law-ai/InLegalBERT).
- **IPC → BNS** correspondences were checked against a [table compiled at CAPT Bhopal](https://www.keralaprisons.gov.in/userfiles/act-and-rules/comparison_summary_BNS_to_IPC.pdf). This is a secondary source; an official MHA table was not accessible.
