# Running the BERT experiments in Google Colab

Everything runs from one notebook, `colab/train_bert.ipynb`. You change two config lines between runs.

## 0. One-time upload

| What | From (Mac) | To (Google Drive) |
|---|---|---|
| Data bundle (≈210 MB) | `colab_upload.zip` (project root) | `MyDrive/ilsi/colab_upload.zip` |
| Notebook | `colab/train_bert.ipynb` | Open it in Colab: colab.research.google.com → *File → Upload notebook* (or put it anywhere in Drive and *Open with → Colaboratory*) |

Then in Colab: *Runtime → Change runtime type → T4 GPU* (or L4/A100 if you have them).

If `colab_upload.zip` is missing, rebuild it on the Mac with `make colab-pack` (or `.venv/bin/python src/pack_colab.py`).

## 1. Runs, in order

Edit only the **config cell** (first code cell), then *Runtime → Run all*.

| # | Purpose | `MODEL_NAME` | `TRUNCATION` | `SMOKE_TEST` | Run name (folder on Drive) |
|---|---|---|---|---|---|
| 1 | Smoke test (≈5 min) | `"law-ai/InLegalBERT"` | `"head"` | `True` | `inlegalbert_head_128_smoke` |
| 2 | Legal-domain BERT | `"law-ai/InLegalBERT"` | `"head"` | `False` | `inlegalbert_head_512` |
| 3 | General BERT | `"bert-base-uncased"` | `"head"` | `False` | `bert-base-uncased_head_512` |
| 4 | Best-model variant | `"law-ai/InLegalBERT"` | `"head_tail"` | `False` | `inlegalbert_head_tail_512` |

Leave everything else at its default (MAX_LEN 512, 4 epochs, LR 2e-5, batch 16 × grad-accum 2, pos_weight clipped at 30, seed 42).

* **Smoke test:** succeeds if the last cell prints a metrics table and `DOWNLOAD: ...` (the scores will be low; that's expected with 500 docs). You don't need to download anything from it.
* **Full runs:** each takes roughly 1.5–2.5 h on a T4 (estimate). You can run 2–4 back to back in one session or across days.
* **Disconnects:** reconnect, keep the same config, *Run all*. Tokenization is cached in `MyDrive/ilsi/tok_cache/` and training resumes from the last epoch checkpoint in `MyDrive/ilsi/checkpoints/<run_name>/`.
* **Drive space:** each run keeps up to 2 checkpoints (~1.3 GB each). After you've downloaded a run's results, you can delete `MyDrive/ilsi/checkpoints/<run_name>/`.

## 2. What to download

For **every full run** (2, 3, 4):

* `MyDrive/ilsi/runs/<run_name>/<run_name>_results.zip` (a few MB). It contains `dev_probs.npy`, `test_probs.npy`, `thresholds.json`, `metrics.json`, `train_log.csv`, `config.json` and `split_ids.json`.

For **the best run only**, also download the model weights:

* `MyDrive/ilsi/runs/<run_name>/model/` (≈420 MB; Drive zips the folder when you download it).
* Pick "best" by the line `DEV micro-F1 at global threshold = ...` that each run prints. Use the **dev** score, not test. If unsure, take `inlegalbert_head_tail_512`.

## 3. Where to put the files on the Mac

```bash
cd "/Users/shriyansnayak/IPC statute classification project"
mkdir -p colab_downloads
# results zips: each zip already contains a <run_name>/ folder
unzip -o ~/Downloads/inlegalbert_head_512_results.zip       -d colab_downloads/
unzip -o ~/Downloads/bert-base-uncased_head_512_results.zip -d colab_downloads/
unzip -o ~/Downloads/inlegalbert_head_tail_512_results.zip  -d colab_downloads/
# model weights of the best run (Drive's download zip is named model-*.zip or similar)
mkdir -p colab_downloads/<best_run_name>/model
unzip -o ~/Downloads/model-*.zip -d colab_downloads/<best_run_name>/model/
```

The result should look like this:

```
colab_downloads/
  inlegalbert_head_512/          dev_probs.npy test_probs.npy thresholds.json metrics.json train_log.csv config.json split_ids.json
  bert-base-uncased_head_512/    (same files)
  inlegalbert_head_tail_512/     (same files)
  <best_run_name>/model/         config.json model.safetensors tokenizer.json tokenizer_config.json vocab.txt ...
```

If Drive's zip of `model/` unpacks into an extra nested `model/model/` folder, move the files up one level so that `colab_downloads/<best_run_name>/model/config.json` exists. Phase 5 validation (`src/compare.py`) checks the shapes, label order and test IDs of everything here.

Then tell Claude: **"Colab runs are downloaded."**
