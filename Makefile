# IPC statute identification (ILSI) -> BNS. Run targets in order; BERT training happens in Colab.
PY      := .venv/bin/python
JUPYTER := .venv/bin/jupyter
NBEXEC  := $(JUPYTER) nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=ilsi --ExecutePreprocessor.timeout=3600

# Baselines as actually run for the report: C=1 only (chosen on dev from {0.5, 1}), no SVM.
# Full grid from the original plan: make baselines MODELS=keyword,regex_mention,tfidf_lr,tfidf_svm C_GRID=0.5,1,4,10
MODELS  ?= keyword,regex_mention,tfidf_lr
C_GRID  ?= 1
N_JOBS  ?= 4

.PHONY: help setup data eda baselines colab-pack compare bns report app test all

help:
	@echo "make setup | data | eda | baselines | colab-pack | (Colab runs) | compare | bns | report | app | test"

setup:                      ## create .venv (Python 3.11), install pinned deps, register Jupyter kernel
	python3.11 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m ipykernel install --user --name ilsi --display-name "Python (ilsi)"

data:                       ## download ILSI from Zenodo (MD5-checked), inspect, build parquet + label list
	$(PY) src/download_data.py
	$(PY) src/inspect_data.py
	$(PY) src/prepare_data.py

eda:                        ## execute the EDA notebook (writes outputs/eda/ and label_buckets.json)
	$(NBEXEC) notebooks/01_eda.ipynb

baselines:                  ## keyword, regex diagnostic and TF-IDF LR (~35 min on an 8 GB M1)
	$(PY) src/baselines.py --models $(MODELS) --c_grid $(C_GRID) --n_jobs $(N_JOBS)

colab-pack:                 ## build colab_upload.zip for the Colab notebook
	$(PY) src/pack_colab.py

compare:                    ## validate colab_downloads/ + baselines, master table, analysis notebook
	$(PY) src/compare.py
	$(NBEXEC) notebooks/02_analysis.ipynb

bns:                        ## IPC->BNS mapping CSVs, relabelling/coverage and zero-shot retrieval
	$(PY) src/bns_mapping.py
	$(PY) src/bns_transfer.py

report:                     ## regenerate report/figures, report/tables, REPORT.html and REPORT.pdf (needs Chrome)
	$(PY) src/report_figures.py
	$(PY) src/build_report.py

app:                        ## web demo on http://127.0.0.1:7860
	$(PY) app/app.py

test:
	$(PY) -m pytest -q

all: data eda baselines compare bns report   ## everything after setup, assuming Colab runs are in colab_downloads/
