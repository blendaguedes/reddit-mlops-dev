PYTHON = .venv/Scripts/python.exe
DVC    = $(PYTHON) -m dvc
PIP    = $(PYTHON) -m pip

.PHONY: help setup pipeline load preprocess train metrics dag status mlflow jupyter clean clean-all

# Alvo padrao
help:
	@echo ""
	@echo "Reddit MLOps - DEV"
	@echo "=================="
	@echo ""
	@echo "Setup"
	@echo "  make setup          Instala dependencias no .venv"
	@echo ""
	@echo "Pipeline (DVC)"
	@echo "  make pipeline       Roda todo o pipeline (load+preprocess+train)"
	@echo "  make load           Roda apenas a stage load"
	@echo "  make preprocess     Roda apenas a stage preprocess"
	@echo "  make train          Roda apenas a stage train"
	@echo ""
	@echo "Monitoramento"
	@echo "  make status         Mostra status do pipeline DVC"
	@echo "  make dag            Mostra o grafo do pipeline"
	@echo "  make metrics        Mostra metricas do ultimo run"
	@echo "  make mlflow         Inicia o MLflow UI (localhost:5000)"
	@echo "  make jupyter        Inicia o Jupyter Lab"
	@echo ""
	@echo "Limpeza"
	@echo "  make clean          Remove artefatos processados e modelos"
	@echo "  make clean-all      Remove tudo incluindo dados brutos"
	@echo ""

# ─── Setup ────────────────────────────────────────────────────────────────────

setup:
	$(PIP) install -r requirements.txt

# ─── Pipeline ─────────────────────────────────────────────────────────────────

pipeline:
	$(DVC) repro

load:
	$(DVC) repro load

preprocess:
	$(DVC) repro preprocess

train:
	$(DVC) repro train

# ─── Monitoramento ────────────────────────────────────────────────────────────

status:
	$(DVC) status

dag:
	$(DVC) dag

metrics:
	$(DVC) metrics show

mlflow:
	$(PYTHON) -m mlflow ui --host 0.0.0.0 --port 5000

jupyter:
	$(PYTHON) -m jupyter lab --notebook-dir=notebooks

# ─── Limpeza ──────────────────────────────────────────────────────────────────

clean:
	rm -f data/processed/df_processed.csv
	rm -f data/processed/X.npz
	rm -f data/processed/y.npy
	rm -f data/processed/vectorizer.joblib
	rm -f models/logistic_regression.joblib
	rm -f models/naive_bayes.joblib
	rm -f models/random_forest.joblib
	rm -f metrics/scores.json
	@echo "Artefatos processados e modelos removidos."

clean-all: clean
	rm -f data/raw/reddit_data.csv
	@echo "Dados brutos removidos. Proximo 'make pipeline' vai re-baixar do Kaggle."
