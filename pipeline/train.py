"""
Stage 3 — Train
Le artefatos de data/processed/, treina os modelos e salva:
  - models/logistic_regression.joblib
  - models/naive_bayes.joblib
  - models/random_forest.joblib
  - metrics/scores.json
Executado pelo DVC: dvc run ou dvc repro
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import yaml
import numpy as np
from scipy.sparse import load_npz
from src.model_trainer import ModelTrainer
from src.config import PROCESSED_DATA_PATH, MODELS_PATH, MODEL_NAME

METRICS_PATH = Path("metrics")

def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    test_size    = params["train"]["test_size"]
    random_state = params["train"]["random_state"]

    print(f"Carregando artefatos processados de: {PROCESSED_DATA_PATH}")
    X = load_npz(PROCESSED_DATA_PATH / "X.npz")
    y = np.load(PROCESSED_DATA_PATH / "y.npy")
    print(f"X={X.shape}, y={y.shape}")

    trainer = ModelTrainer(MODELS_PATH)
    results = trainer.train_models(
        X, y,
        experiment_name=MODEL_NAME,
        test_size=test_size,
    )

    # Salvar metricas em metrics/scores.json (rastreado pelo DVC como metrica)
    METRICS_PATH.mkdir(exist_ok=True)
    scores = {
        name: {k: round(v, 4) for k, v in r["metrics"].items()}
        for name, r in results.items()
    }
    metrics_file = METRICS_PATH / "scores.json"
    with open(metrics_file, "w") as f:
        json.dump(scores, f, indent=2)

    print(f"Stage train concluido. Metricas salvas em: {metrics_file}")
    for name, m in scores.items():
        print(f"  {name}: F1={m['f1']} AUC={m['auc']}")

if __name__ == "__main__":
    main()
