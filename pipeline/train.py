"""
Stage 3 — Train
Le artefatos de data/processed/, treina os modelos e envia tudo para o DagsHub via MLflow:
  - modelos registrados no MLflow Model Registry
  - vectorizer logado como artifact em cada run
  - metrics/scores.json (unico artefato local, rastreado pelo DVC)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
import tempfile

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import yaml
from dotenv import load_dotenv
from scipy.sparse import load_npz
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from src.config import MODEL_NAME, PROCESSED_DATA_PATH

load_dotenv()

METRICS_PATH = Path("metrics")

MODEL_CLASSES = {
    "logistic_regression": LogisticRegression,
    "naive_bayes": MultinomialNB,
    "random_forest": RandomForestClassifier,
}


def build_models(models_params: dict) -> dict:
    return {
        name: MODEL_CLASSES[name](**params) for name, params in models_params.items()
    }


def train_and_log(X, y, vectorizer, models, test_size, experiment_name):
    mlflow.set_experiment(experiment_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"Split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

    # Salvar vectorizer em arquivo temporario para poder logar como artifact
    tmp_dir = tempfile.mkdtemp()
    vec_path = os.path.join(tmp_dir, "vectorizer.joblib")
    joblib.dump(vectorizer, vec_path)

    results = {}
    for model_name, model in models.items():
        print(f"\nTreinando {model_name}...")
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            auc = (
                roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                if hasattr(model, "predict_proba")
                else 0.0
            )

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "auc": auc,
            }

            mlflow.log_param("model_type", model_name)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                model,
                artifact_path=model_name,
                registered_model_name=model_name,
            )
            # Vectorizer junto com cada run para a API conseguir baixar pelo run_id
            mlflow.log_artifact(vec_path)

            run_id = mlflow.active_run().info.run_id
            print(
                f"  Run ID: {run_id} | F1={metrics['f1']:.4f} AUC={metrics['auc']:.4f}"
            )
            results[model_name] = {"metrics": metrics, "run_id": run_id}

    return results


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_USERNAME"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    test_size = params["train"]["test_size"]
    models = build_models(params["train"]["models"])

    print(f"Modelos configurados: {list(models.keys())}")
    print(f"Carregando artefatos de: {PROCESSED_DATA_PATH}")
    X = load_npz(PROCESSED_DATA_PATH / "X.npz")
    y = np.load(PROCESSED_DATA_PATH / "y.npy")
    vectorizer = joblib.load(PROCESSED_DATA_PATH / "vectorizer.joblib")
    print(f"X={X.shape}, y={y.shape}")

    results = train_and_log(X, y, vectorizer, models, test_size, MODEL_NAME)

    METRICS_PATH.mkdir(exist_ok=True)
    scores = {
        name: {k: round(v, 4) for k, v in r["metrics"].items()}
        for name, r in results.items()
    }
    with open(METRICS_PATH / "scores.json", "w") as f:
        json.dump(scores, f, indent=2)

    print(f"\nStage train concluido. Scores: {scores}")


if __name__ == "__main__":
    main()
