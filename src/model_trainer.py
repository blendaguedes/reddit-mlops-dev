import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Treinar modelos de classificacao"""

    def __init__(self, models_path: Path = None):
        self.models_path = Path(models_path or "./models")
        self.models_path.mkdir(parents=True, exist_ok=True)

    def create_models(self):
        """Criar diferentes modelos"""
        return {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'naive_bayes': MultinomialNB(),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        }

    def train_models(self, X, y, experiment_name="reddit_classification", test_size=0.2):
        """Treinar multiplos modelos com MLFlow"""

        mlflow.set_experiment(experiment_name)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"\nSplit: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
        print(f"Classes: {np.unique(y_train)} (Distribution: {np.bincount(y_train)})")

        results = {}
        models = self.create_models()

        for model_name, model in models.items():
            print(f"\nTreinando {model_name}...")

            with mlflow.start_run(run_name=model_name):
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_pred_proba)
                else:
                    auc = 0.0

                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'auc': auc,
                }

                mlflow.log_param("model_type", model_name)
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, model_name)

                model_path = self.models_path / f"{model_name}.joblib"
                joblib.dump(model, model_path)
                print(f"Salvo em: {model_path}")

                results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'path': model_path,
                    'run_id': mlflow.active_run().info.run_id,
                }

                print(f"Metrics: {metrics}")

        return results
