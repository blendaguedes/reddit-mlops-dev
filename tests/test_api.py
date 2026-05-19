import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import app

# Env vars minimas para o lifespan nao crashar ao tentar ler os.environ[...]
_ENV = {
    "DAGSHUB_USERNAME": "test_user",
    "DAGSHUB_TOKEN": "test_token",
    "MLFLOW_TRACKING_URI": "http://localhost:5000",
    "CHAMPION_MODEL": "logistic_regression",
}


@pytest.fixture
def mock_model():
    m = MagicMock()
    m.predict.return_value = np.array([1])
    m.predict_proba.return_value = np.array([[0.3, 0.7]])
    return m


@pytest.fixture
def mock_vectorizer():
    v = MagicMock()
    v.transform.return_value = MagicMock()
    return v


@pytest.fixture
def client(mock_model, mock_vectorizer):
    """TestClient com todas as chamadas ao MLflow/DagsHub substituidas por mocks.
    O lifespan roda normalmente mas sem conexao de rede."""
    mock_mv = MagicMock()
    mock_mv.version = "1"
    mock_mv.run_id = "fake-run-id"

    mock_mlflow_client = MagicMock()
    mock_mlflow_client.get_latest_versions.return_value = [mock_mv]

    with (
        patch.dict(os.environ, _ENV),
        patch("mlflow.set_tracking_uri"),
        patch("src.api.MlflowClient", return_value=mock_mlflow_client),
        patch("mlflow.sklearn.load_model", return_value=mock_model),
        patch(
            "mlflow.artifacts.download_artifacts", return_value="/tmp/vectorizer.joblib"
        ),
        patch("joblib.load", return_value=mock_vectorizer),
    ):
        with TestClient(app) as c:
            yield c


# ── Testes ────────────────────────────────────────────────────────────────────


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "Reddit Engagement Predictor"


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model"] == "loaded"
    assert data["vectorizer"] == "loaded"


def test_predict_retorna_estrutura_correta(client):
    response = client.post("/predict", json={"text": "This is a Reddit post"})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "model" in data


def test_predict_valores_validos(client):
    response = client.post("/predict", json={"text": "This is a Reddit post"})
    data = response.json()
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0
    assert data["model"] == "logistic_regression"


def test_predict_chama_vectorizer(client, mock_vectorizer):
    client.post("/predict", json={"text": "hello world"})
    mock_vectorizer.transform.assert_called_once_with(["hello world"])


def test_predict_sem_texto_retorna_422(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422
