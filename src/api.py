import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from mlflow import MlflowClient

from src.schemas import PredictRequest, PredictResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
vectorizer = None
champion_name = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, vectorizer, champion_name

    os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_USERNAME"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    # Permite trocar o modelo em produção sem redeploy
    # Fallback para logistic_regression pois é o modelo padrão registrado no MLflow após o primeiro treino.
    champion = os.environ.get("CHAMPION_MODEL", "logistic_regression")
    champion_name = champion

    client = MlflowClient()
    versions = client.get_latest_versions(champion)
    if not versions:
        raise RuntimeError(
            f"Nenhuma versao encontrada para o modelo '{champion}' no MLflow"
        )
    mv = versions[0]

    model = mlflow.sklearn.load_model(f"models:/{champion}/{mv.version}")

    vec_local = mlflow.artifacts.download_artifacts(
        run_id=mv.run_id, artifact_path="vectorizer.joblib"
    )
    vectorizer = joblib.load(vec_local)

    logger.info(
        f"Modelo '{champion}' v{mv.version} (run {mv.run_id}) carregado do DagsHub"
    )
    yield
    logger.info("Encerrando API")


app = FastAPI(
    title="Reddit Engagement Predictor API",
    description="Classifica posts do Reddit como alto/baixo engajamento",
    version="1.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "model": "loaded" if model else "not_loaded",
        "vectorizer": "loaded" if vectorizer else "not_loaded",
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest) -> PredictResponse:
    try:
        X = vectorizer.transform([request.text])
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]

        logger.info(
            json.dumps(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "text_length": len(request.text),
                    "prediction": int(pred),
                    "probability": float(prob),
                }
            )
        )

        return PredictResponse(
            prediction=int(pred), probability=float(prob), model=champion_name
        )

    except Exception as e:
        logger.error(f"Erro na predicao: {e}")
        raise


@app.get("/", tags=["Info"])
def root():
    return {"name": "Reddit Engagement Predictor", "version": "1.0", "docs": "/docs"}
