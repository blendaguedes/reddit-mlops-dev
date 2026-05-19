import os
from dotenv import load_dotenv
from pathlib import Path

# Carregar .env
load_dotenv()

# Caminhos
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
MODELS_PATH = PROJECT_ROOT / "models"
LOGS_PATH = PROJECT_ROOT / "logs"
MLFLOW_PATH = PROJECT_ROOT / "mlruns"

# Criar diretórios se não existirem
for path in [DATA_PATH, RAW_DATA_PATH, PROCESSED_DATA_PATH, MODELS_PATH, LOGS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Variáveis de ambiente
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY = os.getenv("KAGGLE_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "reddit_sentiment_classifier")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
ENV = os.getenv("ENV", "development")
DEBUG = os.getenv("DEBUG", "True") == "True"

# MLFlow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file:{MLFLOW_PATH}")
MLFLOW_ARTIFACT_LOCATION = os.getenv("MLFLOW_ARTIFACT_LOCATION", str(PROJECT_ROOT / "mlflow_artifacts"))

# Model config
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES = 5000
MAX_DF = 0.8
MIN_DF = 5

# Champion model (NOVO)
CHAMPION_MODEL = os.getenv("CHAMPION_MODEL", "logistic_regression")
MODEL_PATH = MODELS_PATH / f"{CHAMPION_MODEL}.joblib"
VECTORIZER_PATH = MODELS_PATH / "vectorizer.joblib"

print(f"Config loaded (ENV={ENV})")
print(f"Data path: {PROCESSED_DATA_PATH}")
print(f"Models path: {MODELS_PATH}")
print(f"Champion model: {CHAMPION_MODEL}")