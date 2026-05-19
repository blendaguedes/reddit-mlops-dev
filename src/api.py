from fastapi import FastAPI
import joblib
import logging
import json
from contextlib import asynccontextmanager
from datetime import datetime
from src.config import MODEL_PATH, VECTORIZER_PATH, CHAMPION_MODEL
from src.schemas import PredictRequest, PredictResponse

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== FastAPI App =====
app = FastAPI(
    title='Reddit Engagement Predictor API',
    description='Classifica posts do Reddit como alto/baixo engajamento',
    version='1.0'
)

# ===== State =====
model = None
vectorizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, vectorizer
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    logger.info(f"Modelo carregado: {MODEL_PATH}")
    yield
    logger.info("Encerrando API")

app = FastAPI(
    title='Reddit Engagement Predictor API',
    description='Classifica posts do Reddit como alto/baixo engajamento',
    version='1.0',
    lifespan=lifespan
)

# ... resto do código

app = FastAPI(
    title='Reddit Engagement Predictor API',
    description='Classifica posts do Reddit como alto/baixo engajamento',
    version='1.0',
    lifespan=lifespan
)

# ===== Health Check =====
@app.get('/health', tags=['Health'])
def health_check():
    """Verifica saúde da API"""
    return {
        'status': 'ok',
        'model': 'loaded' if model else 'not_loaded',
        'vectorizer': 'loaded' if vectorizer else 'not_loaded',
        'champion_model': CHAMPION_MODEL
    }

# ===== Prediction =====
@app.post('/predict', response_model=PredictResponse, tags=['Prediction'])
async def predict(request: PredictRequest) -> PredictResponse:
    """Faz predição de engajamento para um texto"""
    
    try:
        # Vetorizar texto
        X = vectorizer.transform([request.text])
        
        # Predição
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]
        
        # Log estruturado
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'text_length': len(request.text),
            'prediction': int(pred),
            'probability': float(prob)
        }
        logger.info(json.dumps(log_entry))
        
        return PredictResponse(
            prediction=int(pred),
            probability=float(prob)
        )
    
    except Exception as e:
        logger.error(f'Erro na predição: {e}')
        raise

# ===== Root =====
@app.get('/', tags=['Info'])
def root():
    """Informação da API"""
    return {
        'name': 'Reddit Engagement Predictor',
        'version': '1.0',
        'docs': '/docs',
        'champion_model': CHAMPION_MODEL
    }