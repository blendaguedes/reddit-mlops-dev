import logging
from pathlib import Path
import numpy as np

from src.config import *
from src.data_loader import RedditDataLoader
from src.preprocessor import TextPreprocessor
from src.model_trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Pipeline completo: Load -> Preprocess -> Train (cada etapa persiste em disco)"""

    print("="*60)
    print("REDDIT SENTIMENT CLASSIFICATION - DEV VERSION")
    print("="*60)
    print(f"Environment: {ENV}")
    print(f"Debug: {DEBUG}\n")

    # ===== FASE 1: CARREGAR DADOS =====
    # Saida: data/raw/reddit_data.csv
    print("\nFASE 1: LOADING DATA")
    print("-"*60)
    loader = RedditDataLoader(RAW_DATA_PATH)
    df = loader.load_reddit_data()
    loader.explore_data(df)

    # ===== FASE 2: PREPROCESSING =====
    # Entrada: df em memoria
    # Saida: data/processed/df_processed.csv, X.npz, y.npy, vectorizer.joblib
    print("\nFASE 2: PREPROCESSING")
    print("-"*60)
    preprocessor = TextPreprocessor(
        processed_path=PROCESSED_DATA_PATH,
        max_features=MAX_FEATURES,
        max_df=MAX_DF,
        min_df=MIN_DF
    )
    df_processed, X, y = preprocessor.run(df, text_column='body')
    del df

    # ===== FASE 3: TRAINING =====
    # Entrada: data/processed/X.npz, y.npy
    # Saida: models/*.joblib
    print("\nFASE 3: TRAINING MODELS")
    print("-"*60)
    trainer = ModelTrainer(MODELS_PATH)
    results = trainer.train_models(X, y, experiment_name=MODEL_NAME)
    del X, y

    # ===== RESULTADOS =====
    print("\n" + "="*60)
    print("RESULTADOS FINAIS")
    print("="*60)
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        for metric, value in result['metrics'].items():
            print(f"  {metric}: {value:.4f}")
        print(f"  Path: {result['path']}")
        print(f"  MLFlow Run ID: {result['run_id']}")

    print("\n" + "="*60)
    print("PIPELINE COMPLETO!")
    print("="*60)
    print(f"\nArtefatos salvos:")
    print(f"  Raw data : {RAW_DATA_PATH}/reddit_data.csv")
    print(f"  Processed: {PROCESSED_DATA_PATH}/")
    print(f"  Models   : {MODELS_PATH}/")
    print(f"\nMLFlow: execute 'mlflow ui' e acesse http://localhost:5000")

if __name__ == "__main__":
    main()
