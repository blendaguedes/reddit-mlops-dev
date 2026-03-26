"""
Stage 2 — Preprocess
Le data/raw/reddit_data.csv, limpa textos, vetoriza com TF-IDF e salva:
  - data/processed/df_processed.csv
  - data/processed/X.npz
  - data/processed/y.npy
  - data/processed/vectorizer.joblib
Executado pelo DVC: dvc run ou dvc repro
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import pandas as pd
from src.preprocessor import TextPreprocessor
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH

def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    text_column   = params["data"]["text_column"]
    max_features  = params["preprocess"]["max_features"]
    max_df        = params["preprocess"]["max_df"]
    min_df        = params["preprocess"]["min_df"]

    print(f"Carregando dados brutos de: {RAW_DATA_PATH / 'reddit_data.csv'}")
    df = pd.read_csv(RAW_DATA_PATH / "reddit_data.csv", low_memory=False)
    print(f"Linhas carregadas: {len(df):,}")

    preprocessor = TextPreprocessor(
        processed_path=PROCESSED_DATA_PATH,
        max_features=max_features,
        max_df=max_df,
        min_df=min_df,
    )

    # Forca reprocessamento: remove artefatos anteriores para que
    # is_processed() retorne False e o DVC sempre obtenha resultados frescos
    for artifact in [
        PROCESSED_DATA_PATH / "df_processed.csv",
        PROCESSED_DATA_PATH / "X.npz",
        PROCESSED_DATA_PATH / "y.npy",
        PROCESSED_DATA_PATH / "vectorizer.joblib",
    ]:
        if artifact.exists():
            artifact.unlink()

    df_processed, X, y = preprocessor.run(df, text_column=text_column)

    print(f"Stage preprocess concluido: X={X.shape}, y={y.shape}")
    print(f"Artefatos salvos em: {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    main()
