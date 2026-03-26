"""
Stage 1 — Load
Baixa o dataset do Kaggle e salva em data/raw/reddit_data.csv.
Executado pelo DVC: dvc run ou dvc repro
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.data_loader import RedditDataLoader
from src.config import RAW_DATA_PATH

def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    dataset = params["data"]["dataset"]
    print(f"Dataset: {dataset}")

    # Forca novo download ignorando cache local (DVC controla reexecucao)
    output_file = RAW_DATA_PATH / "reddit_data.csv"
    if output_file.exists():
        output_file.unlink()

    loader = RedditDataLoader(RAW_DATA_PATH)
    df = loader.load_reddit_data()
    print(f"Stage load concluido: {len(df):,} linhas salvas em {output_file}")

if __name__ == "__main__":
    main()
