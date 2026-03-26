import kagglehub
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class RedditDataLoader:
    """Carregar dataset do Reddit do Kaggle"""

    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)

    def download_dataset(self):
        """Baixar dataset do Kaggle"""
        print("Baixando dataset do Kaggle...")
        print("Isso pode levar alguns minutos na primeira vez...")

        try:
            path = kagglehub.dataset_download("pavellexyr/the-reddit-dataset-dataset")
            print(f"Dataset baixado em: {path}")
            return path
        except Exception as e:
            print(f"Erro ao baixar: {str(e)}")
            print("\nPara usar o dataset, voce precisa:")
            print("1. Criar conta em kaggle.com")
            print("2. Ir em Account -> Settings -> API")
            print("3. Baixar kaggle.json")
            print("4. Colocar em ~/.kaggle/kaggle.json")
            return None

    def load_reddit_data(self, dataset_path: str = None) -> pd.DataFrame:
        """
        Carregar dados do Reddit e salvar em data/raw/

        Args:
            dataset_path: Caminho do dataset (se None, baixa automaticamente)

        Returns:
            DataFrame com dados do Reddit
        """
        output_file = self.data_path / "reddit_data.csv"

        # Se já foi salvo localmente, carregar direto
        if output_file.exists():
            print(f"Carregando dados locais de: {output_file}")
            df = pd.read_csv(output_file, low_memory=False)
            print(f"Carregado: {len(df)} linhas")
            return df

        if dataset_path is None:
            dataset_path = self.download_dataset()

        if dataset_path is None:
            raise ValueError("Nao foi possivel carregar o dataset")

        dataset_path = Path(dataset_path)

        # Procurar por arquivos CSV
        csv_files = list(dataset_path.glob("**/*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em {dataset_path}")

        print(f"Encontrados {len(csv_files)} arquivos CSV")

        dfs = []
        for csv_file in csv_files[:3]:
            print(f"Carregando {csv_file.name}...")
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                print(f"   Carregado {len(df)} linhas")
                dfs.append(df)
            except Exception as e:
                print(f"   Erro ao carregar {csv_file.name}: {str(e)}")

        if not dfs:
            raise ValueError("Nenhum arquivo CSV foi carregado com sucesso")

        df_combined = pd.concat(dfs, ignore_index=True)
        print(f"\nTotal: {len(df_combined)} linhas")
        print(f"Colunas: {list(df_combined.columns)}")

        # Salvar em data/raw/
        df_combined.to_csv(output_file, index=False)
        print(f"Dados salvos em: {output_file}")

        return df_combined

    def explore_data(self, df: pd.DataFrame):
        """Explorar dados carregados"""
        print("\n" + "="*50)
        print("EXPLORACAO DOS DADOS")
        print("="*50)
        print(f"Shape: {df.shape}")
        print(f"\nColunas: {list(df.columns)}")
        print(f"\nTipos:\n{df.dtypes}")
        print(f"\nPrimeiras linhas:")
        print(df.head())
        print(f"\nValores nulos:\n{df.isnull().sum()}")
        print(f"\nEstatisticas:")
        print(df.describe())
