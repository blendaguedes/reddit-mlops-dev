import re
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import logging

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Preprocessar textos do Reddit e persistir artefatos em disco"""

    def __init__(self, processed_path: Path, max_features=5000, max_df=0.8, min_df=5):
        self.processed_path = Path(processed_path)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.max_features = max_features
        self.max_df = max_df
        self.min_df = min_df
        self.vectorizer = None

    # Caminhos dos artefatos
    @property
    def _df_path(self):
        return self.processed_path / "df_processed.csv"

    @property
    def _X_path(self):
        return self.processed_path / "X.npz"

    @property
    def _y_path(self):
        return self.processed_path / "y.npy"

    @property
    def _vectorizer_path(self):
        return self.processed_path / "vectorizer.joblib"

    def is_processed(self) -> bool:
        """Verifica se os artefatos processados ja existem em disco"""
        return all([
            self._df_path.exists(),
            self._X_path.exists(),
            self._y_path.exists(),
            self._vectorizer_path.exists(),
        ])

    @staticmethod
    def clean_text(text):
        """Limpar texto"""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = ' '.join(text.split())
        return text

    def run(self, df: pd.DataFrame, text_column='body'):
        """
        Executa o preprocessing completo e salva todos os artefatos em disco.
        Se os artefatos ja existem, carrega do disco sem reprocessar.

        Returns:
            df_processed, X, y
        """
        if self.is_processed():
            print("Artefatos de preprocessing encontrados. Carregando do disco...")
            return self.load()

        print(f"Preprocessando textos ({len(df)} linhas)...")

        # Limpar textos
        df['cleaned_text'] = df[text_column].fillna('').apply(self.clean_text)
        df = df[df['cleaned_text'].str.len() > 0].reset_index(drop=True)
        print(f"{len(df)} linhas apos limpeza")

        # Fit e transform TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            max_df=self.max_df,
            min_df=self.min_df,
            stop_words='english'
        )
        X = self.vectorizer.fit_transform(df['cleaned_text'])
        print(f"Vectorizer fitted ({X.shape[1]} features), matriz: {X.shape}")

        # Criar target (score acima da mediana = 1)
        if 'score' in df.columns:
            median_score = df['score'].median()
            y = (df['score'] > median_score).astype(int).values
        else:
            rng = np.random.default_rng(42)
            y = rng.integers(0, 2, len(df))

        print(f"Target distribution: {np.bincount(y)}")

        # Salvar tudo em disco
        self.save(df, X, y)

        return df, X, y

    def save(self, df: pd.DataFrame, X, y: np.ndarray):
        """Salvar todos os artefatos em disco"""
        df.to_csv(self._df_path, index=False)
        print(f"Salvo: {self._df_path}")

        save_npz(self._X_path, X)
        print(f"Salvo: {self._X_path}")

        np.save(self._y_path, y)
        print(f"Salvo: {self._y_path}")

        joblib.dump(self.vectorizer, self._vectorizer_path)
        print(f"Salvo: {self._vectorizer_path}")

    def load(self):
        """Carregar artefatos processados do disco"""
        df = pd.read_csv(self._df_path, low_memory=False)
        X = load_npz(self._X_path)
        y = np.load(self._y_path)
        self.vectorizer = joblib.load(self._vectorizer_path)
        print(f"Carregado: df={df.shape}, X={X.shape}, y={y.shape}")
        return df, X, y
