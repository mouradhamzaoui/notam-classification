"""
data_loader.py
Responsabilité unique : charger et valider le dataset NOTAM.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ── Constantes ────────────────────────────────────────────────────────────────
RAW_PATH       = Path("data/raw/notams.csv")
PROCESSED_PATH = Path("data/processed/notams_clean.csv")

LABEL_COL   = "category"
TEXT_COL    = "body_text"
ICAO_COL    = "icao_location"
QCODE_COL   = "q_code"

CATEGORIES = [
    "RUNWAY_CLOSURE",
    "NAVIGATION_AID",
    "AIRSPACE_RESTRICTION",
    "LIGHTING",
    "OBSTACLE",
    "AERODROME_PROCEDURE",
]


class NOTAMDataLoader:
    """
    Charge, valide et expose le dataset NOTAM sous forme de DataFrame pandas.

    Usage
    -----
    >>> loader = NOTAMDataLoader()
    >>> df = loader.load()
    >>> X_train, X_test, y_train, y_test = loader.split(df)
    """

    def __init__(
        self,
        path: Path = PROCESSED_PATH,
        label_col: str = LABEL_COL,
        text_col: str = TEXT_COL,
        test_size: float = 0.20,
        random_state: int = 42,
    ):
        self.path         = Path(path)
        self.label_col    = label_col
        self.text_col     = text_col
        self.test_size    = test_size
        self.random_state = random_state

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """Charge le CSV et applique les validations de base."""
        if not self.path.exists():
            raise FileNotFoundError(
                f"Dataset introuvable : {self.path}\n"
                f"Exécute d'abord : uv run python data/download_dataset.py"
            )

        df = pd.read_csv(self.path)
        df = self._validate(df)
        df = self._add_meta_features(df)
        print(f"[DataLoader] ✅ Loaded {len(df):,} rows | {df[self.label_col].nunique()} classes")
        return df

    def split(self, df: pd.DataFrame):
        """
        Stratified train/test split.

        Stratification → garantit la même proportion de chaque classe
        dans train ET test, essentiel pour des métriques fiables.
        """
        from sklearn.model_selection import train_test_split

        X = df.drop(columns=[self.label_col])
        y = df[self.label_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )
        print(
            f"[DataLoader] Split → Train: {len(X_train):,} | Test: {len(X_test):,} "
            f"(stratified, seed={self.random_state})"
        )
        return X_train, X_test, y_train, y_test

    # ── Private helpers ───────────────────────────────────────────────────────

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vérifie les colonnes requises et supprime les doublons/NaN."""
        required = [self.text_col, self.label_col]
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes : {missing_cols}")

        before = len(df)
        df = df.dropna(subset=required).drop_duplicates(subset=[self.text_col])
        after = len(df)
        if before != after:
            print(f"[DataLoader] ⚠️  Dropped {before - after} invalid rows")

        unknown = set(df[self.label_col].unique()) - set(CATEGORIES)
        if unknown:
            print(f"[DataLoader] ⚠️  Unknown labels found: {unknown}")

        return df.reset_index(drop=True)

    def _add_meta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute des features numériques dérivées du texte brut."""
        t = df[self.text_col].astype(str)
        df["char_count"]      = t.str.len()
        df["word_count"]      = t.str.split().str.len()
        df["upper_ratio"]     = t.apply(lambda s: sum(c.isupper() for c in s) / max(len(s), 1))
        df["digit_ratio"]     = t.apply(lambda s: sum(c.isdigit() for c in s) / max(len(s), 1))
        df["slash_count"]     = t.str.count(r"/")
        df["has_time_pattern"]= t.str.contains(r"\b\d{4}Z?\b", regex=True).astype(int)
        df["has_coordinates"] = t.str.contains(r"\d{4}[NS]\d{5}[EW]", regex=True).astype(int)
        df["q_prefix"]        = df[QCODE_COL].str[:2] if QCODE_COL in df.columns else "QX"
        return df