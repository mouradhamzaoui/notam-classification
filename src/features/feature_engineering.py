"""
feature_engineering.py

4 Stratégies de Feature Engineering pour la classification NOTAM :

  Strategy 1 — TF-IDF Vectorization (Baseline)
  Strategy 2 — TF-IDF + Meta Features (Hybrid Numeric)
  Strategy 3 — Label Encoding pour les features catégorielles
  Strategy 4 — Pipeline Sklearn complet prêt pour la modélisation

Justifications mathématiques incluses dans les docstrings.
"""

import re
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import hstack, issparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Constantes ────────────────────────────────────────────────────────────────
AVIATION_STOPWORDS = set(stopwords.words("english")) | {
    "rwy",
    "twy",
    "apt",
    "acft",
    "notam",
    "notamn",
    "avbl",
    "effective",
    "clsd",
    "due",
    "will",
    "not",
    "auth",
    "ots",
    "opr",
    "ops",
    "flt",
    "info",
    "ppr",
    "req",
}

META_FEATURES = [
    "char_count",
    "word_count",
    "upper_ratio",
    "digit_ratio",
    "slash_count",
    "has_time_pattern",
    "has_coordinates",
]


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1 — TF-IDF (Term Frequency × Inverse Document Frequency)
# ══════════════════════════════════════════════════════════════════════════════
class TFIDFStrategy:
    """
    Baseline : représentation bag-of-n-grams pondérée par TF-IDF.

    Formule :
        TF(t, d)   = (occurrences de t dans d) / (total mots dans d)
        IDF(t, D)  = log( (1 + |D|) / (1 + df(t)) ) + 1     [smooth]
        TF-IDF(t,d) = TF(t,d) × IDF(t,D)

    Pourquoi c'est adapté aux NOTAMs ?
    - Les NOTAMs sont courts et techniques → les termes rares sont très
      discriminants (ex: "PAPI" n'apparaît que dans LIGHTING).
    - L'IDF pénalise les mots ubiquitaires ("ACFT", "RWY") et booste
      les termes spécifiques à chaque classe.
    - Les bi-grammes capturent "RWY CLSD" ou "ILS CAT" comme unités sémantiques.

    Paramètres clés :
        max_features  : vocabulaire limité pour éviter l'overfitting
        sublinear_tf  : applique log(1+TF) → réduit l'effet des répétitions
        ngram_range   : (1,2) = unigrammes + bigrammes
    """

    def __init__(self, max_features: int = 5000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,  # log-normalization TF
            strip_accents="unicode",
            analyzer="word",
            stop_words=list(AVIATION_STOPWORDS),
            min_df=2,  # ignore les hapax
            max_df=0.95,  # ignore les quasi-ubiquitaires
            norm="l2",  # normalisation L2 par doc
        )

    def fit(self, X_text: pd.Series):
        self.vectorizer.fit(X_text)
        return self

    def transform(self, X_text: pd.Series):
        return self.vectorizer.transform(X_text)

    def fit_transform(self, X_text: pd.Series):
        return self.vectorizer.fit_transform(X_text)

    @property
    def feature_names(self):
        return self.vectorizer.get_feature_names_out()


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2 — Preprocessing textuel avec Stemming
# ══════════════════════════════════════════════════════════════════════════════
class NOTAMTextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Prétraitement NLP adapté au domaine aéronautique.

    Pipeline de normalisation :
        1. Lowercasing    → uniformisation de la casse
        2. Digit removal  → les chiffres (fréquences, altitudes) sont bruités
        3. Stemming       → réduit "closure/closed/closing" → "clos"
           Algorithme : Porter Stemmer (Lancaster trop agressif pour l'aviation)
        4. Token filtering → retire stopwords + tokens < 2 chars

    Justification du stemming vs lemmatisation :
        Le lemmatiseur nécessite un POS-tag correct. Les NOTAMs sont en
        "abbreviated English" (pas de POS standard) → le stemmer est plus
        robuste sur ce corpus technique.
    """

    def __init__(self, use_stemming: bool = True):
        self.use_stemming = use_stemming
        self.stemmer = PorterStemmer()

    def _preprocess(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"\b\d+[\w./]*", "", text)  # supprime nombres/codes
        text = re.sub(r"[^a-z\s]", " ", text)  # garde lettres seulement
        tokens = text.split()
        tokens = [t for t in tokens if t not in AVIATION_STOPWORDS and len(t) > 2]
        if self.use_stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]
        return " ".join(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X["body_text"].apply(self._preprocess)
        return pd.Series(X).apply(self._preprocess)


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3 — Meta Features Numériques
# ══════════════════════════════════════════════════════════════════════════════
class MetaFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extrait et normalise les features numériques/structurelles.

    Features et leur justification :
    ┌─────────────────────┬──────────────────────────────────────────────────┐
    │ Feature             │ Justification                                    │
    ├─────────────────────┼──────────────────────────────────────────────────┤
    │ char_count          │ Les NOTAMs OBSTACLE sont plus longs (coords GPS) │
    │ word_count          │ Proxy de complexité informationnelle             │
    │ upper_ratio         │ Les NOTAMs sont en majuscules → ratio stable    │
    │ digit_ratio         │ OBSTACLE/AIRSPACE ont plus de chiffres (coords) │
    │ slash_count         │ "/" fréquent dans les Q-codes et dates ICAO     │
    │ has_time_pattern    │ Pattern HHMM présent dans restrictions temporelles│
    │ has_coordinates     │ Discriminant fort pour OBSTACLE                  │
    └─────────────────────┴──────────────────────────────────────────────────┘

    Normalisation : StandardScaler → mean=0, std=1
        Nécessaire car les classifieurs à distance (SVM, KNN) sont sensibles
        aux échelles différentes entre char_count (∈[10,200]) et
        has_coordinates (∈{0,1}).
    """

    def __init__(self, feature_cols: list = None):
        self.feature_cols = feature_cols or META_FEATURES
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y=None):
        features = self._extract(X)
        self.scaler.fit(features)
        return self

    def transform(self, X: pd.DataFrame):
        features = self._extract(X)
        return self.scaler.transform(features)

    def _extract(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing:
            # Re-calcule les features manquantes si nécessaire
            t = X["body_text"].astype(str)
            for col in missing:
                if col == "char_count":
                    X = X.copy()
                    X[col] = t.str.len()
                if col == "word_count":
                    X = X.copy()
                    X[col] = t.str.split().str.len()
                if col == "upper_ratio":
                    X = X.copy()
                    X[col] = t.apply(lambda s: sum(c.isupper() for c in s) / max(len(s), 1))
                if col == "digit_ratio":
                    X = X.copy()
                    X[col] = t.apply(lambda s: sum(c.isdigit() for c in s) / max(len(s), 1))
                if col == "slash_count":
                    X = X.copy()
                    X[col] = t.str.count("/")
                if col == "has_time_pattern":
                    X = X.copy()
                    X[col] = t.str.contains(r"\b\d{4}Z?\b").astype(int)
                if col == "has_coordinates":
                    X = X.copy()
                    X[col] = t.str.contains(r"\d{4}[NS]\d{5}[EW]").astype(int)
        return X[self.feature_cols].fillna(0)


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 4 — Pipeline Hybride Complet (TF-IDF + Meta Features)
# ══════════════════════════════════════════════════════════════════════════════
class NOTAMFeaturePipeline:
    """
    Pipeline hybride combinant :
        [TF-IDF sur texte stemmatisé] ⊕ [Meta features normalisées]

    Concaténation sparse :
        X_final = hstack([X_tfidf (sparse), X_meta (dense→sparse)])

    Dimension finale ≈ 5000 (TF-IDF) + 7 (meta) = 5007 features

    Avantage :
        Le modèle peut utiliser SIMULTANÉMENT le signal lexical (TF-IDF)
        ET le signal structurel (longueur, coords, chiffres).
        Exemple : deux NOTAMs avec même vocabulaire mais l'un avec
        coordonnées GPS → le meta feature "has_coordinates=1" oriente
        vers OBSTACLE.
    """

    def __init__(self, max_tfidf_features: int = 5000):
        self.preprocessor = NOTAMTextPreprocessor(use_stemming=True)
        self.tfidf = TFIDFStrategy(max_features=max_tfidf_features)
        self.meta = MetaFeatureExtractor()
        self.label_encoder = LabelEncoder()
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        text_clean = self.preprocessor.fit_transform(X)
        self.tfidf.fit(text_clean)
        self.meta.fit(X)
        self.label_encoder.fit(y)
        self._fitted = True
        print(
            f"[Pipeline] ✅ Fitted | TF-IDF vocab: {len(self.tfidf.feature_names):,} "
            f"| Meta features: {len(self.meta.feature_cols)} "
            f"| Classes: {list(self.label_encoder.classes_)}"
        )
        return self

    def transform(self, X: pd.DataFrame):
        if not self._fitted:
            raise RuntimeError("Pipeline not fitted. Call .fit() first.")
        text_clean = self.preprocessor.transform(X)
        X_tfidf = self.tfidf.transform(text_clean)
        X_meta = self.meta.transform(X)
        from scipy.sparse import csr_matrix

        X_meta_sp = csr_matrix(X_meta)
        return hstack([X_tfidf, X_meta_sp])

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):
        return self.fit(X, y).transform(X)

    def encode_labels(self, y: pd.Series) -> np.ndarray:
        return self.label_encoder.transform(y)

    def decode_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        return self.label_encoder.inverse_transform(y_encoded)

    @property
    def n_features(self) -> int:
        return len(self.tfidf.feature_names) + len(self.meta.feature_cols)

    def save(self, path: str = "data/processed/feature_pipeline.pkl"):
        import joblib

        Path(path).parent.mkdir(exist_ok=True)
        joblib.dump(self, path)
        print(f"[Pipeline] 💾 Saved to {path}")

    @classmethod
    def load(cls, path: str = "data/processed/feature_pipeline.pkl"):
        import joblib

        obj = joblib.load(path)
        print(f"[Pipeline] 📂 Loaded from {path}")
        return obj
