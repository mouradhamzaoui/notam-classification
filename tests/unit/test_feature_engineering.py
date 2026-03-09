"""
Tests unitaires — Feature Engineering Pipeline
"""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse

from src.features.feature_engineering import (
    MetaFeatureExtractor,
    NOTAMFeaturePipeline,
    NOTAMTextPreprocessor,
    TFIDFStrategy,
)


@pytest.mark.unit
class TestNOTAMTextPreprocessor:
    def test_lowercasing(self):
        prep = NOTAMTextPreprocessor(use_stemming=False)
        result = prep._preprocess("RWY 28L CLSD")
        assert result == result.lower()

    def test_digits_removed(self):
        prep = NOTAMTextPreprocessor(use_stemming=False)
        result = prep._preprocess("RWY 28L CLSD 1200FT")
        assert "28" not in result
        assert "1200" not in result

    def test_stemming_reduces_tokens(self):
        prep_no = NOTAMTextPreprocessor(use_stemming=False)
        prep_yes = NOTAMTextPreprocessor(use_stemming=True)
        text = "RUNWAY CLOSED CLOSING CLOSURE"
        res_no = prep_no._preprocess(text)
        res_yes = prep_yes._preprocess(text)
        # Le stemming doit produire moins ou autant de tokens uniques
        assert len(set(res_yes.split())) <= len(set(res_no.split()))

    def test_transform_series(self, sample_df):
        prep = NOTAMTextPreprocessor()
        result = prep.fit_transform(sample_df)
        assert len(result) == len(sample_df)
        assert all(isinstance(t, str) for t in result)

    def test_empty_string_handled(self):
        prep = NOTAMTextPreprocessor()
        result = prep._preprocess("")
        assert isinstance(result, str)


@pytest.mark.unit
class TestTFIDFStrategy:
    def test_fit_transform_shape(self, sample_df):
        texts = sample_df["body_text"].astype(str)
        tfidf = TFIDFStrategy(max_features=200)
        X = tfidf.fit_transform(texts)
        assert X.shape[0] == len(sample_df)
        assert X.shape[1] <= 200

    def test_output_is_sparse(self, sample_df):
        texts = sample_df["body_text"].astype(str)
        tfidf = TFIDFStrategy(max_features=100)
        X = tfidf.fit_transform(texts)
        assert issparse(X)

    def test_transform_consistent_features(self, sample_df):
        """Train vs test doivent avoir le même nombre de features."""
        texts = sample_df["body_text"].astype(str)
        tfidf = TFIDFStrategy(max_features=100)
        X_tr = tfidf.fit_transform(texts[:200])
        X_te = tfidf.transform(texts[200:])
        assert X_tr.shape[1] == X_te.shape[1]

    def test_feature_names_available(self, sample_df):
        texts = sample_df["body_text"].astype(str)
        tfidf = TFIDFStrategy(max_features=50)
        tfidf.fit_transform(texts)
        names = tfidf.feature_names
        assert len(names) <= 50
        assert all(isinstance(n, str) for n in names)


@pytest.mark.unit
class TestMetaFeatureExtractor:
    def test_output_shape(self, sample_df):
        meta = MetaFeatureExtractor()
        X = meta.fit_transform(sample_df)
        assert X.shape[0] == len(sample_df)
        assert X.shape[1] == len(meta.feature_cols)

    def test_no_nan_values(self, sample_df):
        meta = MetaFeatureExtractor()
        X = meta.fit_transform(sample_df)
        assert not np.isnan(X).any()

    def test_output_is_numeric(self, sample_df):
        """Vérifie que la sortie est bien un array numérique."""
        meta = MetaFeatureExtractor()
        X = meta.fit_transform(sample_df)
        assert X.dtype in [np.float32, np.float64]

    def test_scaled_mean_near_zero(self, sample_df):
        """Après StandardScaler, la moyenne doit être proche de 0."""
        meta = MetaFeatureExtractor()
        X = meta.fit_transform(sample_df)
        col_means = X.mean(axis=0)
        assert all(abs(m) < 0.5 for m in col_means)

    def test_raw_char_count_positive(self, sample_df):
        """Vérifie les valeurs brutes AVANT scaling."""
        meta = MetaFeatureExtractor()
        raw_df = meta._extract(sample_df)
        assert all(raw_df["char_count"] >= 0)

    def test_raw_ratios_between_0_and_1(self, sample_df):
        """Vérifie les ratios bruts AVANT scaling."""
        meta = MetaFeatureExtractor()
        raw_df = meta._extract(sample_df)
        for feat in ["upper_ratio", "digit_ratio"]:
            if feat in meta.feature_cols:
                assert all(0 <= v <= 1 for v in raw_df[feat])
