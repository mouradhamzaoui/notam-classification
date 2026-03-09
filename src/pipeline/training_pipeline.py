"""
training_pipeline.py
Pipeline d'entraînement Production-Ready.
Orchestre : chargement → features → entraînement → évaluation → sauvegarde.
"""

from __future__ import annotations
import time
import joblib
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    f1_score, accuracy_score,
    classification_report, confusion_matrix,
)

from src.utils.config import Config
from src.utils.logger import get_logger
from src.data.data_loader import NOTAMDataLoader
from src.features.feature_engineering import NOTAMFeaturePipeline

logger = get_logger(__name__)


@dataclass
class PipelineArtifacts:
    """Contient tous les artefacts produits par le pipeline d'entraînement."""
    feature_pipeline: Any
    model:            Any
    model_name:       str
    params:           dict
    metrics:          dict = field(default_factory=dict)
    classes:          list = field(default_factory=list)
    report:           str  = ""
    confusion_mat:    Any  = None
    train_time_s:     float = 0.0

    def summary(self) -> str:
        m = self.metrics
        return (
            f"\n{'═'*55}\n"
            f"  ✅ PIPELINE COMPLETE — {self.model_name}\n"
            f"{'═'*55}\n"
            f"  CV  F1-macro  : {m.get('cv_f1_mean', 0):.4f} "
            f"± {m.get('cv_f1_std', 0):.4f}\n"
            f"  Test F1-macro : {m.get('test_f1_macro', 0):.4f}\n"
            f"  Test Accuracy : {m.get('test_accuracy', 0):.4f}\n"
            f"  Train time    : {self.train_time_s:.1f}s\n"
            f"{'═'*55}"
        )


class TrainingPipeline:
    """
    Pipeline d'entraînement complet et configurable.

    Responsabilités :
      1. Charger les données via NOTAMDataLoader
      2. Construire les features via NOTAMFeaturePipeline
      3. Entraîner + CV + GridSearch
      4. Évaluer sur le test set
      5. Persister les artefacts (model + pipeline)

    Usage :
        pipeline = TrainingPipeline()
        artifacts = pipeline.run()
    """

    def __init__(self, cfg: Config | None = None):
        self.cfg       = cfg or Config.get()
        self.artifacts: PipelineArtifacts | None = None
        Path(self.cfg.model.artifacts_dir).mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, tune: bool = True) -> PipelineArtifacts:
        """
        Exécute le pipeline complet.
        Args:
            tune: Si True, lance GridSearchCV pour optimiser C.
        """
        logger.info(f"[Pipeline] Starting training pipeline | tune={tune}")

        # 1. Data
        X_train, X_test, y_train, y_test = self._load_data()

        # 2. Features
        feat_pipeline, X_tr, X_te, y_tr, y_te = self._build_features(
            X_train, X_test, y_train, y_test
        )

        # 3. Train
        model, model_name, params, train_time = self._train(X_tr, y_tr, tune=tune)

        # 4. Evaluate
        metrics, report, cm = self._evaluate(model, X_te, y_te, feat_pipeline)

        # 5. Save
        self.artifacts = PipelineArtifacts(
            feature_pipeline = feat_pipeline,
            model            = model,
            model_name       = model_name,
            params           = params,
            metrics          = metrics,
            classes          = list(feat_pipeline.label_encoder.classes_),
            report           = report,
            confusion_mat    = cm,
            train_time_s     = train_time,
        )
        self._save_artifacts()
        logger.info(self.artifacts.summary())
        return self.artifacts

    # ── Steps ─────────────────────────────────────────────────────────────────

    def _load_data(self):
        logger.info("[Step 1/5] Loading data...")
        loader = NOTAMDataLoader(
            path         = self.cfg.data.processed_path,
            label_col    = self.cfg.data.label_col,
            text_col     = self.cfg.data.text_col,
            test_size    = self.cfg.data.test_size,
            random_state = self.cfg.data.random_state,
        )
        df = loader.load()
        return loader.split(df)

    def _build_features(self, X_train, X_test, y_train, y_test):
        logger.info("[Step 2/5] Building features...")
        pipeline = NOTAMFeaturePipeline(
            max_tfidf_features=self.cfg.features.tfidf_max_features
        )
        X_tr = pipeline.fit_transform(X_train, y_train)
        X_te = pipeline.transform(X_test)
        y_tr = pipeline.encode_labels(y_train)
        y_te = pipeline.encode_labels(y_test)
        logger.info(f"[Features] Shape: {X_tr.shape} | Classes: {list(pipeline.label_encoder.classes_)}")
        return pipeline, X_tr, X_te, y_tr, y_te

    def _train(self, X_train, y_train, tune: bool = True):
        logger.info(f"[Step 3/5] Training {'+ GridSearch' if tune else '(no tuning)'}...")
        t0 = time.time()

        cfg_model = self.cfg.candidates if hasattr(self.cfg, "candidates") else {}
        C_default = float(self.cfg.model.best_C or 1.0)

        if tune:
            model, params = self._grid_search(X_train, y_train, C_default)
            model_name = f"LinearSVC_Tuned_C{params.get('estimator__C', C_default)}"
        else:
            model, params = self._build_svc(C_default)
            model.fit(X_train, y_train)
            model_name = "LinearSVC_Default"

        train_time = time.time() - t0
        logger.info(f"[Train] Done in {train_time:.1f}s | params={params}")
        return model, model_name, params, train_time

    def _grid_search(self, X_train, y_train, C_default: float):
        base = CalibratedClassifierCV(
            estimator=LinearSVC(
                multi_class="crammer_singer",
                max_iter=int(self.cfg.model.max_iter or 2000),
                random_state=42,
                class_weight="balanced",
            ),
            cv=3, method="sigmoid",
        )
        grid = GridSearchCV(
            estimator=base,
            param_grid={"estimator__C": [0.01, 0.1, 1.0, 5.0, 10.0]},
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        logger.info(f"[GridSearch] Best params: {grid.best_params_} | CV F1: {grid.best_score_:.4f}")
        return grid.best_estimator_, grid.best_params_

    def _build_svc(self, C: float):
        model = CalibratedClassifierCV(
            estimator=LinearSVC(
                C=C, multi_class="crammer_singer",
                max_iter=2000, random_state=42,
                class_weight="balanced",
            ),
            cv=3, method="sigmoid",
        )
        return model, {"C": C}

    def _evaluate(self, model, X_test, y_test, feat_pipeline):
        logger.info("[Step 4/5] Evaluating on test set...")
        y_pred    = model.predict(X_test)
        classes   = feat_pipeline.label_encoder.classes_
        f1_macro  = f1_score(y_test, y_pred, average="macro")
        accuracy  = accuracy_score(y_test, y_pred)
        report    = classification_report(y_test, y_pred, target_names=classes)
        cm        = confusion_matrix(y_test, y_pred)

        # Cross-validation sur le test final
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_test, y_test,
                                    cv=cv, scoring="f1_macro", n_jobs=-1)

        metrics = {
            "test_f1_macro":  float(f1_macro),
            "test_accuracy":  float(accuracy),
            "cv_f1_mean":     float(cv_scores.mean()),
            "cv_f1_std":      float(cv_scores.std()),
        }
        logger.info(f"[Eval] F1-macro={f1_macro:.4f} | Accuracy={accuracy:.4f}")
        return metrics, report, cm

    def _save_artifacts(self):
        logger.info("[Step 5/5] Saving artifacts...")
        base = Path(self.cfg.model.artifacts_dir)

        # Feature pipeline
        fp_path = Path("data/processed/feature_pipeline.pkl")
        self.artifacts.feature_pipeline.save(str(fp_path))

        # Model
        model_path = base / "best_model.pkl"
        joblib.dump(self.artifacts.model, model_path)
        logger.info(f"[Save] Model → {model_path}")

        # Metadata
        import json
        meta = {
            "model_name":    self.artifacts.model_name,
            "params":        str(self.artifacts.params),
            "metrics":       self.artifacts.metrics,
            "classes":       self.artifacts.classes,
            "train_time_s":  self.artifacts.train_time_s,
        }
        meta_path = base / "model_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"[Save] Metadata → {meta_path}")