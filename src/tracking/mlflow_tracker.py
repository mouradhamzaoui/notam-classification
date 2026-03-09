"""
mlflow_tracker.py
Intégration MLflow pour le tracking des expériences NOTAM.

Logue :
  - Hyperparamètres
  - Métriques (F1, Accuracy, CV scores)
  - Artifacts (model, feature pipeline, confusion matrix)
  - Tags (dataset info, git commit, etc.)
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import matplotlib
import mlflow
import mlflow.sklearn
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLflowTracker:
    """
    Wrapper MLflow adapté au projet NOTAM.

    Gère :
      - Connexion au tracking server (local ou distant)
      - Création/récupération de l'expérience
      - Logging structuré des runs
      - Enregistrement du modèle dans le Model Registry

    Usage :
        tracker = MLflowTracker()
        with tracker.start_run("LinearSVC_v1") as run:
            tracker.log_params({"C": 1.0})
            tracker.log_metrics({"f1_macro": 0.94})
            tracker.log_model(model, "notam-classifier")
    """

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config.get()
        self._run = None
        self._connected = False
        self._setup()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup(self):
        """Configure le tracking URI et l'expérience MLflow."""
        tracking_uri = self.cfg.mlflow.tracking_uri or "http://localhost:5000"

        try:
            mlflow.set_tracking_uri(tracking_uri)
            # Test de connectivité
            mlflow.search_experiments()
            self._connected = True
            logger.info(f"[MLflow] ✅ Connected → {tracking_uri}")
        except Exception as e:
            # Fallback : tracking local dans ./mlruns
            local_uri = Path("mlruns").absolute().as_uri()
            mlflow.set_tracking_uri(local_uri)
            self._connected = False
            logger.warning(
                f"[MLflow] Server unavailable ({e})\n"
                f"         Falling back to local tracking → {local_uri}"
            )

        # Crée ou récupère l'expérience
        experiment_name = self.cfg.mlflow.experiment_name
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            mlflow.create_experiment(
                experiment_name,
                tags={"project": self.cfg.project.name, "version": self.cfg.project.version},
            )
            logger.info(f"[MLflow] Created experiment: {experiment_name}")
        mlflow.set_experiment(experiment_name)

    # ── Run management ────────────────────────────────────────────────────────

    def start_run(self, run_name: str) -> mlflow.ActiveRun:
        self._run = mlflow.start_run(run_name=run_name)
        logger.info(f"[MLflow] Run started → {self._run.info.run_id[:12]}...")
        return self._run

    def end_run(self, status: str = "FINISHED"):
        mlflow.end_run(status=status)
        logger.info(f"[MLflow] Run ended → {status}")

    # ── Logging ───────────────────────────────────────────────────────────────

    def log_params(self, params: dict):
        """Logue les hyperparamètres (flatten si nested)."""
        flat = self._flatten(params)
        mlflow.log_params(flat)
        logger.debug(f"[MLflow] Params logged: {flat}")

    def log_metrics(self, metrics: dict, step: int = None):
        """Logue les métriques numériques."""
        clean = {k: float(v) for k, v in metrics.items() if v is not None}
        mlflow.log_metrics(clean, step=step)
        logger.debug(f"[MLflow] Metrics logged: {clean}")

    def log_tags(self, tags: dict):
        """Logue des tags informatifs."""
        mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def log_model(self, model, artifact_path: str = "model"):
        """Enregistre le modèle sklearn dans MLflow."""
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            registered_model_name=self.cfg.mlflow.registered_model,
        )
        logger.info(f"[MLflow] Model logged → artifact_path={artifact_path}")

    def log_feature_pipeline(self, pipeline_path: str):
        """Logue le pipeline de features comme artifact."""
        mlflow.log_artifact(pipeline_path, artifact_path="pipeline")
        logger.info(f"[MLflow] Feature pipeline logged → {pipeline_path}")

    def log_confusion_matrix(self, cm: np.ndarray, classes: list):
        """Génère et logue une image de la matrice de confusion."""
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")

        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        short = [c.replace("_", "\n") for c in classes]

        sns.heatmap(
            cm_norm,
            ax=ax,
            annot=True,
            fmt=".0%",
            cmap="Blues",
            xticklabels=short,
            yticklabels=short,
            linewidths=0.5,
            linecolor="#0d1117",
            annot_kws={"size": 9},
        )
        ax.set_title("Confusion Matrix (Normalized)", color="white")
        ax.tick_params(colors="#8b949e", labelsize=8)
        ax.set_xlabel("Predicted", color="#8b949e")
        ax.set_ylabel("Actual", color="#8b949e")

        cm_path = Path("data/processed/mlflow_confusion_matrix.png")
        plt.savefig(cm_path, dpi=120, bbox_inches="tight", facecolor="#0d1117")
        plt.close()

        mlflow.log_artifact(str(cm_path), artifact_path="plots")
        logger.info("[MLflow] Confusion matrix logged")

    def log_classification_report(self, report: str):
        """Logue le rapport de classification comme fichier texte."""
        report_path = Path("data/processed/classification_report.txt")
        report_path.write_text(report, encoding="utf-8")
        mlflow.log_artifact(str(report_path), artifact_path="reports")

    def log_system_tags(self):
        """Logue les informations système pour la reproductibilité."""
        tags = {
            "python_version": platform.python_version(),
            "platform": platform.system(),
            "hostname": platform.node(),
        }
        # Git commit hash si disponible
        try:
            commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
            tags["git_commit"] = commit
        except Exception:
            tags["git_commit"] = "unknown"

        self.log_tags(tags)

    # ── Full run helper ───────────────────────────────────────────────────────

    def log_full_run(self, artifacts, run_name: str | None = None) -> str:
        """
        Logue un run complet depuis un objet PipelineArtifacts.
        Retourne le run_id MLflow.
        """
        name = run_name or artifacts.model_name

        with self.start_run(name):
            # Params
            self.log_params(artifacts.params)
            self.log_params(
                {
                    "n_features": getattr(artifacts.feature_pipeline, "n_features", 0),
                    "cv_folds": self.cfg.model.cv_folds,
                    "tfidf_vocab": self.cfg.features.tfidf_max_features,
                }
            )

            # Métriques
            self.log_metrics(artifacts.metrics)
            self.log_metrics({"train_time_s": artifacts.train_time_s})

            # Tags système
            self.log_system_tags()
            self.log_tags(
                {
                    "model_class": artifacts.model_name,
                    "n_classes": len(artifacts.classes),
                    "classes": str(artifacts.classes),
                }
            )

            # Artifacts
            self.log_model(artifacts.model)
            self.log_feature_pipeline("data/processed/feature_pipeline.pkl")
            self.log_confusion_matrix(artifacts.confusion_mat, artifacts.classes)
            self.log_classification_report(artifacts.report)

            run_id = mlflow.active_run().info.run_id
            logger.info(f"[MLflow] ✅ Full run logged → run_id={run_id}")

        return run_id

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
        """Aplatit un dict nested pour MLflow (pas de dicts imbriqués)."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(MLflowTracker._flatten(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    @property
    def is_connected(self) -> bool:
        return self._connected
