"""
scripts/run_experiment.py
Lance un run complet avec tracking MLflow + persistance PostgreSQL.

Usage :
    uv run python scripts/run_experiment.py
    uv run python scripts/run_experiment.py --no-tune
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.utils.logger import get_logger
from src.pipeline.training_pipeline import TrainingPipeline
from src.tracking.mlflow_tracker import MLflowTracker
from src.tracking.database import DatabaseManager

logger = get_logger("run_experiment")


def parse_args():
    parser = argparse.ArgumentParser(
        description="NOTAM — Full MLflow Experiment Run"
    )
    parser.add_argument("--no-tune", action="store_true",
                        help="Désactive GridSearchCV")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Nom custom du run MLflow")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = Config.get()

    logger.info("=" * 60)
    logger.info("  NOTAM Classification — MLflow Experiment")
    logger.info("=" * 60)
    logger.info(f"  Project  : {cfg.project.name} v{cfg.project.version}")
    logger.info(f"  Tune     : {not args.no_tune}")

    # ── 1. Entraînement ───────────────────────────────────────────────────────
    pipeline  = TrainingPipeline(cfg=cfg)
    artifacts = pipeline.run(tune=not args.no_tune)

    # ── 2. MLflow tracking ────────────────────────────────────────────────────
    tracker = MLflowTracker(cfg=cfg)
    run_id  = tracker.log_full_run(
        artifacts,
        run_name=args.run_name or artifacts.model_name
    )

    # ── 3. PostgreSQL / SQLite persistence ────────────────────────────────────
    db = DatabaseManager.get_instance()
    db.save_experiment_run(run_id, artifacts)

    # ── 4. Résumé ─────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  ✅ Experiment Complete")
    logger.info("=" * 60)
    logger.info(f"  MLflow Run ID : {run_id}")
    logger.info(f"  Model         : {artifacts.model_name}")
    logger.info(f"  F1-macro      : {artifacts.metrics['test_f1_macro']:.4f}")
    logger.info(f"  Accuracy      : {artifacts.metrics['test_accuracy']:.4f}")

    if tracker.is_connected:
        logger.info(f"  MLflow UI     : {cfg.mlflow.tracking_uri}")
    else:
        logger.info("  MLflow UI     : uv run mlflow ui --port 5000")

    return 0


if __name__ == "__main__":
    sys.exit(main())