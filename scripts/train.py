"""
scripts/train.py
Point d'entrée CLI pour lancer le pipeline d'entraînement.

Usage :
    uv run python scripts/train.py
    uv run python scripts/train.py --no-tune
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from src.utils.config import Config
from src.pipeline.training_pipeline import TrainingPipeline

logger = get_logger("train")


def parse_args():
    parser = argparse.ArgumentParser(description="NOTAM Classification — Training Pipeline")
    parser.add_argument("--no-tune", action="store_true",
                        help="Désactive le GridSearchCV (plus rapide)")
    parser.add_argument("--config", type=str, default=None,
                        help="Chemin vers un config.yaml alternatif")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("=" * 55)
    logger.info("  NOTAM Classification — Training Pipeline")
    logger.info("=" * 55)

    cfg = Config.get()
    logger.info(f"Project : {cfg.project.name} v{cfg.project.version}")

    pipeline   = TrainingPipeline(cfg=cfg)
    artifacts  = pipeline.run(tune=not args.no_tune)

    logger.info("✅ Training complete!")
    logger.info(f"   Model     : {artifacts.model_name}")
    logger.info(f"   F1-macro  : {artifacts.metrics['test_f1_macro']:.4f}")
    logger.info(f"   Accuracy  : {artifacts.metrics['test_accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())