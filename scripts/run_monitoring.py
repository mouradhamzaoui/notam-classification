"""
scripts/run_monitoring.py
Lance une analyse de dérive complète en ligne de commande.

Usage :
    uv run python scripts/run_monitoring.py
    uv run python scripts/run_monitoring.py --drift
    uv run python scripts/run_monitoring.py --n 300
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from src.monitoring.drift_detector import NOTAMDriftDetector

logger = get_logger("monitoring")


def main():
    parser = argparse.ArgumentParser(description="NOTAM Drift Monitoring")
    parser.add_argument("--drift", action="store_true",
                        help="Simule une dérive artificielle")
    parser.add_argument("--n", type=int, default=200,
                        help="Nombre de prédictions production à simuler")
    args = parser.parse_args()

    logger.info("=" * 55)
    logger.info("  NOTAM Classification — Drift Monitoring")
    logger.info("=" * 55)

    # ── Init ──────────────────────────────────────────────────────────────────
    detector = NOTAMDriftDetector()
    detector.load_reference_from_csv("data/processed/notams_clean.csv")

    # ── Production data ───────────────────────────────────────────────────────
    current_df = detector.generate_synthetic_production_data(
        n=args.n, drift=args.drift
    )
    logger.info(f"Production data: {len(current_df)} rows "
                f"({'WITH drift' if args.drift else 'no drift'})")

    # ── Drift report ──────────────────────────────────────────────────────────
    metrics = detector.run_data_drift_report(current_df, save=True)
    logger.info(f"Drift metrics: {metrics}")

    # ── Test suite ────────────────────────────────────────────────────────────
    tests = detector.run_test_suite(current_df, save=True)
    logger.info(f"Test suite: {tests}")

    # ── Alert ─────────────────────────────────────────────────────────────────
    alert = detector.check_alert(metrics)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("  📊 MONITORING SUMMARY")
    logger.info("=" * 55)
    logger.info(f"  Dataset drift    : {'⚠️  YES' if metrics['dataset_drift'] else '✅ NO'}")
    logger.info(f"  Drifted features : {metrics['n_drifted']}/{metrics['n_features']}")
    logger.info(f"  Drift share      : {metrics['drift_share']:.0%}")
    logger.info(f"  Tests passed     : {tests['passed']}/{tests['total']}")
    logger.info(f"  ALERT            : {'🔴 YES' if alert else '🟢 NO'}")
    logger.info(f"\n  Reports saved to : reports/drift/")

    return 1 if alert else 0


if __name__ == "__main__":
    sys.exit(main())