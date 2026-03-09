"""
routers/monitoring.py
Endpoints de monitoring et statistiques des prédictions.
"""

from fastapi import APIRouter, Depends, Query
from src.api.schemas import PredictionLogResponse
from src.api.dependencies import get_db
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


@router.get(
    "/predictions",
    response_model=list[PredictionLogResponse],
    summary="Recent Predictions",
    description="Retourne les N dernières prédictions loggées.",
)
async def get_recent_predictions(
    limit: int = Query(default=50, ge=1, le=500,
                       description="Nombre de prédictions à retourner"),
    db = Depends(get_db),
):
    try:
        logs = db.get_recent_predictions(limit=limit)
        return logs
    except Exception as e:
        logger.error(f"[Monitoring] Error fetching predictions: {e}")
        return []


@router.get(
    "/stats",
    summary="Prediction Statistics",
    description="Statistiques agrégées sur les prédictions récentes.",
)
async def get_stats(
    limit: int = Query(default=500, ge=1, le=5000),
    db = Depends(get_db),
):
    try:
        from collections import Counter
        logs = db.get_recent_predictions(limit=limit)

        if not logs:
            return {"message": "Aucune prédiction en base", "total": 0}

        categories = [l.predicted   for l in logs]
        confidences= [l.confidence  for l in logs]
        latencies  = [l.latency_ms  for l in logs if l.latency_ms]

        cat_counts = Counter(categories)
        correct    = sum(1 for l in logs if l.is_correct is True)
        total_fb   = sum(1 for l in logs if l.is_correct is not None)

        return {
            "total_predictions":   len(logs),
            "category_distribution": dict(cat_counts),
            "avg_confidence":      round(sum(confidences)/len(confidences), 4),
            "min_confidence":      round(min(confidences), 4),
            "max_confidence":      round(max(confidences), 4),
            "avg_latency_ms":      round(sum(latencies)/len(latencies), 2) if latencies else None,
            "feedback_accuracy":   round(correct/total_fb, 4) if total_fb > 0 else None,
            "total_feedback":      total_fb,
        }
    except Exception as e:
        logger.error(f"[Monitoring] Stats error: {e}")
        return {"error": str(e)}