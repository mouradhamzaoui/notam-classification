"""
routers/classify.py
Endpoints de classification NOTAM.
"""

import time
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.schemas import (
    ClassifyRequest, ClassifyResponse,
    BatchClassifyRequest, BatchClassifyResponse,
    FeedbackRequest,
)
from src.api.dependencies import get_inference_pipeline, get_db, get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/classify", tags=["Classification"])


@router.post(
    "",
    response_model=ClassifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify a NOTAM",
    description="""
Classifie un NOTAM texte brut vers une catégorie ICAO.

**Catégories supportées :**
- `RUNWAY_CLOSURE` — Fermeture de piste
- `NAVIGATION_AID` — Panne aide à la navigation
- `AIRSPACE_RESTRICTION` — Restriction d'espace aérien
- `LIGHTING` — Panne éclairage
- `OBSTACLE` — Nouvel obstacle
- `AERODROME_PROCEDURE` — Procédure aérodrome
    """,
)
async def classify_notam(
    request:  ClassifyRequest,
    pipeline = Depends(get_inference_pipeline),
    db       = Depends(get_db),
    cfg      = Depends(get_config),
):
    try:
        result = pipeline.predict(request.text)
    except Exception as e:
        logger.error(f"[API] Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )

    # Log en base de données (non-bloquant)
    try:
        db.log_prediction(result, model_version=request.model_version)
    except Exception as e:
        logger.warning(f"[API] DB log failed (non-critical): {e}")

    # Vérification du seuil de confiance
    min_conf = float(cfg.thresholds.min_confidence or 0.60)
    if result.confidence < min_conf:
        logger.warning(
            f"[API] Low confidence: {result.confidence:.2%} < {min_conf:.0%} "
            f"for text: {request.text[:50]}..."
        )

    return ClassifyResponse(
        category      = result.category,
        confidence    = result.confidence,
        probabilities = result.probabilities,
        latency_ms    = result.latency_ms,
        priority      = result.meta.get("priority", "UNKNOWN"),
        icon          = result.meta.get("icon", "✈️"),
        model_version = request.model_version,
    )


@router.post(
    "/batch",
    response_model=BatchClassifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch classify NOTAMs",
    description="Classifie une liste de NOTAMs en une seule requête (max 100).",
)
async def batch_classify(
    request:  BatchClassifyRequest,
    pipeline = Depends(get_inference_pipeline),
    db       = Depends(get_db),
):
    if len(request.texts) > 100:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Maximum 100 NOTAMs par requête batch",
        )

    t0 = time.time()
    responses = []

    try:
        results = pipeline.predict_batch(request.texts)
    except Exception as e:
        logger.error(f"[API] Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {str(e)}",
        )

    for result in results:
        try:
            db.log_prediction(result, model_version=request.model_version)
        except Exception:
            pass

        responses.append(ClassifyResponse(
            category      = result.category,
            confidence    = result.confidence,
            probabilities = result.probabilities,
            latency_ms    = result.latency_ms,
            priority      = result.meta.get("priority", "UNKNOWN"),
            icon          = result.meta.get("icon", "✈️"),
            model_version = request.model_version,
        ))

    duration_ms = (time.time() - t0) * 1000
    logger.info(
        f"[API] Batch: {len(results)} NOTAMs classified "
        f"in {duration_ms:.1f}ms"
    )

    return BatchClassifyResponse(
        results     = responses,
        total       = len(responses),
        duration_ms = duration_ms,
    )


@router.post(
    "/feedback",
    status_code=status.HTTP_200_OK,
    summary="Submit prediction feedback",
    description="Soumet un feedback expert sur une prédiction (pour le retraining).",
)
async def submit_feedback(
    request: FeedbackRequest,
    db      = Depends(get_db),
):
    try:
        with db.session() as session:
            from src.tracking.database import PredictionLog
            log = session.query(PredictionLog).filter(
                PredictionLog.id == request.prediction_id
            ).first()

            if not log:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Prediction {request.prediction_id} not found",
                )

            log.true_label = request.true_label
            log.is_correct = (log.predicted == request.true_label)
            session.commit()

        return {
            "message":    "Feedback enregistré",
            "id":         request.prediction_id,
            "is_correct": log.is_correct,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )